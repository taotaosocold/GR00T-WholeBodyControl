"""CPU/Gloo checks for the WAM data-parallel training path.

Each test launches two real processes with different seeds and trajectories.
These checks exercise collectives and checkpoint ownership, rather than
mocking a distributed accelerator in a single process.
"""

from datetime import timedelta
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from gear_sonic.tests.test_wam_training import (
    Config,
    VectorEnv,
    make_original_fsq_actor,
)
from gear_sonic.trl.modules.wam import RunningNorm
from gear_sonic.trl.trainer.wam_trainer import WAMTrainer
from gear_sonic.trl.world_model.distributed import DistributedContext


def _assert_equal_on_ranks(module):
    for name, value in module.state_dict().items():
        copies = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(copies, value.contiguous())
        for other in copies[1:]:
            torch.testing.assert_close(copies[0], other, rtol=0, atol=0, msg=name)


def _make_trainer(directory, rank, checkpoint=None, resume=False):
    accelerator = SimpleNamespace(
        num_processes=dist.get_world_size(), process_index=rank,
        local_process_index=rank, is_main_process=(rank == 0),
        device=torch.device("cpu"), wait_for_everyone=dist.barrier,
    )
    env = VectorEnv()
    # Distinct per-rank observations ensure all-reduced gradients/normalizers
    # are needed: equal results cannot arise from identical local minibatches.
    env.x += rank * 0.75
    env.goals += rank * 0.5
    # Modality coverage is deliberately different on each rank. This exercises
    # unused encoder gradients through the original UniversalTokenModule, in
    # addition to the isolated sparse-branch test below.
    modes = torch.zeros(env.num_envs, dtype=torch.long) if rank == 0 else (
        1 + torch.arange(env.num_envs) % 2
    )
    env.routing = torch.eye(3)[modes]
    return WAMTrainer(
        args=SimpleNamespace(num_total_batches=1),
        config=Config(gamma=0.99, lam=0.95),
        env=env, model=make_original_fsq_actor(), accelerator=accelerator,
        log_dir=directory, checkpoint=checkpoint, resume=resume,
        settings={
            "window": 3, "horizon": 3, "batch_size": 12,
            "world_updates": 2, "critic_epochs": 2, "replay_steps": 16,
            "collect_steps": 6, "warmup_world_updates": 1,
            "save_replay": True,
            "exogenous_tokenizer_features": ["g1_goal", "teleop_goal", "smpl_goal"],
        },
        world_model_config={
            "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
            "cfg": {"hidden": [16, 16], "term_hidden": [8], "reward_hidden": [8]},
        },
    )


def _exercise_training(directory, rank):
    trainer = _make_trainer(directory, rank)
    _assert_equal_on_ranks(trainer.policy)
    _assert_equal_on_ranks(trainer.world_model)
    _assert_equal_on_ranks(trainer.critic)
    trainer.collect(6)
    _assert_equal_on_ranks(trainer.world_model)
    trainer.fit_world_model()
    _assert_equal_on_ranks(trainer.world_model)
    trainer.improve_policy()
    _assert_equal_on_ranks(trainer.policy)
    _assert_equal_on_ranks(trainer.world_model)
    _assert_equal_on_ranks(trainer.critic)

    # Independent replay stays different while the learned models agree.
    local_replay = trainer.replay.state_dict()["storage"]["state"].clone()
    replay_copies = [torch.empty_like(local_replay) for _ in range(dist.get_world_size())]
    dist.all_gather(replay_copies, local_replay)
    assert not torch.equal(replay_copies[0], replay_copies[1])

    path = Path(directory) / "last.pt"
    # Record writes in each process. A non-main rank may write its own replay
    # shard, but must not race rank zero for the shared model checkpoint.
    writes = []
    original_save = torch.save

    def tracked_save(obj, destination, *args, **kwargs):
        writes.append(str(destination))
        return original_save(obj, destination, *args, **kwargs)

    torch.save = tracked_save
    try:
        trainer.save_checkpoint(path)
    finally:
        torch.save = original_save
    dist.barrier()
    all_writes = [None] * dist.get_world_size()
    dist.all_gather_object(all_writes, writes)
    shared_writes = set(all_writes[0])
    assert shared_writes, "rank zero did not write the shared checkpoint"
    assert not shared_writes.intersection(all_writes[1]), "ranks raced for the same checkpoint"
    assert path.exists()

    restored = _make_trainer(directory, rank, checkpoint=path, resume=True)
    for name, expected in trainer.policy.state_dict().items():
        torch.testing.assert_close(expected, restored.policy.state_dict()[name], rtol=0, atol=0)
    for name, expected in trainer.world_model.state_dict().items():
        torch.testing.assert_close(expected, restored.world_model.state_dict()[name], rtol=0, atol=0)
    restored_replay = restored.replay.state_dict()["storage"]["state"]
    torch.testing.assert_close(local_replay, restored_replay, rtol=0, atol=0)
    restored.collect(3)
    restored.fit_world_model()
    restored.improve_policy()
    _assert_equal_on_ranks(restored.policy)
    _assert_equal_on_ranks(restored.world_model)
    _assert_equal_on_ranks(restored.critic)
    # Also cover the orchestration path: readiness consensus, reduced metrics,
    # rank-zero logging, and the final checkpoint invoked by train().
    restored.train()
    assert restored.iteration == 1
    trained = _make_trainer(directory, rank, checkpoint=path, resume=True)
    assert trained.iteration == 1
    _assert_equal_on_ranks(trained.policy)
    _assert_equal_on_ranks(trained.world_model)
    _assert_equal_on_ranks(trained.critic)


def _exercise_sparse_gradients(rank):
    context = DistributedContext(device="cpu")
    module = nn.ModuleDict({name: nn.Linear(2, 1, bias=False) for name in ("a", "b", "unused")})
    context.broadcast_modules(module)
    # Only one branch participates locally. The other branch must receive the
    # remote gradient, and a branch unused everywhere must retain grad=None.
    branch = "a" if rank == 0 else "b"
    module[branch](torch.full((1, 2), float(rank + 1))).sum().backward()
    context.average_gradients(module.parameters())
    torch.testing.assert_close(module["a"].weight.grad, torch.full((1, 2), 0.5))
    torch.testing.assert_close(module["b"].weight.grad, torch.ones(1, 2))
    assert module["unused"].weight.grad is None
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    _assert_equal_on_ranks(module)


def _exercise_normalizers(rank):
    context = DistributedContext(device="cpu")
    norm = RunningNorm(2)
    before = context.snapshot_running_norms([norm])
    first = torch.tensor([[1., 3.], [3., 5.]]) if rank == 0 else torch.tensor([[5., 7.]])
    norm.update(first)
    context.sync_running_norms([norm], before)
    torch.testing.assert_close(norm.count, torch.tensor(3.))
    torch.testing.assert_close(norm.mean, torch.tensor([3., 5.]))
    torch.testing.assert_close(norm.var, torch.full((2,), 8. / 3.))
    before = context.snapshot_running_norms([norm])
    second = torch.tensor([[7., 9.]]) if rank == 0 else torch.tensor([[9., 11.], [11., 13.]])
    norm.update(second)
    context.sync_running_norms([norm], before)
    # The common first-round statistics must only be counted once.
    torch.testing.assert_close(norm.count, torch.tensor(6.))
    torch.testing.assert_close(norm.mean, torch.tensor([6., 8.]))
    torch.testing.assert_close(norm.var, torch.full((2,), 35. / 3.))
    before = context.snapshot_running_norms([norm])
    if rank == 1:
        norm.update(torch.tensor([[13., 15.]]))
    context.sync_running_norms([norm], before)
    torch.testing.assert_close(norm.count, torch.tensor(7.))
    torch.testing.assert_close(norm.mean, torch.tensor([7., 9.]))
    torch.testing.assert_close(norm.var, torch.full((2,), 16.))
    _assert_equal_on_ranks(norm)
    metrics = context.reduce_metrics({"reward": rank + 1., "elapsed": torch.tensor(rank + 2.)})
    assert metrics == {"reward": 1.5, "elapsed": 2.5}


class _BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, corrupt):
        ctx.corrupt = corrupt
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        if ctx.corrupt:
            gradient = torch.full_like(gradient, float("nan"))
        return gradient, None


def _exercise_nonfinite(rank):
    context = DistributedContext(device="cpu")
    model = nn.Linear(2, 1)
    context.broadcast_modules(model)
    optimizer = torch.optim.Adam(model.parameters())
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for corrupt_gradient in (False, True):
        loss = model(torch.ones(1, 2)).sum()
        if corrupt_gradient:
            loss = _BadGradient.apply(loss, rank == 1)
        elif rank == 1:
            loss = loss * float("nan")
        rejected = False
        try:
            WAMTrainer.optimizer_step(loss, optimizer, model.parameters(), distributed=context)
        except FloatingPointError:
            rejected = True
        assert rejected, "every rank must reject a non-finite loss or gradient from any rank"
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    _assert_equal_on_ranks(model)


def _worker(rank, world_size, rendezvous, directory, scenario):
    torch.set_num_threads(1)
    torch.manual_seed(17 + 113 * rank)
    dist.init_process_group(
        backend="gloo", init_method=Path(rendezvous).resolve().as_uri(),
        rank=rank, world_size=world_size, timeout=timedelta(seconds=30),
    )
    try:
        if scenario == "training":
            _exercise_training(directory, rank)
        elif scenario == "gradients":
            _exercise_sparse_gradients(rank)
        elif scenario == "normalizers":
            _exercise_normalizers(rank)
        elif scenario == "nonfinite":
            _exercise_nonfinite(rank)
        else:
            raise AssertionError(f"Unknown distributed test scenario: {scenario}")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires torch Gloo")
class DistributedWAMTests(unittest.TestCase):
    def launch(self, scenario):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "gloo-init")
            processes = mp.spawn(
                _worker, args=(2, rendezvous, directory, scenario), nprocs=2, join=False,
            )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if processes.join(timeout=1):
                    return
            for process in processes.processes:
                process.terminate()
            for process in processes.processes:
                process.join(timeout=5)
            self.fail("distributed WAM test timed out (possible collective mismatch)")

    def test_real_fsq_training_and_rank_local_checkpoint_restore(self):
        self.launch("training")

    def test_parameters_unused_on_one_or_all_ranks(self):
        self.launch("gradients")

    def test_incremental_normalizers_and_reduced_metrics(self):
        self.launch("normalizers")

    def test_nonfinite_loss_and_gradient_rejected_on_all_ranks(self):
        self.launch("nonfinite")


if __name__ == "__main__":
    unittest.main()
