"""Fast checks for the optional FSQ/WAM integration."""

import torch

from gear_sonic.trl.world_model import (
    FSQObservationAdapter,
    LatentMPC,
    ModelBasedActorCriticTrainer,
    TransitionReplayBuffer,
    WAM,
)


def test_fsq_multimodal_adapter_broadcasts_temporal_tokens():
    adapter = FSQObservationAdapter(token_key="tokens")
    tokens = torch.randn(2, 4, 2, 8)
    proprio = torch.randn(2, 6)
    features = adapter({"tokens": tokens, "proprio": proprio})
    assert features.shape == (2, 4, 22)


def test_wam_loss_imagination_and_mpc():
    model = WAM(obs_dim=6, action_dim=2, latent_dim=8, hidden_dim=16)
    obs = torch.randn(3, 6)
    next_obs = torch.randn(3, 6)
    action = torch.randn(3, 2)
    losses = model.loss(
        {"obs": obs, "next_obs": next_obs, "action": action, "reward": torch.randn(3)}
    )
    assert losses["loss"].ndim == 0
    actor = torch.nn.Linear(8, 2)
    rollout = model.imagine(obs, actor, horizon=3)
    assert rollout["states"].shape == (3, 3, 8)
    assert rollout["next_states"].shape == (3, 3, 8)
    planner = LatentMPC(model, horizon=2, population=8, elite_count=2, iterations=1)
    assert planner(obs).shape == (3, 2)


def test_replay_and_model_based_update():
    model = WAM(obs_dim=4, action_dim=1, latent_dim=6, hidden_dim=12)
    actor = torch.nn.Linear(6, 1)
    buffer = TransitionReplayBuffer(capacity=4)
    for _ in range(4):
        state = torch.randn(4)
        buffer.add(state, torch.randn(1), 0.1, torch.randn(4))
    batch = buffer.sample(2)
    trainer = ModelBasedActorCriticTrainer(model, actor, replay_buffer=buffer, horizon=2)
    metrics = trainer.update(torch.randn(2, 4), replay_batch=batch)
    assert "actor/loss" in metrics


def test_single_transition_and_window_accept_batch_one_labels():
    model = WAM(obs_dim=4, action_dim=1, latent_dim=6, hidden_dim=12)
    one_step = model.loss(
        {
            "obs": torch.randn(1, 4),
            "next_obs": torch.randn(1, 4),
            "action": torch.randn(1, 1),
            "reward": torch.tensor([0.5]),
            "done": torch.tensor([1.0]),
        }
    )
    assert torch.isfinite(one_step["loss"])
    window = model.window_loss(
        {
            "obs": torch.randn(1, 4, 4),
            "action": torch.randn(1, 3, 1),
            "reward": torch.randn(1, 3),
            "done": torch.zeros(1, 3),
        }
    )
    assert torch.isfinite(window["loss"])
    window["loss"].backward()
    # Error estimates have their own supervised objective, rather than an
    # untrained random head that merely supplies a planning penalty.
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.uncertainty_head.parameters()
    )


def test_lazy_imagination_updates_actor_without_world_model_gradients():
    torch.manual_seed(3)
    model = WAM(obs_dim=None, action_dim=1, latent_dim=6, hidden_dim=12)
    actor = torch.nn.Linear(6, 1)
    trainer = ModelBasedActorCriticTrainer(model, actor, horizon=3)
    old_weight = actor.weight.detach().clone()
    # No replay warmup: this materializes LazyLinear during imagination.
    metrics = trainer.update(torch.randn(2, 4))
    assert torch.isfinite(metrics["actor/loss"])
    assert not torch.equal(old_weight, actor.weight)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
    # The materialized encoder remains trainable on the next replay fit.
    trainer.world_model_update(
        {
            "obs": torch.randn(2, 4),
            "next_obs": torch.randn(2, 4),
            "action": torch.randn(2, 1),
            "reward": torch.randn(2),
            "done": torch.zeros(2),
        }
    )
    assert all(parameter.grad is not None for parameter in model.obs_encoder.parameters())


def test_default_critic_matches_world_model_device_and_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WAM(obs_dim=4, action_dim=1, latent_dim=6, hidden_dim=12).to(
        device=device, dtype=torch.float64
    )
    actor = torch.nn.Linear(6, 1).to(device=device, dtype=torch.float64)
    trainer = ModelBasedActorCriticTrainer(model, actor, horizon=2)
    critic_parameter = next(trainer.critic.parameters())
    assert critic_parameter.device == device
    assert critic_parameter.dtype == torch.float64
    metrics = trainer.update(torch.randn(2, 4, device=device, dtype=torch.float64))
    assert torch.isfinite(metrics["critic/loss"])


if __name__ == "__main__":
    for name, test in sorted(list(globals().items())):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
