"""Explicit data parallelism for alternating world-model / FSQ BPTT updates.

The trainer executes many policy forwards before one backward pass and freezes
different modules in each phase. Reducing the finished gradients explicitly
avoids attaching DDP hooks to those alternating autograd graphs. All ranks must
call the collective methods in the same order; simulation and replay remain
local to each rank.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn


class DistributedContext:
    """Use an already-initialized process group, or act as a single-rank no-op.

    Accelerator is responsible for process-group initialization and selection
    of each CUDA device. The helper never creates a second process group.
    ``device`` selects the device for small consensus/control tensors; pass the
    trainer's device when using NCCL.
    """

    def __init__(self, device: torch.device | str | None = None):
        self.enabled = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.enabled else 0
        self.world_size = dist.get_world_size() if self.enabled else 1
        self.is_main = self.rank == 0
        if device is None:
            backend = str(dist.get_backend()) if self.enabled else ""
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if "nccl" in backend
                else torch.device("cpu")
            )
        self.device = torch.device(device)

    @torch.no_grad()
    def broadcast_modules(self, *modules: nn.Module) -> None:
        """Make initial parameters and buffers identical despite per-rank seeds."""
        if self.world_size == 1:
            return
        for module in modules:
            for parameter in module.parameters():
                self._broadcast_tensor(parameter.data)
            for buffer in module.buffers():
                self._broadcast_tensor(buffer)

    @staticmethod
    def _broadcast_tensor(tensor: Tensor) -> None:
        # NCCL requires a dense contiguous communication buffer. Preserve the
        # registered tensor object (and parameter layout) when copying a view.
        communication = tensor if tensor.is_contiguous() else tensor.contiguous()
        dist.broadcast(communication, src=0)
        if communication is not tensor:
            tensor.copy_(communication)

    def check_consistent(self, value, label: str = "metadata") -> None:
        """Fail on all ranks when small, plain-Python metadata disagree.

        Use before broadcasting modules or entering phase loops to validate
        observation layouts, module shapes, and iteration settings. ``value``
        should contain ordinary Python scalars/containers, not tensor values.
        """
        if self.world_size == 1:
            return
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        if any(item != gathered[0] for item in gathered[1:]):
            raise ValueError(f"Distributed {label} differs across ranks")

    @torch.no_grad()
    def all_finite(self, value: Tensor | bool) -> bool:
        """Return false on every rank if any rank has a non-finite value.

        A boolean can represent the result of a more involved local check.
        Call this before backward for losses and before stepping for gradients
        so a failed rank cannot leave its peers in a collective or update only
        a subset of model replicas.
        """
        if isinstance(value, Tensor):
            result = value.detach().all() if value.dtype == torch.bool else torch.isfinite(value).all()
            finite = result.to(device=self.device)
        else:
            finite = torch.tensor(bool(value), device=self.device)
        flag = finite.to(dtype=torch.int32)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @torch.no_grad()
    def average_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        """Average dense gradients while handling rank-specific unused encoders.

        A modality encoder may have no local samples, hence ``grad is None`` on
        one rank and a gradient on another. A shared presence vector determines
        the same collective schedule on every rank. Missing local gradients
        become zeros only if the parameter is used somewhere; globally unused
        parameters retain ``None`` so optimizers do not update their moments.
        """
        if self.world_size == 1:
            return
        parameters = list(parameters)
        if not parameters:
            return
        presence = torch.tensor(
            [parameter.grad is not None for parameter in parameters],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(presence, op=dist.ReduceOp.SUM)
        used = presence.cpu().tolist()
        sparse = any(
            parameter.grad is not None and parameter.grad.is_sparse
            for parameter in parameters
        )
        sparse_flag = torch.tensor(int(sparse), dtype=torch.int32, device=self.device)
        dist.all_reduce(sparse_flag, op=dist.ReduceOp.MAX)
        if sparse_flag.item():
            raise ValueError("WAM distributed updates require dense gradients")

        groups: dict[tuple[torch.device, torch.dtype], list[nn.Parameter]] = defaultdict(list)
        for parameter, count in zip(parameters, used):
            if count == 0:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            groups[(parameter.device, parameter.dtype)].append(parameter)

        # Bound temporary allocations for large FSQ decoders. One oversized
        # parameter is reduced alone; ordinary parameters share 25 MiB buckets.
        for (_, dtype), group in groups.items():
            limit = max(1, (25 * 1024 * 1024) // torch.empty((), dtype=dtype).element_size())
            bucket, size = [], 0
            for parameter in group:
                if bucket and size + parameter.numel() > limit:
                    self._reduce_bucket(bucket)
                    bucket, size = [], 0
                bucket.append(parameter)
                size += parameter.numel()
            if bucket:
                self._reduce_bucket(bucket)

    def _reduce_bucket(self, parameters: list[nn.Parameter]) -> None:
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count

    @staticmethod
    def snapshot_running_norms(norms: Iterable[nn.Module]) -> list[dict[str, Tensor]]:
        """Capture globally shared statistics immediately before local collection."""
        return [
            {
                "count": norm.count.detach().double().clone(),
                "mean": norm.mean.detach().double().clone(),
                "var": norm.var.detach().double().clone(),
            }
            for norm in norms
        ]

    @torch.no_grad()
    def sync_running_norms(
        self, norms: Iterable[nn.Module], before_snapshot: list[dict[str, Tensor]]
    ) -> None:
        """Merge new local samples without counting shared history repeatedly.

        Each rank starts from the same count/mean/variance, updates its own
        ``RunningNorm`` during collection, and supplies that starting snapshot.
        Subtracting the old sufficient statistics isolates the local increment.
        Only those increments are summed across ranks; the common prior is
        added exactly once. All arithmetic uses float64 before copying back to
        each normalizer's original dtype.
        """
        if self.world_size == 1:
            return
        norms = list(norms)
        if len(norms) != len(before_snapshot):
            raise ValueError("RunningNorm snapshots must match the normalizer list")
        for norm, before in zip(norms, before_snapshot):
            old_n = before["count"].to(norm.count.device)
            old_mean = before["mean"].to(norm.mean.device)
            old_var = before["var"].to(norm.var.device)
            new_n = norm.count.double()
            new_mean, new_var = norm.mean.double(), norm.var.double()
            local_n = (new_n - old_n).clamp_min(0)
            first = new_n * new_mean - old_n * old_mean
            second = new_n * (new_var + new_mean.square()) - old_n * (
                old_var + old_mean.square()
            )
            # No local sample can change the statistics; explicitly zeroing
            # these differences also avoids roundoff in a no-update rank.
            if local_n.item() == 0:
                first.zero_()
                second.zero_()
            packed = torch.cat([local_n.reshape(1), first.reshape(-1), second.reshape(-1)])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            width = norm.mean.numel()
            count = old_n + packed[0]
            if count.item() == 0:
                continue
            mean = (old_n * old_mean + packed[1:1 + width].view_as(old_mean)) / count
            second_moment = (
                old_n * (old_var + old_mean.square())
                + packed[1 + width:].view_as(old_mean)
            ) / count
            variance = (second_moment - mean.square()).clamp_min(0)
            norm.count.copy_(count.to(norm.count.dtype))
            norm.mean.copy_(mean.to(norm.mean.dtype))
            norm.var.copy_(variance.to(norm.var.dtype))

    @torch.no_grad()
    def reduce_metrics(self, metrics: Mapping[str, float | Tensor]) -> dict[str, float]:
        """Average available scalar metrics across ranks, allowing missing keys."""
        local_keys = sorted(metrics)
        if self.world_size == 1:
            return {key: float(metrics[key]) for key in local_keys}
        gathered_keys: list[list[str] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_keys, local_keys)
        keys = sorted({key for group in gathered_keys for key in group})
        if not keys:
            return {}
        values = torch.tensor(
            [float(metrics.get(key, 0.0)) for key in keys], dtype=torch.float64,
            device=self.device,
        )
        counts = torch.tensor(
            [float(key in metrics) for key in keys], dtype=torch.float64, device=self.device
        )
        packed = torch.stack([values, counts])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        means = (packed[0] / packed[1].clamp_min(1)).cpu().tolist()
        return dict(zip(keys, means))


__all__ = ["DistributedContext"]
