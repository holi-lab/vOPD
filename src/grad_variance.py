from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch


__all__ = [
    "adam_moment_variance",
    "empirical_total_variance",
    "gradient_variance_ratio",
    "unwrap_optimizer",
    "GradientVarianceTracker",
    "TotalVarianceAccumulator",
]


def unwrap_optimizer(optimizer: Any) -> Any:
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        state = getattr(current, "state", None)
        if isinstance(state, dict) and any(
            isinstance(entry, dict) and "exp_avg" in entry for entry in state.values()
        ):
            return current
        inner = getattr(current, "optimizer", None)
        if inner is None or inner is current:
            break
        current = inner
    return current if current is not None else optimizer


def _states_are_sharded(accelerator: Any) -> bool:
    if accelerator is None:
        return False
    distributed_type = str(getattr(accelerator, "distributed_type", "")).upper()
    if "FSDP" in distributed_type:
        return True
    if "DEEPSPEED" in distributed_type:
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        zero_stage = getattr(plugin, "zero_stage", 2) if plugin is not None else 2
        return int(zero_stage) >= 1
    return False


def _as_step(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().flatten()[0].item())
    if value is None:
        return 0
    return int(value)


@torch.no_grad()
def adam_moment_variance(
    optimizer: Any,
    *,
    accelerator: Any = None,
    sharded: bool | None = None,
    fallback_step: int | None = None,
    prefix: str = "grad_var",
) -> dict[str, float]:
    inner = unwrap_optimizer(optimizer)
    param_groups = getattr(inner, "param_groups", None)
    state_dict = getattr(inner, "state", None)
    if not param_groups or not isinstance(state_dict, dict):
        return {}

    if sharded is None:
        sharded = _states_are_sharded(accelerator)

    second_moment: dict[torch.device, torch.Tensor] = {}
    first_moment_sq: dict[torch.device, torch.Tensor] = {}
    numel = 0
    skipped = 0
    last_step = 0

    for group in param_groups:
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        for param in group.get("params", []):
            state = state_dict.get(param)
            if not isinstance(state, dict):
                continue
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None or exp_avg_sq is None:
                continue
            if not (exp_avg.is_floating_point() and exp_avg_sq.is_floating_point()):
                skipped += exp_avg.numel()
                continue

            step = _as_step(state.get("step", group.get("step")))
            if step <= 0:
                step = _as_step(fallback_step)
            if step <= 0:
                continue
            last_step = max(last_step, step)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            device = exp_avg.device
            if device not in second_moment:
                second_moment[device] = torch.zeros((), dtype=torch.float64, device=device)
                first_moment_sq[device] = torch.zeros((), dtype=torch.float64, device=device)

            m_hat = exp_avg.to(torch.float64) / bias_correction1
            v_hat = exp_avg_sq.to(torch.float64) / bias_correction2
            second_moment[device] += v_hat.sum()
            first_moment_sq[device] += m_hat.pow(2).sum()
            numel += exp_avg.numel()

    if numel == 0:
        return {}

    total_v = float(sum(value.sum().item() for value in second_moment.values()))
    total_m2 = float(sum(value.sum().item() for value in first_moment_sq.values()))
    counts = torch.tensor([total_v, total_m2, float(numel), float(skipped)], dtype=torch.float64)

    if sharded and torch.distributed.is_available() and torch.distributed.is_initialized():
        counts = counts.to(_reduce_device())
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        counts = counts.cpu()

    total_v, total_m2, total_numel, total_skipped = (float(x) for x in counts.tolist())
    proxy = total_v - total_m2

    metrics = {
        f"{prefix}/adam_proxy": proxy,
        f"{prefix}/adam_proxy_relu": max(proxy, 0.0),
        f"{prefix}/adam_proxy_mean": proxy / total_numel if total_numel > 0 else 0.0,
        f"{prefix}/second_moment": total_v,
        f"{prefix}/first_moment_sq": total_m2,
        f"{prefix}/num_elements": total_numel,
        f"{prefix}/step": float(last_step),
    }
    if total_skipped > 0:
        metrics[f"{prefix}/skipped_elements"] = total_skipped
    return metrics


def _reduce_device() -> torch.device:
    backend = torch.distributed.get_backend()
    if backend == torch.distributed.Backend.NCCL and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class GradientVarianceTracker:
    def __init__(self, every_n_steps: int = 1, prefix: str = "grad_var") -> None:
        self.every_n_steps = max(int(every_n_steps), 1)
        self.prefix = prefix

    def should_compute(self, step: int) -> bool:
        return step > 0 and step % self.every_n_steps == 0

    def metrics(self, optimizer: Any, *, accelerator: Any = None, step: int = 0) -> dict[str, float]:
        if optimizer is None or not self.should_compute(step):
            return {}
        return adam_moment_variance(
            optimizer,
            accelerator=accelerator,
            fallback_step=step,
            prefix=self.prefix,
        )


def _flatten_gradient(gradient: Any) -> torch.Tensor:
    if isinstance(gradient, torch.Tensor):
        return gradient.detach().reshape(-1).to(torch.float64)
    parts = [part.detach().reshape(-1).to(torch.float64) for part in gradient if part is not None]
    if not parts:
        raise ValueError("Empty gradient")
    return torch.cat(parts)


def empirical_total_variance(gradients: Sequence[Any] | Iterable[Any]) -> float:
    accumulator = TotalVarianceAccumulator()
    for gradient in gradients:
        accumulator.update(gradient)
    return accumulator.total_variance()


class TotalVarianceAccumulator:
    def __init__(self) -> None:
        self._sum: torch.Tensor | None = None
        self._sum_sq_norm: float = 0.0
        self._count: int = 0

    @torch.no_grad()
    def update(self, gradient: Any) -> None:
        flat = _flatten_gradient(gradient)
        if self._sum is None:
            self._sum = torch.zeros_like(flat)
        elif self._sum.numel() != flat.numel():
            raise ValueError(
                f"Gradient size mismatch: expected {self._sum.numel()}, got {flat.numel()}"
            )
        self._sum += flat
        self._sum_sq_norm += float(flat.pow(2).sum().item())
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> torch.Tensor:
        if self._sum is None or self._count == 0:
            raise ValueError("No gradients accumulated")
        return self._sum / self._count

    def total_variance(self) -> float:
        if self._count == 0:
            raise ValueError("No gradients accumulated")
        mean_sq_norm = self._sum_sq_norm / self._count
        mean_norm_sq = float(self.mean().pow(2).sum().item())
        return max(mean_sq_norm - mean_norm_sq, 0.0)

    def estimation_error(self, ground_truth: Any) -> float:
        return float((self.mean() - _flatten_gradient(ground_truth)).norm().item())


def gradient_variance_ratio(total_variance: float, reference_total_variance: float) -> float:
    if reference_total_variance <= 0.0:
        raise ValueError("reference_total_variance must be positive")
    return total_variance / reference_total_variance
