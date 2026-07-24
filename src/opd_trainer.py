from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from trl.experimental.minillm import MiniLLMTrainer
from trl.trainer.utils import selective_log_softmax

from src.grad_variance import GradientVarianceTracker


class CustomMiniLLMTrainer(MiniLLMTrainer):
    def __init__(
        self,
        *args,
        use_baseline: bool = False,
        kl_top_k: int = -1,
        opd_top_k: int = -1,
        log_grad_variance: bool = False,
        grad_variance_logging_steps: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_baseline = use_baseline
        self.kl_top_k = kl_top_k
        self.opd_top_k = opd_top_k
        self.use_top_k_opd = opd_top_k > 0
        self.grad_variance_tracker = (
            GradientVarianceTracker(every_n_steps=grad_variance_logging_steps)
            if log_grad_variance
            else None
        )

    @staticmethod
    def _topk_log_softmax(logits, indices):
        selected_logits = torch.gather(logits, dim=-1, index=indices)
        return selected_logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    def log(self, logs, start_time=None):
        if self.grad_variance_tracker is not None and self.model.training:
            logs.update(
                self.grad_variance_tracker.metrics(
                    self.optimizer,
                    accelerator=self.accelerator,
                    step=self.state.global_step,
                )
            )
        super().log(logs, start_time)

    def _topk_opd_single_step_loss(self, student_logits, teacher_logits, mask):
        top_k = min(self.opd_top_k, student_logits.size(-1))
        topk_ids = torch.topk(student_logits.detach(), top_k, dim=-1).indices  # [B, T, K]
        student_topk_logp = self._topk_log_softmax(student_logits, topk_ids)  # [B, T, K]

        with torch.no_grad():
            teacher_topk_logp = self._topk_log_softmax(teacher_logits, topk_ids)  # [B, T, K]

        student_topk_logp = student_topk_logp - torch.logsumexp(
            student_topk_logp, dim=-1, keepdim=True
        )
        teacher_topk_logp = teacher_topk_logp - torch.logsumexp(
            teacher_topk_logp, dim=-1, keepdim=True
        )

        student_topk_prob = student_topk_logp.exp()
        token_loss = (
            student_topk_prob
            * (student_topk_logp - teacher_topk_logp.detach())
        ).sum(dim=-1)  # [B, T]

        loss = (token_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        needs_full_log_probs = self.single_step_decomposition or (
            self.use_baseline and self.rkl_advantage and self.kl_top_k <= 0
        )
        if needs_full_log_probs:
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_log_probs_on_labels = torch.gather(
                student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)
            teacher_log_probs_on_labels = torch.gather(
                teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)
        else:
            student_log_probs = None
            teacher_log_probs = None
            student_log_probs_on_labels = selective_log_softmax(student_logits, shifted_labels)
            teacher_log_probs_on_labels = selective_log_softmax(teacher_logits, shifted_labels)

        mask = inputs["completion_mask"].bool()

        if self.use_baseline:
            if self.rkl_advantage:
                if self.kl_top_k > 0:
                    top_k = min(self.kl_top_k, student_logits.size(-1))
                    kl_indices = torch.topk(student_logits, top_k, dim=-1).indices

                    if student_log_probs is None:
                        student_log_probs_for_baseline = self._topk_log_softmax(student_logits, kl_indices)
                        teacher_log_probs_for_baseline = self._topk_log_softmax(teacher_logits, kl_indices)
                    else:
                        student_log_probs_for_baseline = torch.gather(
                            student_log_probs, dim=-1, index=kl_indices
                        )
                        teacher_log_probs_for_baseline = torch.gather(
                            teacher_log_probs, dim=-1, index=kl_indices
                        )

                    student_log_mass = torch.logsumexp(
                        student_log_probs_for_baseline, dim=-1, keepdim=True
                    )
                    teacher_log_mass = torch.logsumexp(
                        teacher_log_probs_for_baseline, dim=-1, keepdim=True
                    )
                    student_log_probs_for_baseline = student_log_probs_for_baseline - student_log_mass
                    teacher_log_probs_for_baseline = teacher_log_probs_for_baseline - teacher_log_mass
                    student_probs = student_log_probs_for_baseline.exp()
                else:
                    student_log_probs_for_baseline = student_log_probs
                    teacher_log_probs_for_baseline = teacher_log_probs
                    student_probs = student_log_probs_for_baseline.exp()

                baseline = (
                    student_probs
                    * (student_log_probs_for_baseline - teacher_log_probs_for_baseline)
                ).sum(dim=-1)
                centered_gap = student_log_probs_on_labels - teacher_log_probs_on_labels - baseline
                reverse_kl_advantage = -centered_gap.detach()

                if os.environ.get("OPD_DEBUG_BASELINE") == "1":
                    rank = getattr(self.accelerator, "process_index", 0)
                    if rank == 0:
                        tensors = {
                            "student_log_probs": student_log_probs_for_baseline,
                            "teacher_log_probs": teacher_log_probs_for_baseline,
                            "student_log_probs_on_labels": student_log_probs_on_labels,
                            "teacher_log_probs_on_labels": teacher_log_probs_on_labels,
                            "baseline": baseline,
                            "centered_gap": centered_gap,
                            "reverse_kl_advantage": reverse_kl_advantage,
                            "input_advantages": inputs["advantages"],
                        }
                        for name, tensor in tensors.items():
                            tensor = tensor.detach().float()
                            print(
                                name,
                                "shape=", tuple(tensor.shape),
                                "mean=", tensor.mean().item(),
                                "std=", tensor.std().item(),
                                "min=", tensor.min().item(),
                                "max=", tensor.max().item(),
                                flush=True,
                            )
                    raise RuntimeError("Stopped after OPD baseline debug")

                inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage

            # Compute GRPO loss on verifiable reward
            loss = self._compute_loss(model, inputs)

            # Compute loss
            if self.single_step_decomposition:
                single_step_decomposition_loss = self._single_step_decomposition_loss(
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    mask=mask,
                )

                loss += single_step_decomposition_loss

        else:
            if self.rkl_advantage:
                reverse_kl_advantage = self._compute_advantage(
                    student_log_probs_on_labels=student_log_probs_on_labels,
                    teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                    mask=mask,
                )

                inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage

            # Compute GRPO loss on verifiable reward
            loss = self._compute_loss(model, inputs)

            if self.single_step_decomposition:
                if self.use_top_k_opd:
                    single_step_decomposition_loss = self._topk_opd_single_step_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        mask=mask,
                    )
                else:
                    single_step_decomposition_loss = self._single_step_decomposition_loss(
                        student_log_probs=student_log_probs,
                        teacher_log_probs=teacher_log_probs,
                        mask=mask,
                    )

                loss += single_step_decomposition_loss

        torch.cuda.empty_cache()
        return (loss, student_outputs) if return_outputs else loss
