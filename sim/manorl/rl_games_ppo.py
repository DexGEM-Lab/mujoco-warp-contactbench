"""rl-games-compatible continuous PPO KL and adaptive scheduling for ManoRL."""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo import compute_gae


RL_GAMES_BOUNDS_SOFT_BOUND = 1.1


def rl_games_policy_kl(
    current_mean: torch.Tensor,
    current_std: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
) -> torch.Tensor:
    """Return the continuous-policy KL used by the source rl-games version."""

    if not (
        current_mean.shape
        == current_std.shape
        == reference_mean.shape
        == reference_std.shape
    ):
        raise ValueError("policy KL tensors must have identical shapes")
    if current_mean.ndim < 2:
        raise ValueError("policy KL tensors must include batch and action dimensions")
    c1 = torch.log(reference_std / current_std + 1.0e-5)
    c2 = (
        current_std.square() + (reference_mean - current_mean).square()
    ) / (2.0 * (reference_std.square() + 1.0e-5))
    return (c1 + c2 - 0.5).sum(dim=-1).mean()


def rl_games_bounds_loss(policy_mean: torch.Tensor) -> torch.Tensor:
    """Return the source rl-games soft-bound penalty averaged over a minibatch."""

    mu_loss_high = torch.square(
        torch.clamp(policy_mean - RL_GAMES_BOUNDS_SOFT_BOUND, min=0.0)
    )
    mu_loss_low = torch.square(
        torch.clamp(policy_mean + RL_GAMES_BOUNDS_SOFT_BOUND, max=0.0)
    )
    return (mu_loss_low + mu_loss_high).sum(dim=-1).mean()


def rl_games_critic_loss(
    current_values: torch.Tensor,
    reference_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    value_clip: float,
) -> torch.Tensor:
    """Return the source rl-games clipped critic MSE before critic weighting."""

    value_losses = F.mse_loss(current_values, returns, reduction="none")
    if value_clip > 0.0:
        clipped_values = reference_values + torch.clamp(
            current_values - reference_values,
            min=-value_clip,
            max=value_clip,
        )
        clipped_losses = F.mse_loss(clipped_values, returns, reduction="none")
        value_losses = torch.maximum(value_losses, clipped_losses)
    return value_losses.mean()


class RlGamesAdaptiveLR:
    """The source legacy adaptive scheduler, invoked once per minibatch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        kl_threshold: float,
        min_lr: float = 1.0e-6,
        max_lr: float = 1.0e-2,
    ) -> None:
        if kl_threshold <= 0.0:
            raise ValueError("kl_threshold must be positive")
        self.optimizer = optimizer
        self.kl_threshold = float(kl_threshold)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self, kl: torch.Tensor | float | None = None) -> None:
        if kl is None:
            return
        value = float(kl)
        for group in self.optimizer.param_groups:
            learning_rate = float(group["lr"])
            if value > 2.0 * self.kl_threshold:
                learning_rate = max(learning_rate / 1.5, self.min_lr)
            if value < 0.5 * self.kl_threshold:
                learning_rate = min(learning_rate * 1.5, self.max_lr)
            group["lr"] = learning_rate
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]


class RlGamesPPO(PPO):
    """skrl PPO objective with the source continuous-policy KL contract."""

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="policy_mean", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="policy_std", size=self.action_space, dtype=torch.float32)
            self._tensors_names.extend(["policy_mean", "policy_std"])
        self._current_policy_mean: torch.Tensor | None = None
        self._current_policy_std: torch.Tensor | None = None

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions, outputs = super().act(
            observations, states, timestep=timestep, timesteps=timesteps
        )
        if self.training and "mean_actions" in outputs and "log_std" in outputs:
            self._current_policy_mean = outputs["mean_actions"].detach()
            self._current_policy_std = outputs["log_std"].detach().exp()
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor | None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        Agent.record_transition(
            self,
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        if not self.training:
            return
        if self._current_policy_mean is None or self._current_policy_std is None:
            raise RuntimeError("training action did not expose policy mean/std")

        self._current_next_observations = next_observations
        self._current_next_states = next_states
        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
        if self.cfg.time_limit_bootstrap and truncated.any():
            with torch.no_grad():
                inputs = {
                    "observations": self._observation_preprocessor(next_observations),
                    "states": self._state_preprocessor(next_states),
                }
                next_values, _ = self.value.act(inputs, role="value")
                next_values = self._value_preprocessor(next_values, inverse=True)
            rewards += self.cfg.discount_factor * next_values * truncated

        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=self._current_values,
            policy_mean=self._current_policy_mean,
            policy_std=self._current_policy_std,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps
        with torch.no_grad(), torch.autocast(
            device_type=self._device_type, enabled=self.cfg.mixed_precision
        ):
            inputs = {
                "observations": self._observation_preprocessor(
                    self._current_next_observations
                ),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        policy_loss_total = 0.0
        entropy_loss_total = 0.0
        value_loss_total = 0.0
        bounds_loss_total = 0.0
        bounds_loss_coef = float(getattr(self.cfg, "bounds_loss_coef", 0.0))
        exact_kls: list[float] = []
        approximate_kls: list[float] = []
        learning_rates = [float(self.optimizer.param_groups[0]["lr"])]
        scheduler_increases = 0
        scheduler_decreases = 0
        completed_minibatches = 0

        for epoch in range(self.cfg.learning_epochs):
            samples = self.memory.sample(
                names=self._tensors_names,
                batch_size=len(self.memory),
                mini_batches=self.cfg.mini_batches,
            )
            index_batches = torch.tensor_split(
                self.memory.sampling_indexes, len(samples)
            )
            for sample, sample_indexes in zip(samples, index_batches, strict=True):
                (
                    sampled_observations,
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                    sampled_reference_mean,
                    sampled_reference_std,
                ) = sample
                with torch.autocast(
                    device_type=self._device_type, enabled=self.cfg.mixed_precision
                ):
                    inputs = {
                        "observations": self._observation_preprocessor(
                            sampled_observations, train=not epoch
                        ),
                        "states": self._state_preprocessor(
                            sampled_states, train=not epoch
                        ),
                    }
                    _, outputs = self.policy.act(
                        {**inputs, "taken_actions": sampled_actions}, role="policy"
                    )
                    next_log_prob = outputs["log_prob"]
                    current_mean = outputs["mean_actions"]
                    current_std = outputs["log_std"].exp()
                    with torch.no_grad():
                        exact_kl = rl_games_policy_kl(
                            current_mean,
                            current_std,
                            sampled_reference_mean,
                            sampled_reference_std,
                        )
                        log_ratio = next_log_prob - sampled_log_prob
                        approximate_kl = (
                            (torch.exp(log_ratio) - 1.0) - log_ratio
                        ).mean()

                    entropy_loss: torch.Tensor | float
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(
                            role="policy"
                        ).mean()
                    else:
                        entropy_loss = 0.0
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio,
                        1.0 - self.cfg.ratio_clip,
                        1.0 + self.cfg.ratio_clip,
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    if bounds_loss_coef:
                        bounds_loss = rl_games_bounds_loss(current_mean)
                    else:
                        bounds_loss = torch.zeros((), device=policy_loss.device)

                    predicted_values, _ = self.value.act(inputs, role="value")
                    value_loss = self.cfg.value_loss_scale * rl_games_critic_loss(
                        predicted_values,
                        sampled_values,
                        sampled_returns,
                        value_clip=self.cfg.value_clip,
                    )

                self.optimizer.zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss + bounds_loss_coef * bounds_loss
                self.scaler.scale(total_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.cfg.grad_norm_clip
                        )
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(
                                self.policy.parameters(), self.value.parameters()
                            ),
                            self.cfg.grad_norm_clip,
                        )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if config.torch.is_distributed:
                    torch.distributed.all_reduce(
                        exact_kl, op=torch.distributed.ReduceOp.SUM
                    )
                    exact_kl /= config.torch.world_size
                previous_lr = float(self.optimizer.param_groups[0]["lr"])
                if self.scheduler is not None:
                    self.scheduler.step(exact_kl.item())
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                scheduler_increases += int(current_lr > previous_lr)
                scheduler_decreases += int(current_lr < previous_lr)
                learning_rates.append(current_lr)

                reference_indexes = sample_indexes.to(
                    device=self.memory.tensors_view["policy_mean"].device
                )
                with torch.no_grad():
                    self.memory.tensors_view["policy_mean"][reference_indexes] = (
                        current_mean.detach()
                    )
                    self.memory.tensors_view["policy_std"][reference_indexes] = (
                        current_std.detach()
                    )

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                bounds_loss_total += bounds_loss.item()
                if self.cfg.entropy_loss_scale:
                    entropy_loss_total += entropy_loss.item()
                exact_kls.append(exact_kl.item())
                approximate_kls.append(approximate_kl.item())
                completed_minibatches += 1

        if completed_minibatches != self.cfg.learning_epochs * self.cfg.mini_batches:
            raise RuntimeError("PPO did not complete every configured minibatch")
        denominator = float(completed_minibatches)
        self.track_data("Loss / Policy loss", policy_loss_total / denominator)
        self.track_data("Loss / Value loss", value_loss_total / denominator)
        self.track_data("Loss / Bounds loss", bounds_loss_total / denominator)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", entropy_loss_total / denominator)
        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item(),
        )
        self.track_data("Learning / Exact KL mean", sum(exact_kls) / denominator)
        self.track_data("Learning / Exact KL min", min(exact_kls))
        self.track_data("Learning / Exact KL max", max(exact_kls))
        self.track_data(
            "Learning / Approximate KL mean", sum(approximate_kls) / denominator
        )
        self.track_data("Learning / Learning rate start", learning_rates[0])
        self.track_data("Learning / Learning rate", learning_rates[-1])
        self.track_data("Learning / Learning rate min", min(learning_rates))
        self.track_data("Learning / Learning rate max", max(learning_rates))
        self.track_data("Learning / Scheduler increases", scheduler_increases)
        self.track_data("Learning / Scheduler decreases", scheduler_decreases)
        self.track_data("Learning / Completed minibatches", completed_minibatches)
