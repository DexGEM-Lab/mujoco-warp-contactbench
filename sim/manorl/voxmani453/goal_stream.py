"""Variable-length structured goal/reference stream contract (step 4).

This module owns the DATA contract only: what a stream is, what makes it valid,
and how physical time becomes features. No learned parameter lives here; the
token encoder, the cross-attention block and the towers live in `model.py`.

A stream is a padded batch of per-world token records:

    raw      [B, N, F] float32  stream-specific payload (pose, rot6d, joints, ...)
    mask     [B, N]    bool     True = keep; padding is a trailing False run
    dt       [B, N]    float32  physical seconds relative to now (future > 0)
    frame_id [B, N]    int64    provenance: index into the original dense record
    type_id  [B, N]    int64    token type (object waypoint, hand reference, ...)
    presence [B]       float32  {0,1} whole-stream gate (0 = channel withheld)

Three properties the contract deliberately guarantees, because the model
depends on each of them:

1. **Position carries no meaning.** Time is `dt` in seconds and identity is
   `type_id`; array index is never encoded. Dropping waypoints therefore cannot
   silently relabel "the 10th token" as a fixed future horizon. Ordering is not
   validated, because the encoder must not depend on it.
2. **`frame_id` is provenance, never a feature.** It exists so a later PPO
   replay can rebuild the exact token set that produced an old log-prob. Feeding
   an absolute dense index to the policy would be a shortcut around `dt`.
3. **Padding is not data.** The encoder rewrites every padded slot to zero with
   `torch.where` before any projection, so padded values (including NaN/Inf
   left by a producer) cannot reach the output and receive exactly zero
   gradient. Validation therefore only requires finiteness on kept slots.

N = 0 is a first-class case: the encoder always prepends one always-valid NULL
token, so no attention row is ever fully masked and softmax never sees an
all -inf row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from . import config as C

STREAM_CONTRACT_VERSION = "v3.varlen-goal-stream.1"

TOKEN_DIM = C.MODEL_DIM  # every stream encodes to this width, whatever F is
TIME_CHANNELS = 1 + 2 * len(C.FOURIER_BANDS)  # dt/T_MAX + 8 bands x (sin, cos) = 17
TYPE_VOCAB = 8

# Stream names consumed by the model. The two channels are kept apart end to
# end (separate encoders, separate softmax) so a denser hand stream cannot
# outvote the object stream inside one attention distribution.
OBJECT_STREAM = "object"
HAND_STREAM = "hand"
STREAM_NAMES = (OBJECT_STREAM, HAND_STREAM)

# Reserved token type ids (stable; append only).
TYPE_OBJECT_WAYPOINT = 0
# Plan rows without protected-anchor semantics are ordinary object references.
TYPE_OBJECT_REFERENCE = TYPE_OBJECT_WAYPOINT
TYPE_OBJECT_ANCHOR = 1
TYPE_HAND_REFERENCE = 2
TYPE_HAND_ANCHOR = 3

# The fixed-goal compatibility payload (per-part [position3, quaternion_wxyz4,
# dt1, flag1] records) and its adapter were retired in v3.abi.9.  The frozen
# v3.abi.8 ABI field table kept for the resume fork lives in train.py
# (STRUCTURED_GOAL_ONLY_OLD_ABI_FIELDS).


def time_channels(dt: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
    """[..., ] seconds -> [..., 17]. Signed: sin is odd, so past/future differ."""
    ang = 2.0 * math.pi * dt.unsqueeze(-1) * bands
    return torch.cat((dt.unsqueeze(-1) / C.T_MAX, ang.sin(), ang.cos()), dim=-1)


@dataclass(frozen=True)
class StreamSpec:
    """Frozen per-stream shape contract, fixed when the model is constructed."""

    name: str
    raw_dim: int
    type_vocab: int = TYPE_VOCAB

    def __post_init__(self):
        if self.name not in STREAM_NAMES:
            raise ValueError(
                f"unknown stream name {self.name!r}; expected one of {STREAM_NAMES}"
            )
        if self.raw_dim < 1:
            raise ValueError("raw_dim must be positive")
        if not 1 <= self.type_vocab <= TYPE_VOCAB:
            raise ValueError(f"type_vocab must be in [1, {TYPE_VOCAB}]")


@dataclass(frozen=True)
class GoalStream:
    """One padded variable-length stream. Immutable; `validate` is explicit."""

    raw: torch.Tensor
    mask: torch.Tensor
    dt: torch.Tensor
    frame_id: torch.Tensor
    type_id: torch.Tensor
    presence: torch.Tensor

    # --- views -------------------------------------------------------------
    @property
    def batch(self) -> int:
        return self.raw.shape[0]

    @property
    def length(self) -> int:
        return self.raw.shape[1]

    @property
    def raw_dim(self) -> int:
        return self.raw.shape[2]

    @property
    def device(self) -> torch.device:
        return self.raw.device

    @property
    def lengths(self) -> torch.Tensor:
        """[B] int64 kept-token count per world."""
        return self.mask.sum(-1)

    # --- construction ------------------------------------------------------
    @classmethod
    def empty(
        cls,
        batch: int,
        raw_dim: int,
        *,
        length: int = 0,
        device=None,
        dtype=torch.float32,
    ) -> "GoalStream":
        """All-padding stream: every world contributes only the NULL token."""
        return cls(
            raw=torch.zeros(batch, length, raw_dim, dtype=dtype, device=device),
            mask=torch.zeros(batch, length, dtype=torch.bool, device=device),
            dt=torch.zeros(batch, length, dtype=dtype, device=device),
            frame_id=torch.zeros(batch, length, dtype=torch.int64, device=device),
            type_id=torch.zeros(batch, length, dtype=torch.int64, device=device),
            presence=torch.zeros(batch, dtype=dtype, device=device),
        )

    @classmethod
    def pack(
        cls,
        records,
        *,
        raw_dim: int,
        length: int | None = None,
        device=None,
        dtype=torch.float32,
    ) -> "GoalStream":
        """Adapter seam: per-world variable-length records -> one padded stream.

        `records` is a sequence of length B; each entry is a mapping with keys
        'raw' [n, F], 'dt' [n], 'frame_id' [n], 'type_id' [n] and an optional
        'presence' scalar (default 1 when n > 0, forced 0 when n == 0). Arrays
        may be numpy or torch. `length` pads to a fixed bucket width; it must be
        at least the longest record. This is the only supported way for the
        environment/PPO layer to build a stream, so the prefix-mask invariant
        cannot be produced by accident elsewhere.
        """
        batch = len(records)
        sizes = []
        for index, record in enumerate(records):
            payload = torch.as_tensor(record["raw"])
            if payload.ndim != 2 or payload.shape[1] != raw_dim:
                raise ValueError(
                    f"record {index} raw must be [n, {raw_dim}], got {tuple(payload.shape)}"
                )
            sizes.append(payload.shape[0])
        width = max(sizes, default=0) if length is None else length
        if width < max(sizes, default=0):
            raise ValueError(
                f"bucket length {width} is shorter than the longest record {max(sizes)}"
            )
        stream = cls.empty(batch, raw_dim, length=width, device=device, dtype=dtype)
        raw, mask = stream.raw.clone(), stream.mask.clone()
        dt, frame_id = stream.dt.clone(), stream.frame_id.clone()
        type_id, presence = stream.type_id.clone(), stream.presence.clone()
        for index, (record, size) in enumerate(zip(records, sizes)):
            if size:
                raw[index, :size] = torch.as_tensor(
                    record["raw"], dtype=dtype, device=device
                )
                dt[index, :size] = torch.as_tensor(
                    record["dt"], dtype=dtype, device=device
                ).reshape(size)
                frame_id[index, :size] = torch.as_tensor(
                    record["frame_id"], dtype=torch.int64, device=device
                ).reshape(size)
                type_id[index, :size] = torch.as_tensor(
                    record["type_id"], dtype=torch.int64, device=device
                ).reshape(size)
                mask[index, :size] = True
            presence[index] = 0.0 if size == 0 else float(record.get("presence", 1.0))
        return cls(
            raw=raw,
            mask=mask,
            dt=dt,
            frame_id=frame_id,
            type_id=type_id,
            presence=presence,
        ).validate(raw_dim=raw_dim)

    def index_rows(self, index) -> "GoalStream":
        """Row subset with the padded width untouched.

        Slicing a rollout-wide rebuild is bit-identical to rebuilding the
        subset: every row's payload is a pure function of its transition key,
        and the padded width comes from the immutable plan, so it cannot
        depend on the online/minibatch grouping (see RolloutBuffer
        .rebuild_streams). This is the accessor behind the 2026-09-28
        throughput fix: 96 per-microbatch rebuilds cost 43 s of pure CPU
        NumPy work per update with the GPU idle.
        """
        return replace(
            self,
            raw=self.raw[index],
            mask=self.mask[index],
            dt=self.dt[index],
            frame_id=self.frame_id[index],
            type_id=self.type_id[index],
            presence=self.presence[index],
        )

    def pad_to(self, length: int) -> "GoalStream":
        """Grow the padded width to a compile/bucket size. Kept tokens untouched."""
        extra = length - self.length
        if extra < 0:
            raise ValueError(f"cannot shrink a stream from {self.length} to {length}")
        if extra == 0:
            return self

        def grow(tensor):
            return torch.cat(
                (tensor, tensor.new_zeros((tensor.shape[0], extra) + tensor.shape[2:])),
                1,
            )

        return replace(
            self,
            raw=grow(self.raw),
            mask=grow(self.mask),
            dt=grow(self.dt),
            frame_id=grow(self.frame_id),
            type_id=grow(self.type_id),
        )

    def to(self, device) -> "GoalStream":
        return replace(
            self,
            raw=self.raw.to(device),
            mask=self.mask.to(device),
            dt=self.dt.to(device),
            frame_id=self.frame_id.to(device),
            type_id=self.type_id.to(device),
            presence=self.presence.to(device),
        )

    # --- contract ----------------------------------------------------------
    def validate_contract(
        self, *, raw_dim: int | None = None, type_vocab: int = TYPE_VOCAB
    ) -> "GoalStream":
        """Metadata-only ingress check: shapes, dtypes and devices.

        All of these are host-side tensor metadata, so this check never
        synchronizes and is therefore free to run on every model call.  The
        value-level invariants (finite kept payloads, leading-True mask prefix,
        id ranges, presence encoding) live in :meth:`validate_values`; the
        per-call contract check is the cheap one on purpose (2026-09-29: the
        full check costs 8 device reductions per stream per forward, which
        drains the pipeline on every rollout step and every PPO microstep).
        """
        name = type(self).__name__
        if self.raw.ndim != 3:
            raise ValueError(
                f"{name}.raw must be [B, N, F], got {tuple(self.raw.shape)}"
            )
        batch, length, width = self.raw.shape
        if raw_dim is not None and width != raw_dim:
            raise ValueError(
                f"{name}.raw feature width {width} does not match the spec {raw_dim}"
            )
        for field, expected_shape, expected_dtype in (
            ("mask", (batch, length), torch.bool),
            ("dt", (batch, length), self.raw.dtype),
            ("frame_id", (batch, length), torch.int64),
            ("type_id", (batch, length), torch.int64),
            ("presence", (batch,), self.raw.dtype),
        ):
            tensor = getattr(self, field)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name}.{field} must be {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"{name}.{field} must be {expected_dtype}, got {tensor.dtype}"
                )
            if tensor.device != self.raw.device:
                raise ValueError(
                    f"{name}.{field} is on {tensor.device}, raw is on {self.raw.device}"
                )
        if self.raw.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"{name}.raw must be a real float tensor, got {self.raw.dtype}"
            )
        return self

    def validate_values(self, *, type_vocab: int = TYPE_VOCAB) -> "GoalStream":
        """Value-level invariants.  Each check reduces on the device, so every
        one of them host-synchronizes when the stream lives on CUDA."""
        name = type(self).__name__
        batch, length, width = self.raw.shape
        if length and bool((~self.mask[:, :-1] & self.mask[:, 1:]).any()):
            raise ValueError(
                f"{name}.mask must be a leading-True prefix; found a kept token after padding"
            )
        kept = self.mask
        if not bool(torch.isfinite(self.raw[kept]).all()):
            raise ValueError(f"{name}.raw is not finite on kept tokens")
        if not bool(torch.isfinite(self.dt[kept]).all()):
            raise ValueError(f"{name}.dt is not finite on kept tokens")
        kept_types = self.type_id[kept]
        if kept_types.numel() and not bool(
            ((kept_types >= 0) & (kept_types < type_vocab)).all()
        ):
            raise ValueError(
                f"{name}.type_id must be in [0, {type_vocab}) on kept tokens"
            )
        kept_frames = self.frame_id[kept]
        if kept_frames.numel() and not bool((kept_frames >= 0).all()):
            raise ValueError(f"{name}.frame_id must be non-negative on kept tokens")
        if not bool(((self.presence == 0) | (self.presence == 1)).all()):
            raise ValueError(f"{name}.presence must be exactly 0 or 1")
        if bool(((self.lengths == 0) & (self.presence != 0)).any()):
            raise ValueError(
                f"{name}.presence must be 0 for a world with no kept tokens"
            )
        return self

    def validate(
        self, *, raw_dim: int | None = None, type_vocab: int = TYPE_VOCAB
    ) -> "GoalStream":
        """Full ingress check = contract (free) + values (host-synchronising).

        Deliberately strict: this is the check the smokes and any explicit
        ingress point should use.
        """
        self.validate_contract(raw_dim=raw_dim, type_vocab=type_vocab)
        return self.validate_values(type_vocab=type_vocab)

    def sanitized(self):
        """Padding-free views used by the encoder: (raw, dt, type_id).

        `torch.where` (not multiplication) is what makes NaN/Inf padding inert
        and gives padded slots exactly zero gradient.
        """
        keep = self.mask.unsqueeze(-1)
        raw = torch.where(keep, self.raw, self.raw.new_zeros(()))
        dt = torch.where(self.mask, self.dt, self.dt.new_zeros(()))
        type_id = torch.where(self.mask, self.type_id, self.type_id.new_zeros(()))
        return raw, dt, type_id


def empty_streams(batch: int, specs, *, device=None) -> dict:
    """All-empty streams for every spec; the N=0 / NULL-token path."""
    return {
        name: GoalStream.empty(batch, spec.raw_dim, device=device)
        for name, spec in specs.items()
    }


# --- reserved adapter layouts ------------------------------------------------
# The step-5 environment/PPO adapter is expected to emit these `raw` layouts.
# They are defaults, not a model constraint: `raw_dim` is a free field of
# StreamSpec, so the payload can change without touching any attention code.

OBJECT_RAW_LAYOUT = ("position3", "rot6d", "per part; then endpoint_flag1")
HAND_RAW_LAYOUT = ("wrist_position3", "wrist_rot6d", "finger_q22")


def object_spec(num_parts: int = 1) -> StreamSpec:
    """Object waypoint stream: [pos3 + rot6d]*num_parts + endpoint flag + 4
    canonical surface points (12) in world meters - the same geometry the
    current-scene object cloud carries, so both sides of the comparison are
    geometric (exp13)."""
    return StreamSpec(OBJECT_STREAM, 9 * num_parts + 1 + 12)


def hand_spec() -> StreamSpec:
    """Hand reference stream: wrist pos3 + rot6d + 22 finger joints + 16
    keypoint positions (48, world meters) - imitation climbs from joint
    space to geometric space (exp13)."""
    return StreamSpec(HAND_STREAM, 9 + C.FINGER_DOF + 48)


def default_specs(num_parts: int = 1) -> dict:
    return {OBJECT_STREAM: object_spec(num_parts), HAND_STREAM: hand_spec()}
