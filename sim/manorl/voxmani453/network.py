"""Architecture primitives from DexGEM-Lab/voxMani @16a80fc.

Port keeps point stems, Rodrigues propagation, dual goal cross-attention and
per-joint readout. Canonical buffers persist for standalone ManoRL checkpoints.
Upstream source: sim/auton_policy_v3/model.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import config as C
from . import goal_stream as GS

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion -> rotation matrix, [..., 4] -> [..., 3, 3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot_6d(q: torch.Tensor) -> torch.Tensor:
    """First two columns of R, flattened [r11,r21,r31,r12,r22,r32] (proposal 1.1)."""
    m = quat_wxyz_to_mat(q)
    return torch.cat((m[..., :, 0], m[..., :, 1]), dim=-1)


def place(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Rigidly transform shared canonical points by per-world poses.

    points (..., P, 3), pose (B..., 7) with wxyz quat -> (B..., ..., P, 3).
    Batch broadcasting follows torch.matmul rules: a shared point set
    (P, 3) is applied to every pose; a per-bone point set (16, P, 3) is
    applied elementwise when the pose batch ends with 16.
    """
    rot = quat_wxyz_to_mat(pose[..., 3:])
    return points @ rot.transpose(-1, -2) + pose[..., None, :3]


def time_features(
    dt: torch.Tensor, flag: torch.Tensor, bands: torch.Tensor
) -> torch.Tensor:
    """Signed time encoding (proposal 2.4): linear + flag + Fourier, 18 channels.

    dt in seconds (past negative, future positive); sin is odd so sign survives.
    """
    ang = 2.0 * math.pi * dt.unsqueeze(-1) * bands  # [..., 8]
    return torch.cat(
        (dt.unsqueeze(-1) / C.T_MAX, flag.unsqueeze(-1), ang.sin(), ang.cos()), dim=-1
    )


def feature_norm(layer, features):
    """FP32-statistics normalization with the feature storage dtype retained."""
    if features.dtype == torch.bfloat16:
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.layer_norm(
                features,
                layer.normalized_shape,
                layer.weight.to(features.dtype),
                layer.bias.to(features.dtype),
                layer.eps,
            )
    return layer(features)


class MHA(nn.Module):
    """Pre-LN multi-head attention + pre-LN FFN, residual both."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn: int,
        kv_dim: int | None = None,
        *,
        with_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(kv_dim) if kv_dim is not None else None
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim or dim, dim)
        self.v = nn.Linear(kv_dim or dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ln_f = nn.LayerNorm(dim) if with_ffn else None
        self.ffn = (
            nn.Sequential(nn.Linear(dim, ffn), nn.SiLU(), nn.Linear(ffn, dim))
            if with_ffn
            else None
        )
        self.checkpoint_parts = False

    def attention_delta(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention update without the residual add.

        kv_mask is [B, S] bool with True = readable, matching the stream
        contract and SDPA's own boolean convention (the inverse of
        nn.MultiheadAttention's key_padding_mask). It is applied inside SDPA,
        before the softmax, so a masked key never contributes weight.
        """
        b, n, d = x.shape
        h = self.heads
        normalized = feature_norm(self.ln_q, x)
        q = self.q(normalized).view(b, n, h, d // h).transpose(1, 2)
        src = normalized if kv is None else feature_norm(self.ln_kv, kv)
        k = self.k(src).view(b, -1, h, d // h).transpose(1, 2)
        v = self.v(src).view(b, -1, h, d // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if kv_mask is None else kv_mask[:, None, None, :]
        )
        return self.o(y.transpose(1, 2).reshape(b, n, d))

    def _attention(
        self, x: torch.Tensor, kv: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x + self.attention_delta(x, kv)

    def _feedforward(self, x):
        return x + self.ffn(feature_norm(self.ln_f, x))

    def forward(self, x, kv=None):
        recompute = self.checkpoint_parts and torch.is_grad_enabled()
        x = (
            checkpoint(
                self._attention, x, kv, use_reentrant=False, preserve_rng_state=False
            )
            if recompute
            else self._attention(x, kv)
        )
        if self.ffn is not None:
            x = (
                checkpoint(
                    self._feedforward, x, use_reentrant=False, preserve_rng_state=False
                )
                if recompute
                else self._feedforward(x)
            )
        return x


# ---------------------------------------------------------------------------
# variable-length goal/reference stream path (contract in goal_stream.py)
# ---------------------------------------------------------------------------


class StreamEncoder(nn.Module):
    """One variable-length stream -> [B, 1+N, D] tokens plus a [B, 1+N] key mask.

    Per-token only: raw payload, physical-time Fourier features and a type
    embedding, then a residual MLP. No self-attention across the stream, so
    cost is linear in N and a dropped token cannot change how its neighbours
    are encoded.

    The leading NULL token is always readable. It is what makes N = 0 a normal
    case: every attention row keeps at least one finite key, so softmax never
    sees an all -inf row and produces NaN.
    """

    def __init__(
        self, raw_dim: int, dim: int = GS.TOKEN_DIM, type_vocab: int = GS.TYPE_VOCAB
    ) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.raw_proj = nn.Linear(raw_dim, dim)
        self.time_proj = nn.Linear(GS.TIME_CHANNELS, dim, bias=False)
        self.type_emb = nn.Embedding(type_vocab, dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("bands", torch.tensor(C.FOURIER_BANDS))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, stream: GS.GoalStream):
        if stream.raw_dim != self.raw_dim:
            raise ValueError(
                f"stream raw width {stream.raw_dim} does not match encoder {self.raw_dim}"
            )
        raw, dt, type_id = stream.sanitized()
        tokens = (
            self.raw_proj(raw)
            + self.time_proj(GS.time_channels(dt, self.bands))
            + self.type_emb(type_id)
        )
        tokens = tokens + self.mlp(feature_norm(self.ln, tokens))
        null = self.null_token.to(tokens.dtype).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((null, tokens), 1)
        key_mask = torch.cat((stream.mask.new_ones(stream.batch, 1), stream.mask), 1)
        return tokens, key_mask, stream.presence


class VarlenGoalBlock(nn.Module):
    """One fusion layer over variable-length streams.

    current <- current_self_attention(current)
    current <- current + g_o * cross(current -> object) + g_h * cross(current -> hand)
    current <- current + FFN(current)

    The two streams are read by separate cross-attentions with separate
    parameters and separate softmax normalisations. Merging them into one
    key set would let whichever stream happens to carry more tokens dominate
    the attention mass for a reason that is an artefact of sampling density.

    The presence gate multiplies a whole branch. When a channel is withheld
    (hand reference removed by curriculum, or simply absent) its contribution
    is exactly zero, not "the NULL token's output".
    """

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = MHA(dim, heads, ffn, with_ffn=False)
        self.object_cross = MHA(dim, heads, ffn, kv_dim=dim, with_ffn=False)
        self.hand_cross = MHA(dim, heads, ffn, kv_dim=dim, with_ffn=False)
        self.ln_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.SiLU(), nn.Linear(ffn, dim))

    def forward(
        self,
        current,
        object_tokens,
        object_mask,
        object_gate,
        hand_tokens,
        hand_mask,
        hand_gate,
    ):
        current = self.self_attn(current)
        gate_o = object_gate.to(current.dtype)[:, None, None]
        gate_h = hand_gate.to(current.dtype)[:, None, None]
        current = (
            current
            + gate_o
            * self.object_cross.attention_delta(current, object_tokens, object_mask)
            + gate_h * self.hand_cross.attention_delta(current, hand_tokens, hand_mask)
        )
        return current + self.ffn(feature_norm(self.ln_f, current))


# ---------------------------------------------------------------------------
# canonical buffers (built once by canonical.py, injected here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalBuffers:
    """Constant geometry, all positions pre-scaled by 1/SCENE_SIGMA."""

    obj_points: torch.Tensor  # (N_part, 64, 3), body frame; explicit part axis
    fine_groups: torch.Tensor  # (16, 4) indices into 64
    coarse_of_fine: torch.Tensor  # (16,) coarse id per fine patch
    coarse_point_idx: torch.Tensor  # (32,) indices into 64 (half-draw)
    coarse_pt_groups: torch.Tensor  # (4, 8) indices into the 32
    desk_points: torch.Tensor  # (64, 3) static, workspace frame
    desk_groups: torch.Tensor  # (16, 4)
    hand_points: torch.Tensor  # (16, 16, 3) per-bone canonical points
    tip_bone: torch.Tensor  # (5,) bone index per fingertip
    tip_offsets: torch.Tensor  # (5, 3) offsets in bone frame


@dataclass(frozen=True)
class SkeletonTree:
    """Hand kinematic tree for the Rodrigues stream (proposal 3.4)."""

    edges: tuple  # ((parent_bone, child_bone, (joint ids in tree order)), ...)
    joint_child: tuple  # (22,) child bone per joint
    joint_parent: tuple  # (22,) parent bone per joint


# ---------------------------------------------------------------------------
# Rodrigues skeleton stream (proposal 3.4; pure PyTorch, no upstream repo)
# ---------------------------------------------------------------------------


class RodriguesBlock(nn.Module):
    """Rodrigues layer (tree-edge sin/cos-modulated propagation) + joint layer
    + link self-attention. Synchronous Jacobi update across edges; stacked axes
    on one edge are chained in tree order within a single layer."""

    def __init__(
        self, num_joints: int, tree: SkeletonTree, *, joint_update: bool = True
    ) -> None:
        super().__init__()
        cl, cj = C.SKEL_CHANNELS_LINK, C.SKEL_CHANNELS_JOINT
        self.w_bias = nn.Parameter(torch.zeros(num_joints, cl, cl, 4, 4))
        self.w_cos = nn.Parameter(torch.zeros(num_joints, cl, cl, cj, 4, 4))
        self.w_sin = nn.Parameter(torch.zeros(num_joints, cl, cl, cj, 4, 4))
        self.joint_w = (
            nn.Parameter(torch.empty(num_joints, C.SKEL_LINK_DIM, cj))
            if joint_update
            else None
        )
        self.joint_b = (
            nn.Parameter(torch.zeros(num_joints, cj)) if joint_update else None
        )
        # Nonzero kernels avoid a dead product on stacked-axis edges. Small
        # modulation about channel identity keeps the three-block chain stable.
        nn.init.normal_(self.w_cos, std=0.01)
        nn.init.normal_(self.w_sin, std=0.01)
        with torch.no_grad():
            for channel in range(cl):
                self.w_bias[:, channel, channel] = torch.eye(4)
        if self.joint_w is not None:
            nn.init.normal_(self.joint_w, std=0.02)
        self.link_ln = nn.LayerNorm(C.SKEL_LINK_DIM)
        self.attn = MHA(C.SKEL_LINK_DIM, C.ATTN_HEADS, 256)
        self.register_buffer(
            "edge_parent", torch.tensor([edge[0] for edge in tree.edges])
        )
        self.register_buffer(
            "edge_child", torch.tensor([edge[1] for edge in tree.edges])
        )
        self.register_buffer("joint_child", torch.tensor(tree.joint_child))
        self.axis_depth = max(len(edge[2]) for edge in tree.edges)
        for depth in range(self.axis_depth):
            active = [
                (i, edge[2][depth])
                for i, edge in enumerate(tree.edges)
                if len(edge[2]) > depth
            ]
            self.register_buffer(
                f"axis_edges_{depth}", torch.tensor([i for i, _ in active])
            )
            self.register_buffer(
                f"axis_joints_{depth}", torch.tensor([j for _, j in active])
            )

    def forward(self, feats: torch.Tensor, theta: torch.Tensor):
        """Jacobi parent reads; stacked axes composed in parallel edge groups."""
        batch = feats.shape[0]
        u = (
            self.w_bias.unsqueeze(0)
            + torch.einsum("jikcAB,xjc->xjikAB", self.w_cos, theta.cos())
            + torch.einsum("jikcAB,xjc->xjikAB", self.w_sin, theta.sin())
        )
        current = feats[:, self.edge_parent]
        for depth in range(self.axis_depth):
            edge_ids = getattr(self, f"axis_edges_{depth}")
            joint_ids = getattr(self, f"axis_joints_{depth}")
            source = current if depth == 0 else current[:, edge_ids]
            transformed = torch.einsum("xeiab,xeikbc->xekac", source, u[:, joint_ids])
            current = (
                transformed
                if depth == 0
                else current.index_copy(1, edge_ids, transformed)
            )
        child = self.link_ln(
            (feats[:, self.edge_child] + current).reshape(batch, -1, C.SKEL_LINK_DIM)
        )
        new_feats = feats.index_copy(1, self.edge_child, child.reshape(current.shape))
        if self.joint_w is not None:
            delta = (
                torch.einsum(
                    "xjd,jdc->xjc",
                    new_feats[:, self.joint_child].reshape(batch, -1, C.SKEL_LINK_DIM),
                    self.joint_w,
                )
                + self.joint_b
            )
            theta = theta + delta
        tokens = self.attn(new_feats.reshape(batch, C.NUM_HAND_BODIES, C.SKEL_LINK_DIM))
        return tokens.reshape(new_feats.shape), theta


class SkeletonStream(nn.Module):
    """22 finger joints + per-bone contacts -> 16 bone features (proposal 3.4).

    palm-relative kinematic prior; wrist pose deliberately absent (3.4).
    Inputs arrive ABI-normalized (qd/10, contact tanh(/10)).
    """

    def __init__(
        self, num_joints: int, tree: SkeletonTree, *, joint_readout: bool = True
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            RodriguesBlock(
                num_joints,
                tree,
                joint_update=joint_readout or layer < C.SKEL_LAYERS - 1,
            )
            for layer in range(C.SKEL_LAYERS)
        )
        self.theta_weight = nn.Parameter(
            torch.empty(num_joints, 2, C.SKEL_CHANNELS_JOINT)
        )
        self.theta_bias = nn.Parameter(torch.zeros(num_joints, C.SKEL_CHANNELS_JOINT))
        nn.init.normal_(self.theta_weight, std=0.02)
        with torch.no_grad():
            self.theta_weight[:, :, 0] = 0
            self.theta_weight[:, 0, 0] = 1
        self.link_weight = nn.Parameter(
            torch.empty(C.NUM_HAND_BODIES, 3, C.SKEL_LINK_DIM)
        )
        self.link_bias = nn.Parameter(torch.zeros(C.NUM_HAND_BODIES, C.SKEL_LINK_DIM))
        nn.init.normal_(self.link_weight, std=0.02)
        self.link_base = nn.Parameter(torch.zeros(C.NUM_HAND_BODIES, C.SKEL_LINK_DIM))

    def forward(
        self,
        finger_q: torch.Tensor,
        finger_qd: torch.Tensor,
        bone_contact: torch.Tensor,
    ):
        """-> (link features (B,16,64), final joint features (B,22,C_J))."""
        theta = (
            torch.einsum(
                "bji,jic->bjc",
                torch.stack((finger_q, finger_qd), -1),
                self.theta_weight,
            )
            + self.theta_bias
        )
        feats = (
            torch.einsum("bli,lid->bld", bone_contact, self.link_weight)
            + self.link_bias
            + self.link_base
        )
        feats = feats.reshape(-1, C.NUM_HAND_BODIES, C.SKEL_CHANNELS_LINK, 4, 4)
        for block in self.blocks:
            feats, theta = block(feats, theta)
        return feats.reshape(feats.shape[0], C.NUM_HAND_BODIES, -1), theta


# ---------------------------------------------------------------------------
# tower
# ---------------------------------------------------------------------------


class PointStem(nn.Sequential):
    """Keep dense point activations BF16, with native FP32 norm accumulation.

    CUDA autocast otherwise promotes LayerNorm's output to FP32, doubling the
    large point buffers. Native BF16 LayerNorm accumulates statistics in FP32;
    affine operands are temporary casts, with FP32 master weights and gradients.
    """

    def forward(self, features):
        return self.after_first(self[0](features))

    def after_first(self, features):
        for index in range(1, len(self)):
            layer = self[index]
            features = (
                feature_norm(layer, features)
                if isinstance(layer, nn.LayerNorm)
                else layer(features)
            )
        return features


class Tower(nn.Module):
    """One full actor-or-critic tower: three heads + fusion trunk + head."""

    def __init__(
        self,
        tree: SkeletonTree,
        *,
        is_critic: bool,
        skeleton_on: bool,
        trunk_layers: int,
        stream_specs: dict | None = None,
    ) -> None:
        super().__init__()
        self.is_critic = is_critic
        self.skeleton_on = skeleton_on
        d, h = C.MODEL_DIM, C.ATTN_HEADS

        # shared embedding tables within tower (head1/head3 same vocab, 3.2)
        self.sem_emb = nn.Embedding(C.SEMANTIC_VOCAB, C.EMB_DIM)
        self.part_emb = nn.Embedding(C.PART_VOCAB, C.EMB_DIM)
        self.fine_emb = nn.Embedding(C.OBJ_FINE_PATCHES, C.EMB_DIM)
        self.coarse_emb = nn.Embedding(C.OBJ_COARSE_PATCHES, C.EMB_DIM)
        self.bone_emb = nn.Embedding(C.NUM_HAND_BODIES, C.EMB_DIM)
        self.register_buffer("bands", torch.tensor(C.FOURIER_BANDS))

        def stem(c_in: int) -> nn.Module:
            return PointStem(
                nn.Linear(c_in, d),
                nn.LayerNorm(d),
                nn.SiLU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.SiLU(),
            )

        self.scene_stem = stem(C.POINT_CHANNELS_SCENE)
        self.hand_stem = stem(C.POINT_CHANNELS_HAND)
        self.head1_attn = MHA(d, h, C.FFN_DIM)
        self.head2_attn = MHA(d, h, C.FFN_DIM)
        self.activation_checkpoint = False

        if skeleton_on:
            # The critic reads links only: the final theta update has no consumer.
            # Omit its dead parameters so strict Adam checkpoint resume stays valid.
            self.skeleton = SkeletonStream(
                C.FINGER_DOF, tree, joint_readout=not is_critic
            )
            self.skel_proj = nn.Linear(C.SKEL_LINK_DIM, d)
            self.gamma = nn.Parameter(torch.zeros(d))  # zero-init LayerScale

        x_low = C.X_LOW_CRITIC_DIM if is_critic else C.X_LOW_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(x_low, 256), nn.SiLU(), nn.Linear(256, d)
        )
        self.readout_token = nn.Parameter(torch.zeros(1, 1, d))

        # Opt-in: absent stream_specs leaves the tower byte-identical to the
        # legacy checkpoint layout, so existing runs load and train unchanged.
        self.stream_specs = stream_specs
        # Hand-reference residual coefficient (2026-09-29 ablation): scales the
        # hand cross-attention branch only. The stream stays present, so its
        # tokens remain in the key set and are still attended - only the
        # residual it adds to the current token is scaled. Set per update by
        # the trainer (ActorCriticV3.set_hand_ref_gain).
        self.hand_ref_gain = 1.0
        if stream_specs is None:
            self.stream_encoders = self.varlen_trunk = None
        else:
            self.stream_encoders = nn.ModuleDict(
                {
                    name: StreamEncoder(spec.raw_dim, d, spec.type_vocab)
                    for name, spec in stream_specs.items()
                }
            )
            self.varlen_trunk = nn.ModuleList(
                [VarlenGoalBlock(d, h, C.FFN_DIM) for _ in range(trunk_layers)]
            )

        if is_critic:
            self.value_head = nn.Sequential(
                nn.Linear(2 * d, 256), nn.SiLU(), nn.Linear(256, 1)
            )
        else:
            self.finger_shared = nn.Sequential(
                nn.Linear(2 * d + C.SKEL_CHANNELS_JOINT, 64), nn.SiLU()
            )
            self.finger_w = nn.Parameter(torch.empty(C.FINGER_DOF, 64))
            self.finger_b = nn.Parameter(torch.zeros(C.FINGER_DOF))
            # 3 wrist translation + 3 wrist rotation (rate commands).
            self.wrist_head = nn.Sequential(
                nn.Linear(2 * d, 64), nn.SiLU(), nn.Linear(64, 6)
            )
            with torch.no_grad():  # small-gain final layers: near-zero initial actions
                self.wrist_head[-1].weight.mul_(0.01)
                self.wrist_head[-1].bias.zero_()
                nn.init.normal_(self.finger_w, std=0.01)
            self.register_buffer("joint_child", torch.tensor(tree.joint_child))
            self.register_buffer("joint_parent", torch.tensor(tree.joint_parent))

    @staticmethod
    def _xyz_affine(xyz, weight):
        return (
            xyz[..., 0:1] * weight[:, 0]
            + xyz[..., 1:2] * weight[:, 1]
            + xyz[..., 2:3] * weight[:, 2]
        )

    def _scene_points(self, data, constant_prefix=0):
        """Exactly decompose Linear69 into XYZ, static IDs and frame time.

        IDs are shared across worlds and time across each32-point frame. Project
        these before expansion; parameters remain the original Linear69 matrix.
        """
        xyz, dt, flag = data["xyz"], data["dt"], data["flag"]
        batch = xyz.shape[0]
        first = self.scene_stem[0]
        low_precision = torch.is_autocast_enabled(xyz.device.type)
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            identity = torch.cat(
                (
                    self.sem_emb(data["sem"][0]) + self.part_emb(data["part"][0]),
                    self.fine_emb(data["fine"][0]),
                    self.coarse_emb(data["coarse"][0]),
                ),
                -1,
            )
            identity_weight = torch.cat(
                (first.weight[:, 3:19], first.weight[:, 37:]), -1
            )
            static = F.linear(identity, identity_weight, first.bias)
            if constant_prefix:
                sample_dt = torch.cat((dt[:, :1], dt[:, constant_prefix::32]), 1)
                sample_flag = torch.cat((flag[:, :1], flag[:, constant_prefix::32]), 1)
            else:
                sample_dt, sample_flag = dt[:, ::32], flag[:, ::32]
            temporal = F.linear(
                time_features(sample_dt, sample_flag, self.bands),
                first.weight[:, 19:37],
            )
            if constant_prefix:
                time_points = torch.cat(
                    (
                        temporal[:, :1].expand(batch, constant_prefix, -1),
                        temporal[:, 1:, None, :].expand(-1, -1, 32, -1).flatten(1, 2),
                    ),
                    1,
                )
            else:
                time_points = (
                    temporal[:, :, None, :].expand(-1, -1, 32, -1).flatten(1, 2)
                )
            projected = (
                self._xyz_affine(xyz, first.weight[:, :3])
                + static[None, :, :]
                + time_points
            )
        if low_precision:
            projected = projected.to(torch.bfloat16)
        return self.scene_stem.after_first(projected)

    def _hand_points(self, xyz, bone_ids):
        first = self.hand_stem[0]
        low_precision = torch.is_autocast_enabled(xyz.device.type)
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            static = F.linear(
                self.bone_emb(bone_ids[0]), first.weight[:, 3:], first.bias
            )
            projected = self._xyz_affine(xyz, first.weight[:, :3]) + static[None, :, :]
        if low_precision:
            projected = projected.to(torch.bfloat16)
        return self.hand_stem.after_first(projected)

    def _run_checkpoint(self, function, *args):
        if self.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                function, *args, use_reentrant=False, preserve_rng_state=False
            )
        return function(*args)

    @staticmethod
    def _point_slice(data, start, stop):
        return {name: value[:, start:stop] for name, value in data.items()}

    def _scene_pool(self, data, width, constant_prefix=0):
        features = self._scene_points(data, constant_prefix)
        return features.reshape(features.shape[0], -1, width, C.MODEL_DIM).amax(2)

    def _hand_pool(self, xyz, bone_ids):
        features = self._hand_points(xyz, bone_ids)
        return features.reshape(
            features.shape[0], -1, C.HAND_POINTS_PER_BONE, C.MODEL_DIM
        ).amax(2)

    def encode_scene(self, data, fine_end):
        if self.activation_checkpoint and torch.is_grad_enabled():
            pieces = [
                self._run_checkpoint(
                    self._scene_pool, self._point_slice(data, 0, fine_end), 4, fine_end
                )
            ]
            for start in range(fine_end, data["xyz"].shape[1], 256):
                pieces.append(
                    self._run_checkpoint(
                        self._scene_pool, self._point_slice(data, start, start + 256), 8
                    )
                )
            tokens = torch.cat(pieces, 1)
        else:
            scene = self._scene_points(data, fine_end)
            batch = scene.shape[0]
            tokens = torch.cat(
                (
                    scene[:, :fine_end].reshape(batch, -1, 4, C.MODEL_DIM).amax(2),
                    scene[:, fine_end:].reshape(batch, -1, 8, C.MODEL_DIM).amax(2),
                ),
                1,
            )
        return self.head1_attn(tokens)

    def encode_hand(self, xyz, bone_ids, finger_q, finger_qd, contacts):
        batch = xyz.shape[0]
        if self.activation_checkpoint and torch.is_grad_enabled():
            skin = torch.cat(
                [
                    self._run_checkpoint(
                        self._hand_pool,
                        xyz[:, start : start + 128],
                        bone_ids[:, start : start + 128],
                    )
                    for start in range(0, xyz.shape[1], 128)
                ],
                1,
            )
        else:
            skin = self._hand_pool(xyz, bone_ids)
        if self.skeleton_on:
            skel_feats, theta_l = self._run_checkpoint(
                self.skeleton, finger_q, finger_qd, contacts
            )
            hand = skin + self.gamma * self.skel_proj(skel_feats)
        else:
            theta_l = torch.zeros(
                batch, C.FINGER_DOF, C.SKEL_CHANNELS_JOINT, device=skin.device
            )
            hand = skin
        if torch.is_autocast_enabled(hand.device.type):
            hand = hand.to(torch.bfloat16)
        return self.head2_attn(hand), theta_l

    def current_tokens(self, enc: dict[str, torch.Tensor]):
        """Fixed-width current side (scene + hand + state + readout) from the ABI."""
        h1 = self.encode_scene(enc["h1"], enc["h1_fine_points"])
        h2, theta_l = self.encode_hand(
            enc["h2_xyz"],
            enc["h2_bone"],
            enc["finger_q"],
            enc["finger_qd"],
            enc["bone_contact"],
        )
        state = self.state_mlp(enc["x_low"])
        readout = self.readout_token.to(state.dtype).expand(state.shape[0], -1, -1)
        return (
            torch.cat((h1, h2, state.unsqueeze(1), readout), dim=1),
            h1.shape[1],
            theta_l,
        )

    @staticmethod
    def _tower_output(current, scene_tokens, theta_l):
        bones = current[:, scene_tokens : scene_tokens + C.HEAD2_TOKENS]
        return {
            "bones": bones,
            "palm": bones[:, 0],
            "readout": current[:, -1],
            "state": current[:, -2],
            "theta": theta_l,
        }

    def forward_varlen(
        self, enc: dict[str, torch.Tensor], streams: dict[str, GS.GoalStream]
    ) -> dict[str, torch.Tensor]:
        """Same current side, variable-length object/hand streams instead of h3."""
        if self.varlen_trunk is None:
            raise RuntimeError("tower was built without stream_specs")
        current, scene_tokens, theta_l = self.current_tokens(enc)
        encoded = []
        for name in GS.STREAM_NAMES:
            tokens, key_mask, gate = self.stream_encoders[name](streams[name])
            if name == GS.HAND_STREAM and self.hand_ref_gain != 1.0:
                gate = gate * self.hand_ref_gain
            encoded.append((tokens, key_mask, gate))
        arguments = tuple(tensor for group in encoded for tensor in group)
        for layer in self.varlen_trunk:
            current = self._run_checkpoint(layer, current, *arguments)
        return self._tower_output(current, scene_tokens, theta_l)

    def head(
        self, out: dict[str, torch.Tensor], *, zero_theta: bool = False
    ) -> torch.Tensor:
        if self.is_critic:
            return self.value_head(
                torch.cat((out["readout"], out["state"]), dim=-1)
            ).squeeze(-1)
        child = out["bones"][:, self.joint_child]
        parent = out["bones"][:, self.joint_parent]
        theta = torch.zeros_like(out["theta"]) if zero_theta else out["theta"]
        hidden = self.finger_shared(torch.cat((child, parent, theta), dim=-1))
        fingers = (hidden * self.finger_w).sum(-1) + self.finger_b
        wrist = self.wrist_head(torch.cat((out["readout"], out["palm"]), dim=-1))
        # Layout must match config.ACTION_DIM (28): wrist 6 (slides + hinge
        # rates), then fingers 22. The force-bias channels are deleted
        # (2026-09-28): force authority is the held servo error itself.
        return torch.cat((wrist, fingers), dim=-1)


# ---------------------------------------------------------------------------
# observation expander: 453 current-state floats -> tower inputs (no parameters)
# ---------------------------------------------------------------------------


class ObsEncoder(nn.Module):
    """Pure ABI expansion. Only immutable geometry/index buffers are retained."""

    def __init__(
        self, buffers: CanonicalBuffers, skeleton_on: bool, force_mode: str = "on"
    ) -> None:
        super().__init__()
        for name in buffers.__dataclass_fields__:
            self.register_buffer(name, getattr(buffers, name).clone(), persistent=True)
        if self.obj_points.ndim != 3 or self.obj_points.shape[1:] != (64, 3):
            raise ValueError(
                "canonical object points must have explicit [part,64,3] axes"
            )
        self.num_parts = self.obj_points.shape[0]
        self.skeleton_on = skeleton_on
        if force_mode not in ("on", "off"):
            raise ValueError("force_mode must be 'on' or 'off'")
        # v0.5 FON/FOFF ablation routing, fixed at construction and never
        # switched mid-run: 'off' substitutes constant zeros for the
        # obj_force_obs field at this model-input boundary only. Physics,
        # reward and every other ABI field are untouched.
        self.force_mode = force_mode
        self._slices = {}
        offset = 0
        for name, width in C.abi_fields(self.num_parts):
            self._slices[name] = slice(offset, offset + width)
            offset += width
        self.abi_dim = offset
        fine = torch.empty(64, dtype=torch.long)
        fine[self.fine_groups] = torch.arange(16)[:, None]
        # Reorder only the derived point view, not canonical data/hash. Each
        # max pool becomes a contiguous reshape reduction with identical tokens.
        fine_order = self.fine_groups.reshape(-1)
        coarse_order = self.coarse_point_idx[self.coarse_pt_groups.reshape(-1)]
        self.register_buffer(
            "fine_points", self.obj_points[:, fine_order], persistent=True
        )
        self.register_buffer(
            "coarse_points", self.obj_points[:, coarse_order], persistent=True
        )
        self.register_buffer("fine_id", fine[fine_order], persistent=True)
        self.register_buffer(
            "coarse_id", self.coarse_of_fine[fine[fine_order]], persistent=True
        )
        self.register_buffer("coarse_fine_id", fine[coarse_order], persistent=True)
        self.register_buffer(
            "coarse_coarse_id", self.coarse_of_fine[fine[coarse_order]], persistent=True
        )
        n = self.num_parts
        self.register_buffer(
            "bone_ids", torch.arange(16).repeat_interleave(16), persistent=True
        )
        self.register_buffer(
            "past_dt", -torch.tensor(C.PAST_FRAMES) * C.CONTROL_DT, persistent=True
        )
        desk_ids = torch.empty(64, dtype=torch.long)
        desk_ids[self.desk_groups] = torch.arange(16)[:, None]
        self.register_buffer("desk_ids", desk_ids, persistent=True)
        self.register_buffer(
            "part_fine", torch.arange(n).repeat_interleave(64), persistent=True
        )
        self.register_buffer(
            "part_coarse", torch.arange(n).repeat_interleave(32), persistent=True
        )

    def _f(self, abi, name):
        return abi[..., self._slices[name]]

    def _object_features(self, xyz, dt, flag, frames, *, coarse):
        batch = xyz.shape[0]
        fine_id = self.coarse_fine_id if coarse else self.fine_id
        coarse_id = self.coarse_coarse_id if coarse else self.coarse_id
        part = self.part_coarse if coarse else self.part_fine
        return {
            "xyz": xyz.reshape(batch, -1, 3),
            "sem": torch.ones_like(part).repeat(frames).expand(batch, -1),
            "part": part.repeat(frames).expand(batch, -1),
            "fine": fine_id.repeat(frames * self.num_parts).expand(batch, -1),
            "coarse": coarse_id.repeat(frames * self.num_parts).expand(batch, -1),
            "dt": dt,
            "flag": flag,
        }

    def encode(self, abi, route="current"):
        batch = abi.shape[0]
        if abi.shape[-1] != self.abi_dim:
            raise ValueError(f"expected ABI width {self.abi_dim}, got {abi.shape[-1]}")
        n = self.num_parts
        obj_pose = self._f(abi, "obj_part_pose").reshape(batch, n, 7)
        past_pose = self._f(abi, "obj_past_pose").reshape(batch, 5, n, 7)
        past_pad = self._f(abi, "obj_past_pad")
        hand_pose = self._f(abi, "hand_body_pose").reshape(batch, 16, 7)
        zeros = abi.new_zeros(batch, n * 64)
        cur = self._object_features(
            place(self.fine_points, obj_pose), zeros, zeros, 1, coarse=False
        )
        desk_zero = abi.new_zeros(batch, 64)
        desk = {
            "xyz": self.desk_points.expand(batch, -1, -1),
            "sem": torch.zeros_like(self.desk_ids).expand(batch, -1),
            "part": torch.zeros_like(self.desk_ids).expand(batch, -1),
            "fine": self.desk_ids.expand(batch, -1),
            "coarse": torch.zeros_like(self.desk_ids).expand(batch, -1),
            "dt": desk_zero,
            "flag": desk_zero,
        }
        past = self._object_features(
            place(self.coarse_points, past_pose),
            self.past_dt.repeat_interleave(n * 32).expand(batch, -1),
            past_pad.repeat_interleave(n * 32, -1),
            5,
            coarse=True,
        )
        h1 = {key: torch.cat((cur[key], desk[key], past[key]), 1) for key in cur}
        palm = hand_pose[:, 0]
        tip_pose = hand_pose[:, self.tip_bone]
        tips = (
            quat_wxyz_to_mat(tip_pose[..., 3:]) @ self.tip_offsets[..., None]
        ).squeeze(-1) + tip_pose[..., :3]
        origin = obj_pose[:, 0, :3]
        q = self._f(abi, "finger_q")
        qd = self._f(abi, "finger_qd")
        contact = self._f(abi, "bone_contact").reshape(batch, 16, 3)
        physical = torch.cat((q, qd, contact.flatten(1)), -1)
        # Pose coordinates are already /sigma: convert back to metres before /0.1.
        # The v0.5 force observation (60 floats) is appended after the skeleton
        # block and is fed in both skeleton modes: it is the new physical state
        # evidence, not a skeleton-stream duplicate. force_mode='off' (the FOFF
        # ablation arm) routes the field by name through a constant-zero
        # substitute at this shared input boundary, so both towers (actor and
        # critic) lose the information together and the ABI itself is never
        # mutated: the environment keeps computing and recording real forces.
        force_obs = self._f(abi, "obj_force_obs")
        if self.force_mode == "off":
            force_obs = torch.zeros_like(force_obs)
        x_low = torch.cat(
            (
                palm[:, :3],
                rot_6d(palm[:, 3:]),
                origin,
                rot_6d(obj_pose[:, 0, 3:]),
                self._f(abi, "wrist_twist"),
                self._f(abi, "obj_twist"),
                self._f(abi, "mass_obs"),
                ((tips - origin[:, None]) * C.SCENE_SIGMA / 0.1).flatten(1),
                ((hand_pose[..., :3] - origin[:, None]) * C.SCENE_SIGMA / 0.1).flatten(
                    1
                ),
                self._f(abi, "prev_action"),
                torch.zeros_like(physical) if self.skeleton_on else physical,
                force_obs,
                self._f(abi, "support_ratio_obs"),
                self._f(abi, "contact_bone_count_obs"),
                self._f(abi, "servo_error"),
                self._f(abi, "terminal_hand_goal"),
            ),
            -1,
        )
        shared = {
            "h1": h1,
            "h2_xyz": place(self.hand_points, hand_pose).flatten(1, 2),
            "h1_fine_points": n * 64 + 64,
            "h2_bone": self.bone_ids.expand(batch, -1),
            "finger_q": q,
            "finger_qd": qd,
            "bone_contact": contact,
        }
        if route != "current":
            raise RuntimeError(FIXED_GOAL_ABI_REMOVED_ERROR)
        return {
            "actor": dict(shared, x_low=x_low),
            "critic": dict(
                shared,
                x_low=torch.cat(
                    (
                        x_low,
                        self._f(abi, "priv_13"),
                        self._f(abi, "critic_keypoint_error"),
                    ),
                    -1,
                ),
            ),
        }


FIXED_GOAL_ABI_REMOVED_ERROR = (
    "the fixed-goal ABI observation channel was removed in v3.abi.9; this build "
    "only serves the structured varlen path (--structured-streams)"
)


# ---------------------------------------------------------------------------
