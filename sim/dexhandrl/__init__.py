"""DexHandRL migration helpers for the MJX-Warp ContactBench repo."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig
    from sim.dexhandrl.skrl_models import DexHandSkrlPolicy, DexHandSkrlValue, SkrlDexHandModelConfig

__all__ = [
    "DexHandRLMJXEnv",
    "DexHandRLMJXEnvConfig",
    "DexHandSkrlPolicy",
    "DexHandSkrlValue",
    "SkrlDexHandModelConfig",
]


def __getattr__(name: str):
    """Resolve runtime-heavy public exports only when a caller requests them."""
    if name in {"DexHandRLMJXEnv", "DexHandRLMJXEnvConfig"}:
        from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

        globals().update(
            DexHandRLMJXEnv=DexHandRLMJXEnv,
            DexHandRLMJXEnvConfig=DexHandRLMJXEnvConfig,
        )
    elif name in {"DexHandSkrlPolicy", "DexHandSkrlValue", "SkrlDexHandModelConfig"}:
        try:  # pragma: no cover - optional training dependency
            from sim.dexhandrl.skrl_models import DexHandSkrlPolicy, DexHandSkrlValue, SkrlDexHandModelConfig
        except ImportError:  # pragma: no cover - base env should stay importable without skrl
            DexHandSkrlPolicy = DexHandSkrlValue = SkrlDexHandModelConfig = None
        globals().update(
            DexHandSkrlPolicy=DexHandSkrlPolicy,
            DexHandSkrlValue=DexHandSkrlValue,
            SkrlDexHandModelConfig=SkrlDexHandModelConfig,
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[name]
