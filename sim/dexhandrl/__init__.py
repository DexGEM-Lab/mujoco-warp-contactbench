"""DexHandRL migration helpers for the MJX-Warp ContactBench repo."""

from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

try:  # pragma: no cover - optional training dependency
    from sim.dexhandrl.skrl_models import DexHandSkrlPolicy, DexHandSkrlValue, SkrlDexHandModelConfig
except ImportError:  # pragma: no cover - base env should stay importable without skrl
    DexHandSkrlPolicy = None
    DexHandSkrlValue = None
    SkrlDexHandModelConfig = None

__all__ = [
    "DexHandRLMJXEnv",
    "DexHandRLMJXEnvConfig",
    "DexHandSkrlPolicy",
    "DexHandSkrlValue",
    "SkrlDexHandModelConfig",
]
