import yaml
import os
from pathlib import Path
from typing import Any, Dict


class ConfigNode(dict):
    """
    Allows attribute-style access:
        config.paths.raw.videos
    while still behaving like a dict.
    """
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return ConfigNode(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _expand_env_vars(value: Any) -> Any:
    """
    Recursively expand ${VAR} expressions inside strings.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _resolve_interpolations(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolves ${paths.data_root}/... style interpolations.
    """
    # Convert to string for environment expansion
    raw_yaml = yaml.dump(config_dict)
    expanded_yaml = os.path.expandvars(raw_yaml)
    return yaml.safe_load(expanded_yaml)


class Config:
    _instance = None

    @classmethod
    def load(cls, path: str = "config.yaml"):
        """
        Loads the config once and returns a singleton.
        """
        if cls._instance is not None:
            return cls._instance

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        # First pass: expand environment variables
        expanded = _expand_env_vars(raw_cfg)

        # Second pass: resolve ${paths.data_root} interpolations
        resolved = _resolve_interpolations(expanded)

        cls._instance = ConfigNode(resolved)
        return cls._instance


# Convenience accessor
config = Config.load()
