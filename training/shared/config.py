"""Config loading and path resolution."""

from pathlib import Path
import yaml


def load_config(yaml_path: str) -> dict:
    """Load YAML config file and resolve relative paths."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config["_config_dir"] = str(yaml_path.parent)
    config["_config_path"] = str(yaml_path)
    return config


def resolve_path(config: dict, key: str, base: str = None) -> Path:
    """Resolve a config path relative to config dir or given base."""
    raw = config
    for k in key.split("."):
        raw = raw[k]
    p = Path(raw)
    if p.is_absolute():
        return p
    base_dir = Path(base) if base else Path(config.get("_config_dir", "."))
    return (base_dir / p).resolve()


def get_component_dir(component_num: int) -> Path:
    """Return the component folder path by number (e.g., 1 → 01_instrument_tracker)."""
    training_dir = Path(__file__).resolve().parent.parent
    matches = list(training_dir.glob(f"{component_num:02d}_*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Component {component_num:02d} not found in {training_dir}")
