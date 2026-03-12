from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = ROOT_DIR / ".env"
DEFAULT_YAML_FILE = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    haarcascade: Path
    trainer: Path


@dataclass(frozen=True)
class PathConfig:
    database: Path
    training_images_dir: Path
    attendance_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class CameraConfig:
    source: int | str


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig
    paths: PathConfig
    camera: CameraConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "haarcascade": "models/haarcascade_frontalface_default.xml",
        "trainer": "data/training/Trainner.yml",
    },
    "paths": {
        "database": "data/student_details/StudentDetails.csv",
        "training_images_dir": "data/training_images",
        "attendance_dir": "data/attendance",
        "output_dir": "output",
    },
    "camera": {"source": 0},
}

ENV_TO_CONFIG_KEY = {
    "FRAS_MODEL_HAARCASCADE": "models.haarcascade",
    "FRAS_MODEL_TRAINER": "models.trainer",
    "FRAS_DB_PATH": "paths.database",
    "FRAS_TRAINING_IMAGES_DIR": "paths.training_images_dir",
    "FRAS_ATTENDANCE_DIR": "paths.attendance_dir",
    "FRAS_OUTPUT_DIR": "paths.output_dir",
    "FRAS_CAMERA_SOURCE": "camera.source",
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _load_dotenv(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip("\"'")
    return values


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _load_yaml_fallback(yaml_path: Path) -> dict[str, Any]:
    """Minimal YAML loader for simple nested mapping config files."""
    parsed: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1]
            parsed[current_section] = {}
            continue
        if current_section and line.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            parsed[current_section][key.strip()] = _parse_scalar(value)
    return parsed


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        return {}
    if yaml is not None:
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return loaded or {}
    return _load_yaml_fallback(yaml_path)


def _coerce_camera_source(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(
    *,
    env_path: Path = DEFAULT_ENV_FILE,
    yaml_path: Path = DEFAULT_YAML_FILE,
    validate: bool = True,
) -> AppConfig:
    config_root = yaml_path.parent if yaml_path.exists() else ROOT_DIR
    yaml_config = _load_yaml(yaml_path)

    merged = _merge_dict(DEFAULT_CONFIG, yaml_config)

    dotenv_values = _load_dotenv(env_path)
    for env_key, dotted_key in ENV_TO_CONFIG_KEY.items():
        env_value = os.environ.get(env_key, dotenv_values.get(env_key))
        if env_value is not None and env_value != "":
            _set_nested(merged, dotted_key, env_value)

    app_config = AppConfig(
        models=ModelConfig(
            haarcascade=_resolve_path(merged["models"]["haarcascade"], config_root),
            trainer=_resolve_path(merged["models"]["trainer"], config_root),
        ),
        paths=PathConfig(
            database=_resolve_path(merged["paths"]["database"], config_root),
            training_images_dir=_resolve_path(
                merged["paths"]["training_images_dir"], config_root
            ),
            attendance_dir=_resolve_path(merged["paths"]["attendance_dir"], config_root),
            output_dir=_resolve_path(merged["paths"]["output_dir"], config_root),
        ),
        camera=CameraConfig(source=_coerce_camera_source(merged["camera"]["source"])),
    )

    if validate:
        validate_config(app_config)

    return app_config


def validate_config(config: AppConfig) -> None:
    required_files = {
        "models.haarcascade": config.models.haarcascade,
        "models.trainer": config.models.trainer,
        "paths.database": config.paths.database,
    }
    required_dirs = {
        "paths.training_images_dir": config.paths.training_images_dir,
        "paths.attendance_dir": config.paths.attendance_dir,
        "paths.output_dir": config.paths.output_dir,
    }

    errors: list[str] = []
    for key, file_path in required_files.items():
        if not file_path.exists() or not file_path.is_file():
            errors.append(f"- Missing required file `{key}`: {file_path}")

    for key, dir_path in required_dirs.items():
        if not dir_path.exists() or not dir_path.is_dir():
            errors.append(f"- Missing required directory `{key}`: {dir_path}")

    if errors:
        raise FileNotFoundError(
            "Configuration validation failed. Ensure the following paths exist:\n"
            + "\n".join(errors)
        )
