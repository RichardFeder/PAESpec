"""Helpers for loading YAML config and applying defaults to argparse parsers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def _import_yaml_module():
    """Import YAML lazily so scripts can still start without PyYAML installed."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config-yaml support. Install with: pip install pyyaml"
        ) from exc
    return yaml


def _normalize_key(key: str) -> str:
    """Normalize keys so YAML can use either hyphens or underscores."""
    return key.replace('-', '_')


def _normalize_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize keys recursively for nested mappings."""
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        nkey = _normalize_key(str(key))
        if isinstance(value, dict):
            normalized[nkey] = _normalize_mapping(value)
        else:
            normalized[nkey] = value
    return normalized


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML file and return normalized dict payload."""
    yaml = _import_yaml_module()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    with path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config YAML must contain a mapping at top level: {path}")

    return _normalize_mapping(raw)


def _extract_section(config: Dict[str, Any], section: str | None) -> Dict[str, Any]:
    """Merge common + section values (section keys win)."""
    merged: Dict[str, Any] = {}

    common = config.get('common', {})
    if isinstance(common, dict):
        merged.update(common)

    if section:
        section_data = config.get(section, {})
        if not isinstance(section_data, dict):
            raise ValueError(f"Section '{section}' must be a mapping in config YAML")
        merged.update(section_data)

    if not section:
        for key, value in config.items():
            if key == 'common':
                continue
            if isinstance(value, dict):
                continue
            merged[key] = value

    return merged


def apply_yaml_defaults(
    parser: argparse.ArgumentParser,
    config_path: str,
    section: str | None = None,
) -> Dict[str, Any]:
    """Apply YAML values to parser defaults and return applied key/value mapping."""
    config = load_yaml_config(config_path)
    section_values = _extract_section(config, section)

    known_dests = {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }

    unknown_keys = sorted(set(section_values.keys()) - known_dests)
    for key in unknown_keys:
        print(f"[config-yaml] warning: unknown key '{key}' ignored")

    defaults = {k: v for k, v in section_values.items() if k in known_dests}
    parser.set_defaults(**defaults)
    return defaults
