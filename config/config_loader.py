"""
Cargador de configuración global del sistema.
Lee settings.yaml y expone la configuración como un objeto accesible.
"""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Carga y provee acceso a la configuración del sistema."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._path = Path(config_path)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Acceso encadenado: get('camera', 'width') -> 640"""
        value = self._data
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
        return value


# Instancia global reutilizable
config = Config()