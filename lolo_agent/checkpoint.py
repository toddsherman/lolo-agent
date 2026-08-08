from __future__ import annotations

import json
from pathlib import Path

from .world_model import EmpiricalWorldModel


def save_model(model: EmpiricalWorldModel, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.checkpoint_bytes()
    path.write_bytes(data)
    return model.checkpoint_digest


def load_model(path: Path, frozen: bool = False) -> EmpiricalWorldModel:
    model = EmpiricalWorldModel.from_dict(json.loads(path.read_text()))
    if frozen:
        model.freeze()
    return model

