"""Phase-2 CD depth augmentation (stub interface).

CD failed_stay/failed_update depth requires searching measurements that preserve the survivor
set under rule_engine constraints. This module documents the planned interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def augment_cd_failed_stay_depth(
    case: Dict[str, Any],
    *,
    n_redundant: int,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    del case, n_redundant, seed
    return None, "cd_failed_stay_depth_not_implemented_phase2"


def augment_cd_failed_update_depth(
    case: Dict[str, Any],
    *,
    delay_turns: int,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    del case, delay_turns, seed
    return None, "cd_failed_update_depth_not_implemented_phase2"
