"""
Provide utilities of common pattern to avoid high-level construct
during bootstrapping
"""

from __future__ import annotations

import typing as _tp

from mcl.vm import machine_op


def tuple_cast[T, R](resty: _tp.Type[R], tup: tuple[T, ...]) -> tuple[R, ...]:
    """Equivalent to `tuple(map(resty, tup))`"""
    return machine_op("tuple_cast", tuple, resty, tup)
