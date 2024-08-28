from __future__ import annotations

import typing as _tp
from mcl.internal import machine_type, machine_op, struct_type

T = _tp.TypeVar("T")


@machine_type(builtin=True, final=True)
class i32:
    __machine_repr__ = "i32"

    def __add__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_add", i32, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is i32:
            return machine_op("int_eq", bool, self, other)
        else:
            return NotImplemented

    def cast(self, restype: _tp.Type[T]) -> T:
        return machine_op("cast", restype, self)


@machine_type(builtin=True, final=True)
class i64:
    __machine_repr__ = "i64"

    def __add__(self, other) -> i64:
        if type(other) is i64:
            return machine_op("int_add", i64, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is i64:
            return machine_op("int_eq", bool, self, other)
        else:
            return NotImplemented


@machine_type(builtin=True, final=True)
class intp:
    __machine_repr__ = "intptr"

    if _tp.TYPE_CHECKING:

        def __init__(self, v): ...

    def __add__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_add", intp, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_mul", intp, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is intp:
            return machine_op("int_eq", bool, self, other)
        else:
            return NotImplemented


@machine_type(builtin=True, final=True)
class pointer:
    __machine_repr__ = "ptr"

    @classmethod
    def new(cls, nbytes: intp) -> pointer:
        return machine_op("malloc", pointer, nbytes)
