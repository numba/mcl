from __future__ import annotations

import typing as _tp

from mcl.vm import machine_op, machine_type

T = _tp.TypeVar("T")


@machine_type(builtin=True, final=True)
class i32:
    __machine_repr__ = "i32"

    def __add__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_add", i32, self, other)
        else:
            return NotImplemented

    def __sub__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_sub", i32, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_mul", i32, self, other)
        else:
            return NotImplemented

    def __floordiv__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_floordiv", i32, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is i32:
            return machine_op("int_eq", bool, self, other)
        else:
            return NotImplemented

    def __lt__(self, other) -> bool:
        if type(other) is i32:
            return machine_op("int_lt", bool, self, other)
        else:
            return NotImplemented
    
    def __mod__(self, other) -> i32:
        if type(other) is i32:
            return machine_op("int_mod", i32, self, other)
        else:
            return NotImplemented


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

    def __sub__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_sub", intp, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_mul", intp, self, other)
        else:
            return NotImplemented

    def __floordiv__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_floordiv", intp, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is intp:
            return machine_op("int_eq", bool, self, other)
        else:
            return NotImplemented

    def __lt__(self, other) -> bool:
        if type(other) is intp:
            return machine_op("int_lt", bool, self, other)
        else:
            return NotImplemented
    
    def __mod__(self, other) -> intp:
        if type(other) is intp:
            return machine_op("int_mod", intp, self, other)
        else:
            return NotImplemented

    def __index__(self) -> int:
        return machine_op("cast", int, self)
    
    def __hash__(self):
        return hash(int(self))

@machine_type(builtin=True, final=True)
class f32:
    __machine_repr__ = "f32"

    def __add__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_add", f32, self, other)
        else:
            return NotImplemented

    def __sub__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_sub", f32, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_mul", f32, self, other)
        else:
            return NotImplemented

    def __floordiv__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_floordiv", f32, self, other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_truediv", f32, self, other)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is f32:
            return machine_op("float_eq", bool, self, other)
        else:
            return NotImplemented

    def __lt__(self, other) -> bool:
        if type(other) is f32:
            return machine_op("float_lt", bool, self, other)
        else:
            return NotImplemented
    
    def __mod__(self, other) -> f32:
        if type(other) is f32:
            return machine_op("float_mod", f32, self, other)
        else:
            return NotImplemented


@machine_type(builtin=True, final=True)
class memref[T]:
    __machine_repr__ = "memref"

    @classmethod
    def alloc(cls, shape: tuple[intp, ...], type: _tp.Type[T]) -> memref[T]:
        return machine_op("memref_alloc", memref, shape, type)

    @classmethod
    def alloc_random(cls, shape: tuple[intp, ...], type: _tp.Type[T]) -> memref[T]:
        return machine_op("memref_alloc_random", memref, shape, type)

    @property
    def shape(self) -> tuple[intp, ...]:
        return machine_op("memref_shape", tuple, self)

    @property
    def strides(self) -> tuple[intp, ...]:
        return machine_op("memref_strides", tuple, self)

    @property
    def offset(self) -> intp:
        return machine_op("memref_offset", tuple, self)

    def copy(self) -> memref[T]:
        return machine_op("memref_copy", memref, self)

    def store(self, indices: tuple[intp, ...], value: T) -> None:
        return machine_op("memref_store", None, self, indices, value)

    def load(self, indices: tuple[intp, ...], restype: _tp.Type[T]) -> T:
        return machine_op("memref_load", restype, self, indices)

    def view(self, shape: tuple[intp, ...], strides: tuple[intp, ...], offset: intp) -> memref[T]:
        return machine_op("memref_view", memref, self, shape, strides, offset)
