from __future__ import annotations

import typing as _tp

from mcl.builtins import tuple_cast
from mcl.machine_types import i32, intp, memref
from mcl.vm import struct_type


@struct_type()
class DType:
    type: _tp.Type[Generic]


@struct_type()
class Generic:
    @classmethod
    def from_memory(cls, data: memref, index: tuple[intp, ...]) -> Generic:
        raise NotImplementedError


@struct_type()
class Number(Generic):
    pass


@struct_type()
class Integer(Number):
    pass


type _IntLike = int | intp
type _Indices = tuple[_IntLike, ...] | _IntLike


@struct_type()
class Int32(Integer):
    value: i32

    @classmethod
    def from_memory(cls, data: memref, index: tuple[intp, ...]) -> Int32:
        return cls(value=data.getitem(index, i32))

    def __eq__(self, other) -> bool:
        if type(other) is i32:
            return self.value == other
        elif isinstance(other, Int32):
            return self.value == other.value
        return NotImplemented


@struct_type(final=True)
class Array[T]:
    dtype: DType
    data: memref

    @property
    def shape(self) -> tuple[intp, ...]:
        return self.data.shape

    @property
    def ndim(self) -> intp:
        return intp(len(self.shape))

    def __setitem__(self, idx: _Indices, value: T):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = tuple_cast(intp, idx)
        self.data.setitem(idx, value)

    def __getitem__(self, idx: _Indices) -> Array[T] | Generic:
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = tuple_cast(intp, idx)
        return self.dtype.type.from_memory(self.data, idx)
