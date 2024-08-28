from __future__ import annotations
import typing as _tp
from mcl.vm import struct_type
from mcl.machine_types import intp, i32, memref


@struct_type()
class DType:
    type: _tp.Type[Generic]


@struct_type()
class Generic:
    pass


@struct_type()
class Number(Generic):
    pass


@struct_type()
class Integer(Number):
    pass


@struct_type()
class Int32(Integer):
    value: i32

    @classmethod
    def from_memory(cls, data: MemRef, index: int) -> Int32:
        return cls(value=data.getitem(index, i32))

    def __eq__(self, other) -> bool:
        if type(other) is i32:
            return self.value == other
        elif isinstance(other, Int32):
            return self.value == other.value


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

    def __setitem__(self, idx: tuple[intp, ...] | intp, value: T):
        if not isinstance(idx, tuple):
            idx = (idx,)
        self.data.setitem(idx, value)

    def __getitem__(self, idx: tuple[intp, ...] | intp) -> Array[T]:
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self.dtype.type.from_memory(self.data, idx)
