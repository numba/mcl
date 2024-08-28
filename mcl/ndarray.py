from __future__ import annotations
import typing as _tp
from mcl.internal import struct_type
from mcl.machine_types import intp, i32
from mcl.memref import MemRef


@struct_type()
class DType:
    name: str


@struct_type()
class IntDType(DType):
    bitwidth: i32


@struct_type(final=True)
class Array[T]:
    dtype: DType
    data: MemRef

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

    def __getitem__(self, idx: tuple[intp, ...] | intp) -> T:
        if not isinstance(idx, tuple):
            idx = (idx,)
        # FIXME: handle dtype
        return self.data.getitem(idx, i32)
