import typing as _tp
from mcl.internal import struct_type
from mcl.machine_types import intp, i32


@struct_type()
class DType:
    name: str


@struct_type()
class IntDType(DType):
    bitwidth: i32


@struct_type(final=True)
class Array:
    shape: tuple[intp, ...]
    dtype: DType
