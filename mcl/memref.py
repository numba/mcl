import typing as _tp
from mcl.internal import struct_type, machine_op
from mcl.machine_types import intp, pointer


@struct_type(builtin=True)
class MemRef:
    shape: tuple[intp, ...]
    strides: tuple[intp, ...]
    baseptr: pointer
    dataptr: pointer

    def setitem[T](self, indices: tuple[intp, ...], val: T):
        machine_op("memref_setitem", None, self, indices, val)

    def getitem[T](self, indices: tuple[intp, ...], restype: _tp.Type[T]) -> T:
        return machine_op("memref_getitem", restype, self, indices)
