from __future__ import annotations

from mcl.machine_types import i32, memref
from mcl.ndarray import Array, to_scalar_array
from mcl.vm import machine_op

def array_exp(array: Array):
    new_memref = machine_op("memref_exp", memref, array.data)
    return Array(dtype=array.dtype, data=new_memref)

def array_sqrt(array: Array):
    new_memref = machine_op("memref_sqrt", memref, array.data)
    return Array(dtype=array.dtype, data=new_memref)

def array_sum(array: Array, axis: int = -1, keepdims: bool = False):
    new_memref = machine_op("memref_sum", memref, array.data, i32(axis), i32(keepdims))
    return Array(dtype=array.dtype, data=new_memref)

def array_max(array: Array, axis: int = -1, keepdims: bool = False):
    new_memref = machine_op("memref_max", memref, array.data, i32(axis), i32(keepdims))
    return Array(dtype=array.dtype, data=new_memref)

def array_maximum(array: Array, other):
    if isinstance(other, Array):
        other = other.data
    else:
        other = to_scalar_array(other, array.dtype).data

    new_memref = machine_op("memref_maximum", memref, array.data, other)
    return Array(dtype=array.dtype, data=new_memref)

def array_matmul(matrix_1: Array, matrix_2: Array):
    new_memref = machine_op("memref_matmul", memref, matrix_1.data, matrix_2.data)
    return Array(dtype=matrix_1.dtype, data=new_memref)
