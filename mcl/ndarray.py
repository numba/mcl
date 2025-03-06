from __future__ import annotations

import typing as _tp

from mcl.builtins import tuple_cast
from mcl.machine_types import i32, intp, memref, f32
from mcl.vm import struct_type, machine_op
from mcl.dialects import LoopNestAPI
from mcl.vm import _get_machine_value


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

@struct_type()
class Float(Number):
    pass

type _IntLike = int | intp
type _Indices = tuple[_IntLike, ...] | _IntLike


@struct_type()
class Int32(Integer):
    value: i32

    @classmethod
    def from_memory(cls, data: memref, index: tuple[intp, ...]) -> Int32:
        return cls(value=data.load(index, i32))

    def __eq__(self, other) -> bool:
        if type(other) is i32:
            return self.value == other
        elif isinstance(other, Int32):
            return self.value == other.value
        return NotImplemented

@struct_type()
class Float32(Float):
    value: f32

    @classmethod
    def from_memory(cls, data: memref, index: tuple[intp, ...]) -> Float32:
        return cls(value=data.load(index, f32))

    def __eq__(self, other) -> bool:
        if type(other) is f32:
            return self.value == other
        elif isinstance(other, Float32):
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
    def strides(self) -> tuple[intp, ...]:
        return self.data.strides

    @property
    def ndim(self) -> intp:
        return intp(len(self.shape))

    def __setitem__(self, idx: _Indices, value: T):
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) < len(self.shape):
            idx = idx + (slice(None),) * (len(self.shape) - len(idx))

        if any(isinstance(i, Array) for i in idx):
            raise ValueError("Array indexing is not supported in setitem")
        elif any(isinstance(i, slice) for i in idx):
            array_view = self.__getitem__(idx)
            if isinstance(value, Array):
                if array_view.shape != value.shape:
                    raise ValueError("Shapes do not match")
                for idx_ in LoopNestAPI.from_tuple(array_view.shape):
                    array_view[idx_] = value[idx_]
            else:
                if isinstance(value, Integer):
                    value = value.value
                for idx in LoopNestAPI.from_tuple(array_view.shape):
                    array_view[idx] = value
        else:
            idx = tuple_cast(intp, idx)
            # TODO: There's no assertion that checks if idx is within bounds.
            if isinstance(value, Integer):
                value = value.value
            self.data.store(idx, value)

    def __getitem__(self, idx: _Indices) -> Array[T] | Generic:
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) < len(self.shape):
            idx = idx + (slice(None),) * (len(self.shape) - len(idx))

        if self.is_advanced(idx):
            res_shape, subspace_shape, subspace_offset = self.fancy_shape(idx)
            new_memref = memref.alloc(res_shape, i32)
            res_array = Array(dtype=self.dtype, data=new_memref)

            for _idx in LoopNestAPI.from_tuple(subspace_shape):
                res_idx = tuple([slice(None)] * subspace_offset + list(_idx))
                array_idx = tuple([i if isinstance(i, (int, slice)) else intp(i[_idx].value) for i in idx])
                res_array[*res_idx] = self[*array_idx]
            return res_array
        elif any(isinstance(i, slice) for i in idx):
            if any(isinstance(i, Array) for i in idx):
                raise ValueError("Singular array indexing is not supported")
            new_shape, new_strides, new_offset = self.new_arrayinfo(idx)
            new_memref = self.data.view(new_shape, new_strides, new_offset)
            return Array(dtype=self.dtype, data=new_memref)
        else:
            # All indices are integers
            idx = tuple_cast(intp, idx)
            # TODO: There's no assertion that checks if idx is within bounds.
            return self.dtype.type.from_memory(self.data, idx)
    
    def __add__(self, other: Array[T]) -> Array[T]:
        if isinstance(other, Array):
            other_memref = other.data
        else:
            other_memref = to_scalar_array(other, self.dtype).data
        machine_op("memref_add", memref, self.data, other_memref)
        return self

    def __truediv__(self, other: Array[T]) -> Array[T]:
        if isinstance(other, Array):
            other_memref = other.data
        else:
            other_memref = to_scalar_array(other, self.dtype).data
        machine_op("memref_truediv", memref, self.data, other_memref)
        return self

    def __sub__(self, other: Array[T]) -> Array[T]:
        if isinstance(other, Array):
            other_memref = other.data
        else:
            other_memref = to_scalar_array(other, self.dtype).data
        machine_op("memref_sub", memref, self.data, other_memref)
        return self

    def broadcast_to(self, shape: tuple[intp, ...]) -> None:
        # This function can also serve as a assertion
        new_shape = self.broadcast_shapes(self.shape, shape)
        assert new_shape == shape

        old_shape = self.shape
        old_strides = self.strides
        new_ndim = len(new_shape)
        ndim_diff = len(new_shape) - len(old_shape)
        extended_shape = (intp(1),) * ndim_diff + old_shape
        extended_strides = (intp(0),) * ndim_diff + old_strides
        new_strides = [intp(0)] * new_ndim
        for i in range(new_ndim):
            if new_shape[i] == extended_shape[i]:
                new_strides[i] = extended_strides[i] * (new_shape[i] // extended_shape[i])

        new_strides = tuple(new_strides)

        self.data = self.data.view(new_shape, new_strides, self.data.offset)

    def reshape(self, shape: tuple[intp, ...], copy: bool=True) -> None:
        # Check if this is a valid reshape
        num_elems_orig = intp(1)
        for i in self.shape:
            num_elems_orig *= i
        num_elems_new = intp(1)
        has_neg_one = False
        neg_one_idx = -1
        for idx, i in enumerate(shape):
            if i == intp(-1):
                if has_neg_one:
                    raise ValueError("Only one dimension can be -1")
                has_neg_one = True
                neg_one_idx = idx
            else:
                num_elems_new *= i
        
        if has_neg_one:
            assert num_elems_orig % num_elems_new == intp(0), f"Cannot reshape array of shape {self.shape} to shape {shape}"

            shape = shape[:neg_one_idx] + (num_elems_orig // num_elems_new,) + shape[neg_one_idx + 1:]
        else:
            assert num_elems_orig == num_elems_new, f"Cannot reshape array of shape {self.shape} to shape {shape}"

        new_strides = [intp(0)] * len(shape)

        if copy:
            self = self.copy()
        
        # TODO: Check if this logic is true
        # If we are at this point, we can assume that the array is contiguous
        # If it wasn't earlier the copy would have made it contiguous

        # TODO: This should be intp(size_of_element) instead of 4
        new_strides[-1] = intp(4)
        
        for i in range(len(shape) - 2, -1, -1):
            new_strides[i] = new_strides[i + 1] * shape[i + 1]

        self.data = self.data.view(shape, new_strides, self.data.offset)
        return self

    def transpose(self, axis: tuple[intp, ...] = None) -> None:
        # Check is axis is valid
        if axis is None:
            axis = tuple([intp(i) for i in range(self.ndim - 1, -1, -1)])

        assert intp(len(axis)) == self.ndim
        assert set(axis) == set([intp(i) for i in range(self.ndim)])

        new_shape = [intp(0)] * self.ndim
        new_strides = [intp(0)] * self.ndim

        # These instances of get_machine_value needs to be removed
        for i, j in enumerate(axis):
            new_shape[i] = self.shape[_get_machine_value(j)]
            new_strides[i] = self.strides[_get_machine_value(j)]

        self.data = self.data.view(tuple(new_shape), tuple(new_strides), self.data.offset)

        return self

    @classmethod
    def is_advanced(cls, idx: _Indices) -> bool:
        return any((isinstance(i, Array) and i.ndim > intp(0)) for i in idx)

    @classmethod
    def broadcast_shapes(cls, *shapes) -> tuple[intp, ...]:
        if len(shapes) == 0:
            raise ValueError("At least one shape is required")

        max_ndim = max(map(len, shapes))
        identity = intp(1)

        # The length can be determined at compile time
        # So something like tuple setitem can be used
        # here instead of building a list.
        result = [identity] * max_ndim

        for i in range(1, max_ndim + 1):
            curr_axis_shape = identity
            for shape in shapes:
                if len(shape) + 1 > i:
                    if shape[-i] == identity:
                        pass
                    elif curr_axis_shape is identity:
                        curr_axis_shape = shape[-i]
                    elif curr_axis_shape != shape[-i]:
                        raise ValueError("Shapes are not broadcastable")
            result[-i] = curr_axis_shape

        return tuple(result)

    def new_arrayinfo(self, idx: _Indices) -> tuple[intp, ...]:
        res_ndim = self.ndim

        for idx_ in idx:
            if isinstance(idx_, (int, intp)):
                res_ndim -= intp(1)

        res_shape = [intp(0)] * res_ndim
        res_strides = [intp(0)] * res_ndim
        res_offset = intp(0)

        curr_idx = 0
        for i, idx_ in enumerate(idx):
            if isinstance(idx_, slice):
                if idx_.step is not None:
                    raise ValueError("Slicing with step is not supported")
                
                if idx_.start is not None and idx_.start < 0:
                    raise ValueError("Negative slicing is not supported")
                
                if idx_.stop is not None and idx_.stop < 0:
                    raise ValueError("Negative slicing is not supported")
                
                if idx_.start is None:
                    idx_start = intp(0)
                else:
                    idx_start = intp(idx_.start)
                
                if idx_.stop is None:
                    idx_stop = self.shape[i]
                else:
                    idx_stop = intp(idx_.stop)
                res_shape[curr_idx] = idx_stop - idx_start
                res_strides[curr_idx] = self.strides[i]
                res_offset += idx_start * self.strides[i]
                curr_idx += 1
            elif isinstance(idx_, (int, intp)):
                res_offset += intp(idx_) * self.strides[i]

        return tuple(res_shape), tuple(res_strides), res_offset

    def fancy_shape(self, idx: _Indices) -> tuple[intp, ...]:
        res_ndim = self.ndim
        subspace_offset = 0
        num_subspaces = 0
        in_subspace = False
        num_slices = 0
        array_shapes = []
        slice_shapes = []

        for i, idx_ in enumerate(idx):
            if isinstance(idx_, (int, intp)):
                res_ndim -= intp(1)
                in_subspace = False
                if subspace_offset <= 0:
                    subspace_offset -= 1
            elif isinstance(idx_, Array):
                if not in_subspace:
                    in_subspace = True
                    num_subspaces += 1
                if subspace_offset <= 0:
                    subspace_offset += i
                res_ndim -= intp(1)
                array_shapes.append(idx_.shape)
            elif isinstance(idx_, slice):
                num_slices += 1
                in_subspace = False
                if idx_.start is None:
                    idx_start = intp(0)
                else:
                    idx_start = intp(idx_.start)
                if idx_.stop is None:
                    idx_stop = self.shape[i]
                else:
                    idx_stop = intp(idx_.stop)
                slice_shapes.append(idx_stop - idx_start)
            else:
                raise ValueError("Invalid index")

        if num_subspaces > 1:
            subspace_offset = 0

        subspace_shape = self.broadcast_shapes(*array_shapes)

        res_shape = slice_shapes[:subspace_offset] + list(subspace_shape) + slice_shapes[subspace_offset:]

        return tuple(res_shape), tuple(subspace_shape), subspace_offset

    def copy(self) -> Array[T]:
        return Array(dtype=self.dtype, data=self.data.copy())

    def print(self) -> None:
        machine_op("memref_print", None, self.data)

    @classmethod
    def random(cls, shape: tuple[intp, ...]) -> None:
        return Array(dtype=DType(Float32), data=memref.alloc_random(shape, f32))

def to_scalar_array(data, dtype):
    temp_memref = machine_op("memref_alloc", memref, (intp(1),), f32)
    machine_op("memref_store", None, temp_memref, (i32(0),), data)
    return Array(dtype=dtype, data=temp_memref)
