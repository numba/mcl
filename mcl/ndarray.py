from __future__ import annotations

import typing as _tp

from mcl.builtins import tuple_cast
from mcl.machine_types import i32, intp, memref
from mcl.vm import struct_type
from mcl.dialects import LoopNestAPI


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
        return cls(value=data.load(index, i32))

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
            raise ValueError("Array indexing is not supported")
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

        if any(isinstance(i, Array) for i in idx):
            new_shape, new_strides, new_offset = self.new_arrayinfo(idx)
            raise ValueError("Array indexing is not supported")
        elif any(isinstance(i, slice) for i in idx):
            new_shape, new_strides, new_offset = self.new_arrayinfo(idx)
            new_memref = self.data.view(new_shape, new_strides, new_offset)
            return Array(dtype=self.dtype, data=new_memref)
        else:
            # All indices are integers
            idx = tuple_cast(intp, idx)
            # TODO: There's no assertion that checks if idx is within bounds.
            return self.dtype.type.from_memory(self.data, idx)

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

    def copy(self) -> Array[T]:
        return Array(dtype=self.dtype, data=self.data.copy())

    def print(self) -> None:
        res = []
        for idx in LoopNestAPI.from_tuple(self.shape):
            res.append(self[idx].value)
        print(res)
