# pytest me
import pytest
from mcl.machine_types import i32, intp, memref
from mcl.ndarray import Array, DType, Int32

def test_array():
    shape = (intp(3), intp(4))
    i32_dtype = DType(Int32)
    print(i32_dtype)

    data = memref.alloc(shape, i32)
    print(data)

    ary = Array(dtype=i32_dtype, data=data)

    print(ary)
    print(ary.shape)
    print(ary.dtype)

    ary[0, 0] = i32(0xcafe)
    res = ary[intp(0), intp(0)]
    assert res == i32(0xcafe)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ary[i, j] = c
            c += i32(1)

    # read loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            got = ary[i, j]
            print(f"ary[{i}, {j}] = {got}")
            assert got == c
            c += i32(1)


def test_broadcast_array():
    shape = (intp(3), intp(1), intp(4))
    i32_dtype = DType(Int32)

    data = memref.alloc(shape, i32)
    ary = Array(dtype=i32_dtype, data=data)
    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[2]):
            ary[i, intp(0), j] = c
            c += i32(1)

    # Broadcast to new shape
    broad_shape = (intp(5), intp(3), intp(4), intp(4))
    ary.broadcast_to(broad_shape)

    assert ary.shape == broad_shape
    # read loop
    for i in range(broad_shape[0]):
        for k in range(broad_shape[2]):
            # Array has been 'replicated' along the first and third axis
            # as a view. (i, k)
            # Underlying data has not been modified
            c = i32(0)
            for j in range(broad_shape[1]):
                for l in range(broad_shape[3]):
                    got = ary[i, j, k, l]
                    # print(f"ary[{i}, {j}, {k}, {l}] = {got}")
                    assert got == c
                    c += i32(1)


def test_broadcast_shapes():
    shape_1 = (intp(3), intp(4))
    shape_2 = (intp(3), intp(1))
    shape_3 = (intp(4), intp(3), intp(1))

    out = Array.broadcast_shapes(shape_1, shape_2, shape_3)
    
    assert out == (intp(4), intp(3), intp(4))

def test_broadcast_shapes_2():
    # Test non broadcastable shape throws error
    shape_1 = (intp(3), intp(4))
    shape_2 = (intp(3), intp(1))
    shape_3 = (intp(4), intp(3), intp(2))

    with pytest.raises(ValueError, match="Shapes are not broadcastable"):
        out = Array.broadcast_shapes(shape_1, shape_2, shape_3)

def test_array_slice_getitem():
    shape = (intp(3), intp(4), intp(5))
    i32_dtype = DType(Int32)

    data = memref.alloc(shape, i32)
    ary = Array(dtype=i32_dtype, data=data)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ary[i, j, k] = c
                c += i32(1)

    # Slice
    slice_1 = ary[0]
    slice_2 = ary[0, slice(None), 0]
    slice_3 = ary[0, slice(1, 3)]

    assert slice_1.shape == (intp(4), intp(5))
    assert slice_2.shape == (intp(4),)
    assert slice_3.shape == (intp(2), intp(5))

    # Check contents
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                c += i32(1)
                assert ary[0, j, k] == slice_1[j, k]
                assert ary[0, j, 0] == slice_2[j]
                if 1 <= j < 3:
                    assert ary[0, j, k] == slice_3[j - 1, k]


def test_array_slice_setitem():
    shape = (intp(3), intp(4))
    i32_dtype = DType(Int32)

    data = memref.alloc(shape, i32)
    ary = Array(dtype=i32_dtype, data=data)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ary[i, j] = c
            c += i32(1)

    # Declare a larger array
    shape_2 = (intp(3), intp(3), intp(4))
    data_2 = memref.alloc(shape_2, i32)
    ary_2 = Array(dtype=i32_dtype, data=data_2)

    # Setitem slice
    ary_2[0] = ary
    ary_2[1] = i32(100)

    # Check contents
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert ary[i, j] == ary_2[0, i, j]
            assert ary_2[1, i, j] == i32(100)


def test_array_copy():
    shape = (intp(3), intp(4))
    i32_dtype = DType(Int32)

    data = memref.alloc(shape, i32)
    ary = Array(dtype=i32_dtype, data=data)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ary[i, j] = c
            c += i32(1)

    # Declare a copy
    ary_copy = ary.copy()

    # Check contents
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert ary[i, j] == ary_copy[i, j]

    # Modify the copy
    ary_copy[0, 0] = i32(10)
    assert ary[0, 0] != ary_copy[0, 0]


def test_array_fancy_getitem():
    shape = (intp(4), intp(5), intp(6))
    i32_dtype = DType(Int32)

    data = memref.alloc(shape, i32)
    ary = Array(dtype=i32_dtype, data=data)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ary[i, j, k] = c
                c += i32(1)

    # Index array
    idx_shape = (intp(2), intp(3))
    idx_data = memref.alloc(idx_shape, i32)
    idx_ary = Array(dtype=i32_dtype, data=idx_data)

    idx_ary[0, 0] = i32(0)
    idx_ary[0, 1] = i32(1)
    idx_ary[0, 2] = i32(2)
    idx_ary[1, 0] = i32(2)
    idx_ary[1, 1] = i32(1)
    idx_ary[1, 2] = i32(0)

    # Fancy getitem
    fancy_slice = ary[idx_ary]

    # Fancy getitem with slice
    fancy_slice_2 = ary[slice(1, 3), idx_ary]

    # Fancy getitem with int
    fancy_slice_3 = ary[0, idx_ary]

    # Check shapes
    assert fancy_slice.shape == (intp(2), intp(3), intp(5), intp(6))
    assert fancy_slice_2.shape == (intp(2), intp(2), intp(3), intp(6))
    assert fancy_slice_3.shape == (intp(2), intp(3), intp(6))

    # Check contents
    pass
