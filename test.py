# pytest me
import pytest
import logging
from mcl.machine_types import i32, i64, intp, pointer
from mcl.internal import Type
from mcl.ndarray import Array, DType, Int32, MemRef



def test_i32():
    a = i32(321)

    assert not isinstance(a, Type)
    assert not issubclass(type(a), Type) # metaclass not subclass
    assert type(a).__mcl_type_descriptor__.machine_repr == "i32"

    b = i32(123)
    c = a + b

    out = c == i32(444)
    assert isinstance(out, bool)
    assert out


def test_i64():
    a = i32(123)
    b = i64(321)

    c = b + a.cast(i64)
    assert c == i64(444)


def test_final():
    with pytest.raises(TypeError, match="final type cannot be subclassed"):
        class sub_i32(i32):
            pass


def test_array():

    shape = (intp(3), intp(4))
    itemsize = intp(4)
    strides = (shape[-1] * itemsize, itemsize)

    i32_dtype = DType(Int32)


    print(i32_dtype)

    ptr = pointer.new(intp(10))

    data = MemRef(shape=shape, strides=strides, baseptr=ptr, dataptr=ptr)
    print(data)

    ary = Array(dtype=i32_dtype, data=data)

    print(ary)
    print(ary.shape)
    print(ary.dtype)

    ary[intp(0), intp(0)] = i32(0xcafe)
    res = ary[intp(0), intp(0)]
    assert res == i32(0xcafe)

    # write loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ary[intp(i), intp(j)] = c
            c += i32(1)

    # read loop
    c = i32(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            got = ary[intp(i), intp(j)]
            print(f"ary[{i}, {j}] = {got}")
            assert got == c
            c += i32(1)