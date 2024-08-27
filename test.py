# pytest me
import pytest
from mcl.machine_types import i32, i64, intp
from mcl.internal import Type
from mcl.ndarray import Array, IntDType

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
    i32_dtype = IntDType(name="int32", bitwidth=32)

    print(i32_dtype)

    ary = Array(shape=shape, dtype=i32_dtype)

    print(ary)
    print(ary.shape)
    print(ary.dtype)

