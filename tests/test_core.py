# pytest me
import pytest
from mcl.machine_types import i32, i64
from mcl.vm import Type

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

    c = b + i64(a)
    assert c == i64(444)


def test_final():
    with pytest.raises(TypeError, match="final type cannot be subclassed"):
        class sub_i32(i32):
            pass

