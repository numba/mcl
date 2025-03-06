"""
The Virtual Machine

This file is not meant to be compiled.
Only for providing an implementation that Python interpreter can evaluate.
"""

from __future__ import annotations

import inspect
import logging
import operator
import struct
import typing as _tp
from dataclasses import dataclass
from functools import reduce

from mcl import machine_types as _mt

import numpy as np

@dataclass(frozen=True)
class TypeDescriptor:
    machine_repr: str
    final: bool
    builtin: bool


class Type(type):
    registry: set[Type] = set()

    __mcl_type_descriptor__: TypeDescriptor | None = None

    def __new__(cls, name, bases, ns, *, td: TypeDescriptor | None = None):

        if td is not None:
            if td.final:
                ns["__init_subclass__"] = _final_init_subclass
        typ = super().__new__(cls, name, bases, ns)
        if td is not None:
            typ.__mcl_type_descriptor__ = td

        cls.registry.add(typ)
        return typ

    def __call__(cls, *args, **kwargs):
        ty = type.__call__(cls, *args, **kwargs)
        return ty


class BaseMachineType(Type):
    def __call__(cls, value):
        obj = object.__new__(cls)
        if type(value) in cls.registry:
            obj.__value = _get_machine_value(value)
        else:
            obj.__value = value
        return obj

    @classmethod
    def get_machine_value(cls, obj):
        return obj.__value


def _make_machine_type_methods(ns: dict) -> dict:
    def m__repr__(self):
        v = _get_machine_value(self)
        return f"{type(self).__name__}({v})"

    ns.setdefault("__repr__", m__repr__)
    return ns


_get_machine_value = BaseMachineType.get_machine_value


def _final_init_subclass(self):
    raise TypeError("final type cannot be subclassed")


def machine_type(*, final=False, builtin=False):
    def wrap(cls):
        ns = dict(**cls.__dict__)
        machine_repr = ns.pop("__machine_repr__")
        td = TypeDescriptor(
            machine_repr=machine_repr, final=final, builtin=builtin
        )
        return BaseMachineType(
            cls.__name__, (), _make_machine_type_methods(ns), td=td
        )

    return wrap


_machine_op_table = {}


def machine_op[T](opname: str, restype: _tp.Type[T], *args) -> T:
    """
    Note: bool is implicit in the system. It is too foundational in Python to
          have an override.
    """
    return _machine_op_table[opname](opname, restype, *args)


def _reg_op(fn):
    name = fn.__name__.lstrip("_")
    _machine_op_table[name] = fn
    return fn


def _binop[T](op, restype: _tp.Type[T], *args) -> T:
    (lhs, rhs) = args
    assert type(lhs) is type(rhs)
    assert type(lhs) is restype
    mv1 = _get_machine_value(lhs)
    mv2 = _get_machine_value(rhs)
    return restype(op(mv1, mv2))


def _cmpop[T](op, restype: _tp.Type[T], *args) -> T:
    (lhs, rhs) = args
    assert type(lhs) is type(rhs)
    mv1 = _get_machine_value(lhs)
    mv2 = _get_machine_value(rhs)
    return restype(op(mv1, mv2))


@_reg_op
def _int_add[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.add, restype, *args)


@_reg_op
def _int_sub[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.sub, restype, *args)


@_reg_op
def _int_mul[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.mul, restype, *args)


@_reg_op
def _int_floordiv[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.floordiv, restype, *args)


@_reg_op
def _int_eq[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.eq, restype, *args)


@_reg_op
def _int_lt[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.lt, restype, *args)

@_reg_op
def _int_mod[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.mod, restype, *args)

@_reg_op
def _float_add[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.add, restype, *args)


@_reg_op
def _float_sub[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.sub, restype, *args)

@_reg_op
def _float_mul[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.mul, restype, *args)


@_reg_op
def _float_floordiv[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.floordiv, restype, *args)

@_reg_op
def _float_truediv[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _binop(operator.truediv, restype, *args)


@_reg_op
def _float_eq[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.eq, restype, *args)


@_reg_op
def _float_lt[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.lt, restype, *args)

@_reg_op
def _float_mod[T](opname: str, restype: _tp.Type[T], *args) -> T:
    return _cmpop(operator.mod, restype, *args)

@_reg_op
def _memref_alloc[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [shape, typ] = args
    assert restype is _mt.memref
    assert type(shape) is tuple
    mv_shape = tuple(map(_get_machine_value, shape))
    memref = _the_memsys.alloc(mv_shape, typ)
    return restype(memref)


@_reg_op
def _memref_alloc_random[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [shape, typ] = args
    assert restype is _mt.memref
    assert type(shape) is tuple

    mv_shape = tuple(map(_get_machine_value, shape))
    memref = _the_memsys.alloc(mv_shape, typ)
    numpy_random = np.random.random(mv_shape).astype(_np_type(memref.datatype))

    _the_memsys.memcpy(memref, numpy_random.tobytes())
    return restype(memref)

@_reg_op
def _memref_shape[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    assert restype is tuple
    memref: MemRef = _get_machine_value(obj)
    shape = tuple(map(_mt.intp, memref.shape))
    return shape


@_reg_op
def _memref_strides[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    assert restype is tuple
    memref: MemRef = _get_machine_value(obj)
    strides = tuple(map(_mt.intp, memref.strides))
    return strides


@_reg_op
def _memref_offset[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    assert restype is tuple
    memref: MemRef = _get_machine_value(obj)
    offset = _mt.intp(memref.offset)
    return offset


@_reg_op
def _memref_store[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, indices, val] = args
    assert type(indices) is tuple
    memref: MemRef = _get_machine_value(obj)
    indices = tuple(map(_get_machine_value, indices))
    _the_memsys.write(memref, indices, val)


@_reg_op
def _memref_load[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, indices] = args
    assert type(indices) is tuple
    memref: MemRef = _get_machine_value(obj)
    indices = tuple(map(_get_machine_value, indices))
    return restype(_the_memsys.read(memref, indices))


@_reg_op
def _memref_view[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, new_shape, new_strides, offset] = args
    assert type(new_shape) is tuple
    memref: MemRef = _get_machine_value(obj)
    new_shape = tuple(map(_get_machine_value, new_shape))
    new_strides = tuple(map(_get_machine_value, new_strides))
    offset = _get_machine_value(offset)

    new_memref = _the_memsys.view(
        memref,
        shape=new_shape,
        strides=new_strides,
        datatype=memref.datatype,
        itemsize=memref.itemsize,
        size=reduce(operator.mul, new_shape) * memref.itemsize,
        offset=offset
    )
    return restype(new_memref)


@_reg_op
def _memref_copy[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    memref: MemRef = _get_machine_value(obj)
    new_memref = _the_memsys.copy(memref)
    return restype(new_memref)

@_reg_op
def _memref_exp[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    memref: MemRef = _get_machine_value(obj)
    new_memref = _the_memsys.apply_np_op(memref, np.exp)
    return restype(new_memref)

@_reg_op
def _memref_sqrt[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    memref: MemRef = _get_machine_value(obj)
    new_memref = _the_memsys.apply_np_op(memref, np.exp)
    return restype(new_memref)

@_reg_op
def _memref_sum[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, axis, keepdims] = args
    memref: MemRef = _get_machine_value(obj)
    new_memref = _the_memsys.apply_np_op(memref, lambda x: np.sum(x, axis=_get_machine_value(axis), keepdims=_get_machine_value(keepdims)))
    return restype(new_memref)

@_reg_op
def _memref_max[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, axis, keepdims] = args
    memref: MemRef = _get_machine_value(obj)
    new_memref = _the_memsys.apply_np_op(memref, lambda x: np.max(x, axis=_get_machine_value(axis), keepdims=_get_machine_value(keepdims)))
    return restype(new_memref)

@_reg_op
def _memref_maximum[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, other] = args
    memref: MemRef = _get_machine_value(obj)
    other_obj: MemRef = _get_machine_value(other)
    new_memref = _the_memsys.apply_np_bin_op(memref, other_obj, np.maximum)
    return restype(new_memref)

@_reg_op
def _memref_add[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, other] = args
    memref: MemRef = _get_machine_value(obj)
    other_obj: MemRef = _get_machine_value(other)
    new_memref = _the_memsys.apply_np_bin_op(memref, other_obj, operator.add)
    return restype(new_memref)

@_reg_op
def _memref_truediv[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, other] = args
    memref: MemRef = _get_machine_value(obj)
    other_obj: MemRef = _get_machine_value(other)
    new_memref = _the_memsys.apply_np_bin_op(memref, other_obj, operator.truediv)
    return restype(new_memref)

@_reg_op
def _memref_sub[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj, other] = args
    memref: MemRef = _get_machine_value(obj)
    other_obj: MemRef = _get_machine_value(other)
    new_memref = _the_memsys.apply_np_bin_op(memref, other_obj, operator.sub)
    return restype(new_memref)

@_reg_op
def _memref_matmul[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [mat_1, mat_2] = args
    memref_mat_1: MemRef = _get_machine_value(mat_1)
    memref_mat_2: MemRef = _get_machine_value(mat_2)

    new_memref = _the_memsys.apply_np_bin_op(memref_mat_1, memref_mat_2, np.matmul)
    return restype(new_memref)

@_reg_op
def _memref_print[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [obj] = args
    memref: MemRef = _get_machine_value(obj)
    memref_copy = _the_memsys.copy(memref)
    buffer = _the_memsys._memmap[memref_copy]
    print(np.frombuffer(buffer, dtype=_np_type(memref.datatype)).reshape(memref.shape))

@_reg_op
def _tuple_cast[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [resty, tup] = args
    assert issubclass(type(resty), BaseMachineType)
    return tuple(map(resty, tup))


@_reg_op
def _cast[T](opname: str, restype: _tp.Type[T], *args) -> T:
    [v0] = args
    return restype(_get_machine_value(v0))


def _from_bytes[T](restype: _tp.Type[T], raw: bytes) -> T:
    match restype:
        case _mt.i32:
            return restype(int.from_bytes(raw, signed=True))
        case _mt.f32:
            return restype(struct.unpack("f", raw)[0])
        case _:
            raise TypeError(restype)
    raise AssertionError


def _to_bytes[T](value: T) -> bytes:
    mv = _get_machine_value(value)
    out: bytes
    match type(value):
        case _mt.i32:
            out = mv.to_bytes(4, signed=True)
        case _mt.f32:
            out = struct.pack("f", mv)
        case _:
            raise TypeError(f"invalid type {type(value)}")

    return out


def _sizeof(restype: _tp.Type) -> int:
    match restype:
        case _mt.i32:
            out = 4
        case _mt.f32:
            out = 4
        case _:
            raise TypeError(f"invalid type {restype}")

    return out

def _np_type(restype: _tp.Type) -> _tp.Type:
    match restype:
        case _mt.i32:
            out = np.int32
        case _mt.f32:
            out = np.float32
        case _:
            raise TypeError(f"invalid type {restype}")

    return out


class BaseStructType(Type):
    def __call__(cls, *args, **kwargs):
        hints = _tp.get_type_hints(cls)
        params = []
        for name, annotation in hints.items():
            params.append(
                inspect.Parameter(
                    name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                )
            )
        bound = inspect.Signature(params).bind(*args, **kwargs)
        fields = {k: v for k, v in bound.arguments.items()}

        obj = object.__new__(cls)
        # TODO add type check
        obj.__mcl_struct_fields__ = fields
        return obj


def _make_struct_methods(ns: dict):
    def m__getattr__(self, k):
        fields = self.__mcl_struct_fields__
        if k not in fields:
            raise AttributeError(k)
        return fields[k]

    if "__getattr__" in ns:
        raise TypeError("struct_type must not define __getattr__")
    ns["__getattr__"] = m__getattr__

    def m__repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__mcl_struct_fields__.items()]
        param_text = ", ".join(params)
        return f"{type(self).__name__}({param_text})"

    ns.setdefault("__repr__", m__repr__)

    return ns


def struct_type(*, final=False, builtin=False):
    def wrap(cls):
        ns = dict(**cls.__dict__)
        machine_repr = "struct"
        td = TypeDescriptor(
            machine_repr=machine_repr, final=final, builtin=builtin
        )
        return BaseStructType(
            cls.__name__, cls.__bases__, _make_struct_methods(ns), td=td
        )

    return wrap


@dataclass(frozen=True)
class MemRef:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    datatype: _tp.Type
    itemsize: int
    size: int
    owner: MemRef | None = None
    offset: int = 0

    def __repr__(self) -> str:
        buf = [f"<{hex(id(self))} shape={self.shape} strides={self.strides}"]
        if self.owner:
            buf.append(f"owner={self.owner}")
        buf.append(">")
        return " ".join(buf)

    def __eq__(self, value: object) -> bool:
        if id(self) == id(value):
            return True        

    def handle(self) -> MemRef:
        if self.owner:
            return self.owner.handle()
        return self


class MemorySystem:
    """The Memory System

    To provide safe memory operation, all memory manipulation must go through
    this class. No pointer arithmetic.
    """

    _memmap: dict[MemRef, bytearray]
    _viewmap: dict[MemRef, list[MemRef]]
    _last_addr: int

    def __init__(self):
        self._memmap = {}
        self._viewmap = {}

    def alloc(self, shape: tuple[int, ...], datatype: _tp.Type) -> MemRef:
        itemsize = _sizeof(datatype)
        nbytes = reduce(operator.mul, shape) * itemsize
        assert nbytes != 0
        # compute strides
        strides = []
        last = itemsize
        for s in reversed(shape):
            strides.append(last)
            last *= s
        strides.reverse()
        assert last == nbytes
        memref = MemRef(
            shape=shape,
            strides=tuple(strides),
            datatype=datatype,
            itemsize=itemsize,
            size=nbytes,
        )
        buffer = bytearray(nbytes)
        self._memmap[memref] = buffer
        return memref

    def write[
        T
    ](self, memref: MemRef, indices: tuple[int, ...], value: T) -> None:
        logging.debug("write %s indices=%s value=%s", memref, indices, value)
        buffer = self._memmap[memref.handle()]
        offset = sum(
            i * s for i, s in zip(indices, memref.strides, strict=True)
        )
        offset += memref.offset
        value_bytes = _to_bytes(value)
        buffer[offset : offset + len(value_bytes)] = value_bytes

    def memcpy[
        T
    ](self, memref: MemRef, value_bytes: bytearray) -> None:
        buffer = self._memmap[memref.handle()]
        offset = memref.offset
        buffer[offset : offset + len(value_bytes)] = value_bytes

    def read(self, memref: MemRef, indices: tuple[int, ...]) -> bytes:
        logging.debug("read %s indices=%s", memref, indices)
        buffer = self._memmap[memref.handle()]
        offset = sum(
            i * s for i, s in zip(indices, memref.strides, strict=True)
        )
        offset += memref.offset
        n = memref.itemsize
        raw_bytes = buffer[offset : offset + n]
        return _from_bytes(memref.datatype, raw_bytes)

    def view(
        self, 
        memref: MemRef,
        shape,
        strides,
        datatype,
        itemsize,
        size,
        offset
    ) -> MemRef:
        new_memref = MemRef(
            shape=shape,
            strides=strides,
            datatype=datatype,
            itemsize=itemsize,
            size=size,
            owner=memref.handle(),
            offset=offset
        )
        self._viewmap.setdefault(memref, []).append(new_memref)
        return new_memref

    def copy(
        self, 
        memref: MemRef
    ) -> MemRef:
        memref_owner = memref.handle()
        if memref_owner != memref:
            strides = []
            last = memref_owner.itemsize
            for s in reversed(memref.shape):
                strides.append(last)
                last *= s
            strides.reverse()
            assert last == memref.size
            new_memref = MemRef(
                shape=memref.shape,
                strides=tuple(strides),
                datatype=memref.datatype,
                itemsize=memref.itemsize,
                size=memref.size,
                owner=None,
                offset=0
            )
            self._memmap[new_memref] = bytearray(memref.size)
            for idx in np.ndindex(memref.shape):
                self.write(new_memref, idx, self.read(memref, idx))
        else:
            new_memref = MemRef(
                shape=memref.shape,
                strides=memref.strides,
                datatype=memref.datatype,
                itemsize=memref.itemsize,
                size=memref.size,
                owner=None,
                offset=memref.offset
            )
            self._memmap[new_memref] = self._memmap[memref_owner].copy()
        return new_memref

    def apply_np_op(self, memref: MemRef, op):
        buffer = self._memmap[self.copy(memref)]
        buffer = np.frombuffer(buffer, dtype=_np_type(memref.datatype)).reshape(memref.shape)
        res_array = op(buffer)
        new_memref = self.alloc(res_array.shape, memref.datatype)
        self.memcpy(new_memref, res_array.tobytes())
        return new_memref

    def apply_np_bin_op(self, memref_1: MemRef, memref_2: MemRef, op):
        buffer_1 = self._memmap[self.copy(memref_1)]
        buffer_2 = self._memmap[self.copy(memref_2)]
        buffer_1 = np.frombuffer(buffer_1, dtype=_np_type(memref_1.datatype)).reshape(memref_1.shape)
        buffer_2 = np.frombuffer(buffer_2, dtype=_np_type(memref_2.datatype)).reshape(memref_2.shape)
        res_array = op(buffer_1, buffer_2)
        new_memref = self.alloc(res_array.shape, memref_1.datatype)
        self.memcpy(new_memref, res_array.tobytes())
        return new_memref

_the_memsys = MemorySystem()
