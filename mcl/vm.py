"""
This file is not meant to be compiled.
Only for providing an implementation that Python interpreter can evaluate.
"""

from __future__ import annotations
import typing as _tp
from dataclasses import dataclass
import inspect
import logging

from mcl import machine_types as _mt


T = _tp.TypeVar("T")


@dataclass(frozen=True)
class TypeDescriptor:
    machine_repr: str
    final: bool
    builtin: bool


class Type(type):
    registry: list[Type] = []

    __mcl_type_descriptor__: TypeDescriptor | None = None

    def __new__(cls, name, bases, ns, *, td: TypeDescriptor | None = None):

        if td is not None:
            if td.final:
                ns["__init_subclass__"] = _final_init_subclass
        typ = super().__new__(cls, name, bases, ns)
        if td is not None:
            typ.__mcl_type_descriptor__ = td
        cls.registry.append(typ)
        return typ

    def __call__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)


class BaseMachineType(Type):
    def __call__(cls, value):
        obj = object.__new__(cls)
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


@_tp.no_type_check
def machine_op(opname: str, restype: _tp.Type[T], *args) -> T:
    """
    Note: bool is implicit in the system. It is too foundational in Python to
          have an override.
    """
    from mcl.machine_types import i64, i32

    match opname:
        # binop
        case "int_add":
            (lhs, rhs) = args
            assert type(lhs) is type(rhs)
            assert type(lhs) is restype
            mv1 = _get_machine_value(lhs)
            mv2 = _get_machine_value(rhs)
            return restype(mv1 + mv2)
        case "int_mul":
            (lhs, rhs) = args
            assert type(lhs) is type(rhs)
            assert type(lhs) is restype
            mv1 = _get_machine_value(lhs)
            mv2 = _get_machine_value(rhs)
            return restype(mv1 * mv2)
        # cmpop
        case "int_eq":
            (lhs, rhs) = args
            assert type(lhs) is type(rhs)
            mv1 = _get_machine_value(lhs)
            mv2 = _get_machine_value(rhs)
            return restype(mv1 == mv2)
        # pointer/memory
        case "malloc":
            [nbytes] = args
            assert restype is _mt.pointer
            assert type(nbytes) is _mt.intp
            mv_nbytes = _get_machine_value(nbytes)
            ptr = _the_memsys.malloc(mv_nbytes)
            return restype(ptr)
        case "memref_setitem":
            [memref, indices, val] = args
            assert type(indices) is tuple
            strides = memref.strides
            offset = sum(
                _get_machine_value(i) * _get_machine_value(s)
                for i, s in zip(indices, strides, strict=True)
            )
            mv_ptr: _Ptr = _get_machine_value(memref.dataptr)
            _the_memsys.write(mv_ptr, offset, _to_bytes(val))
        case "memref_getitem":
            [memref, indices] = args
            assert type(indices) is tuple
            strides = memref.strides
            offset = sum(
                _get_machine_value(i) * _get_machine_value(s)
                for i, s in zip(indices, strides, strict=True)
            )
            size = _sizeof(restype)
            mv_ptr: _Ptr = _get_machine_value(memref.dataptr)
            raw: bytes = _the_memsys.read(mv_ptr, offset, size)

            match restype:
                case _mt.i32:
                    return restype(int.from_bytes(raw))
                case _:
                    raise TypeError(f"invalid type {restype}")
        # misc

        case "cast":
            [v0] = args
            return restype(_get_machine_value(v0))

        case _:
            raise NotImplementedError(opname, args)


def _to_bytes[T](value: T) -> bytes:
    mv = _get_machine_value(value)
    out: bytes
    match type(value):
        case _mt.i32:
            out = mv.to_bytes(4, signed=True)
        case _:
            raise TypeError(f"invalid type {type(value)}")

    return out


def _sizeof(restype: _tp.Type) -> int:
    match restype:
        case _mt.i32:
            out = 4
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


@dataclass(frozen=True, eq=True)
class _Ptr:
    addr: int
    end_addr: int

    def __repr__(self) -> str:
        return f"<_Ptr 0x{self.addr:08x}:0x{self.end_addr:08x}>"


class MemorySystem:
    _memmap: dict[_Ptr, bytearray]
    _last_addr: int
    _null: _Ptr

    def __init__(self):
        self._memmap = {}
        self._last_addr = 0

        reserve = 0x8000_0000
        self._null = self._fresh_pointer(reserve)
        assert self._null.addr == 0
        assert self._last_addr == reserve

    def _fresh_pointer(self, size: int) -> _Ptr:
        base = self._last_addr
        self._last_addr += size
        return _Ptr(addr=base, end_addr=self._last_addr)

    def malloc(self, size: int) -> _Ptr:
        buf = bytearray(size)
        ptr = self._fresh_pointer(size)
        self._memmap[ptr] = buf
        return ptr

    def write(self, ptr: _Ptr, offset: int, value: bytes) -> None:
        logging.debug("write %s offset=0%s value=%s", ptr, offset, value)
        ba = self._memmap[ptr]
        n = len(value)
        ba[offset : offset + n] = value

    def read(self, ptr: _Ptr, offset: int, size: int) -> bytes:
        logging.debug("read %s offset=%s size=%s", ptr, offset, size)
        ba = self._memmap[ptr]
        return ba[offset : offset + size]


_the_memsys = MemorySystem()
