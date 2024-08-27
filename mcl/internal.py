"""
This file is not meant to be compiled.
Only for providing an implementation that Python interpreter can evaluate.
"""

from __future__ import annotations
import typing as _tp
from dataclasses import dataclass
import inspect


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
        case "int_add":
            (lhs, rhs) = args
            assert type(lhs) is type(rhs)
            assert type(lhs) is restype
            mv1 = _get_machine_value(lhs)
            mv2 = _get_machine_value(rhs)
            return restype(mv1 + mv2)
        case "int_eq":
            (lhs, rhs) = args
            assert type(lhs) is type(rhs)
            mv1 = _get_machine_value(lhs)
            mv2 = _get_machine_value(rhs)
            return restype(mv1 == mv2)
        case "cast":
            [v0] = args
            return restype(_get_machine_value(v0))
        case _:
            raise NotImplementedError(opname, args)


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
