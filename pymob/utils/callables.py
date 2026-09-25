"""Serialize callables, classes and instances as JSON-compatible references.

A callable (function or class) is saved as a string: its registry key if it is
registered, otherwise its import path ``"pkg.module:Qual.name"``. An instance is
saved as ``{"__ref__": <class ref>, "init": {...}}`` and rebuilt by calling the
class with the stored constructor arguments. This works for pydantic models (the
fields are the arguments) and for classes that inherit from
:class:`Reinitializable` (the ``__init__`` arguments are recorded).

Use :data:`ObjRef` as a pydantic field type to get this behaviour for free.
"""

import functools
import importlib
import inspect
from collections.abc import Callable
from typing import Annotated, Any, Generic, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    model_validator,
)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

_TAG = "__ref__"
_REGISTRY: dict[str, Callable[..., Any]] = {}
_REVERSE: dict[Callable[..., Any], str] = {}


def register(name: str | None = None) -> Callable[[F], F]:
    """Register a function or class under a stable key (default: import path)."""

    def deco(obj: F) -> F:
        key = name or _import_path(obj)
        if _REGISTRY.get(key, obj) is not obj:
            raise KeyError(f"Registry key {key!r} already taken by {_REGISTRY[key]!r}")
        _REGISTRY[key] = obj
        _REVERSE[obj] = key
        return obj

    return deco


def _import_path(obj: Any) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


def to_ref(obj: Callable[..., Any]) -> str:
    """Function or class -> string reference. Raises if it can't be resolved back."""
    try:
        return _REVERSE[obj]
    except (KeyError, TypeError):  # not registered / unhashable
        pass
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", "")
    if not qualname or module in (None, "__main__") or "<" in qualname:
        raise ValueError(
            f"{obj!r} can't be referenced by import path (lambda, closure, instance "
            "or defined in __main__). Move it into a module or @register it."
        )
    ref = _import_path(obj)
    if from_ref(ref) != obj:  # '==' not 'is': bound classmethods are recreated on access
        raise ValueError(f"{ref!r} does not resolve back to {obj!r}")
    return ref


def from_ref(ref: str) -> Callable[..., Any]:
    """String reference -> function or class. Registry first, then import path."""
    if ref in _REGISTRY:
        return _REGISTRY[ref]
    module, sep, qualname = ref.partition(":")
    if not sep:
        raise KeyError(f"Unknown registry key {ref!r}. Known: {sorted(_REGISTRY)}")
    obj: Any = importlib.import_module(module)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class Reinitializable:
    """Mixin that records the ``__init__`` arguments so an instance can be rebuilt."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            return
        orig = cls.__init__
        sig = inspect.signature(orig)

        @functools.wraps(orig)
        def __init__(self: Any, *args: Any, **kw: Any) -> None:
            if "_init_kwargs" not in self.__dict__:  # only the outermost call records
                bound = sig.bind(self, *args, **kw)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                arguments.pop(next(iter(sig.parameters)))  # drop `self`
                for name, p in sig.parameters.items():
                    if p.kind is p.VAR_KEYWORD:
                        arguments.update(arguments.pop(name, {}))
                    elif p.kind is p.VAR_POSITIONAL and arguments.pop(name, ()):
                        raise TypeError(f"{cls.__name__}: *args can't be saved; use keywords")
                self.__dict__["_init_kwargs"] = arguments
            orig(self, *args, **kw)

        cls.__init__ = __init__  # type: ignore[misc]


def dump(obj: Any) -> Any:
    """Serialize a value to JSON-compatible data.

    Function/class -> ``str``; instance -> ``{"__ref__": ..., "init": {...}}``;
    lists, tuples and dicts are handled recursively; JSON scalars pass through.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_dump_nested(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dump_nested(v) for k, v in obj.items()}
    if isinstance(obj, type) or inspect.isroutine(obj):
        return to_ref(obj)
    if isinstance(obj, BaseModel):
        init = obj.model_dump(mode="json")
    elif (kw := obj.__dict__.get("_init_kwargs") if hasattr(obj, "__dict__") else None) is not None:
        init = {k: _dump_nested(v) for k, v in kw.items()}
    else:
        raise ValueError(
            f"Can't save instance {obj!r}: make its class a pydantic model "
            "or inherit from Reinitializable."
        )
    return {_TAG: to_ref(type(obj)), "init": init}


def _dump_nested(v: Any) -> Any:
    d = dump(v)
    # Tag nested function/class refs so they aren't read back as plain strings.
    if isinstance(d, str) and not isinstance(v, str):
        return {_TAG: d}
    return d


def load(value: Any) -> Any:
    """Inverse of :func:`dump` for top-level values (a bare string is a reference)."""
    if isinstance(value, str):
        return from_ref(value)
    return _load_nested(value)


def _load_nested(v: Any) -> Any:
    if isinstance(v, list):
        return [_load_nested(x) for x in v]
    if not isinstance(v, dict):
        return v
    if _TAG not in v:
        return {k: _load_nested(x) for k, x in v.items()}
    target = from_ref(v[_TAG])
    if "init" not in v:  # nested function or class reference
        return target
    if isinstance(target, type) and issubclass(target, BaseModel):
        return target.model_validate(v["init"])
    return target(**{k: _load_nested(x) for k, x in v["init"].items()})


def _validate(value: Any) -> Any:
    if isinstance(value, str) or (isinstance(value, dict) and _TAG in value):
        return load(value)
    return value


ObjRef = Annotated[
    Any,
    BeforeValidator(_validate),
    PlainSerializer(dump, when_used="json"),
    WithJsonSchema({}),
]
"""Pydantic field type: accepts a function, class or instance, or its serialized form."""


class Module(BaseModel, Generic[T]):
    """An instance stored as its class (``obj``) plus serialized ``init_kwargs``."""

    obj: ObjRef
    init_kwargs: dict[str, Any] = Field(default={})

    @model_validator(mode="after")
    def set_init_kwargs(self):
        # Split an instance into class + init kwargs. A class (loaded from a file
        # or re-validated) is kept as is, together with its init_kwargs.
        if not (isinstance(self.obj, type) or inspect.isroutine(self.obj)):
            self.init_kwargs = dump(self.obj)["init"]
            self.obj = type(self.obj)
        return self

    @property
    def initialized(self) -> T:
        return self.obj(**_load_nested(self.init_kwargs))
        
    model_config = ConfigDict(
        # validate assignment leads to infitite recursion
        validate_assignment=False, 
        extra="forbid", 
        protected_namespaces=()
    )