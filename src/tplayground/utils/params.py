import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import get_args, get_origin


def is_mutable(object_: Any) -> bool:
    return type(object_) in [list, dict, set]


class Params:
    def finalize(self):
        pass

    def finalize_recursive(self):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, Params):
                attribute.finalize_recursive()
        self.finalize()


def preprocess_class_attributes(cls: type):
    """
    this function preprocesses all attributes with annotated types
    and preprocesses them before passing class to dataclass decorator
    it allows to write simpler annotations in params declaration
    as it:
     1) allows to write construction like
        class A:
            a: int

        classs B:
            c: int
            d: A

            def __post_init__(self):
                d.a = 8

     2) allows not to write field decorator

        usually you have to write things like

        @dataclass
        classs A:
            a: Set = field(default_factory=set)
            b: Params = field(default_factory=Params)

        with this decorator you can write with the same effect
        class B:
            a: Set
            b: Params

    3) allows to write optional params in a form like
        class A:
            a: Optional[int]

    4) replaces uninitialized field without Optional annotation with NotImplemented

    """
    cls_attrributes = vars(cls)
    for attribute_name, annotation in cls.__annotations__.items():
        default = cls_attrributes.get(attribute_name, None)
        # if attribute is a Param child class wraps it with field annotation
        # for correct work with dataclass
        if type(annotation) == type and issubclass(annotation, Params):
            if default is None:
                default = annotation
            attr = field(default_factory=deepcopy(default))
        elif attribute_name not in cls_attrributes:
            # Optional[int] == Union[None, int]
            # get_args(Union[None, int]) -> (class 'NoneType', int)
            if (
                type(None) in get_args(annotation)
                or get_origin(annotation) is None
            ):
                attr = None
            else:
                attr = NotImplemented
        # main embedded mutable types are safe
        elif is_mutable(default):
            # this construction is used to avoid closure
            # see https://stackoverflow.com/questions/21053988/lambda-function-accessing-outside-variable
            attr = field(default_factory=lambda x=deepcopy(default): x)
        else:
            attr = default
        setattr(cls, attribute_name, attr)
    return cls


def params_decorator(cls: type):
    return dataclass(preprocess_class_attributes(cls))


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
