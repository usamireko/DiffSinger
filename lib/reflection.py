import importlib
import inspect
from typing import Any

import torch.optim

from lib.config.schema import LRSchedulerConfig


def filter_kwargs_by_class(cls: type, kwargs: dict[str, Any]):
    """
    Filter the kwargs dictionary to only include keys that are valid arguments for the given class.
    """
    parameters = inspect.signature(cls).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        # If the class accepts **kwargs, there's no need to filter
        return kwargs.copy()
    valid_keys = [
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    filtered_kwargs = {name: kwargs[name] for name in valid_keys if name in kwargs}
    return filtered_kwargs


def build_object_from_class_name(cls_name: str, required_parent_cls=None, /, *args, **kwargs):
    """
    Build an object from a class name. The class name should be in the format 'module.ClassName'.
    If `required_parent_cls` is specified, the class should be a subclass of required_parent_cls.
    """
    module_name, class_name = cls_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if required_parent_cls and not issubclass(cls, required_parent_cls):
        raise TypeError(f"Class {cls_name} is not a subclass of {required_parent_cls.__name__}.")
    return cls(*args, **filter_kwargs_by_class(cls, kwargs))


def build_lr_scheduler_from_config(optimizer: torch.optim.Optimizer, config: LRSchedulerConfig):
    """
    Build a learning rate scheduler from a configuration object, recursively.
    """
    kwargs = {}
    for key, value in config.kwargs.items():
        if isinstance(value, LRSchedulerConfig):
            value = build_lr_scheduler_from_config(optimizer, value)
        elif isinstance(value, list):
            value = [
                build_lr_scheduler_from_config(optimizer, item) if isinstance(item, LRSchedulerConfig) else item
                for item in value
            ]
        elif isinstance(value, dict):
            value = {
                k: build_lr_scheduler_from_config(optimizer, v) if isinstance(v, LRSchedulerConfig) else v
                for k, v in value.items()
            }
        kwargs[key] = value
    scheduler = build_object_from_class_name(
        config.cls,
        torch.optim.lr_scheduler.LRScheduler,
        optimizer=optimizer,
        **kwargs
    )
    return scheduler
