from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """
    Generic registry for models, losses, metrics, etc.

    Usage:
        MODEL_REGISTRY = Registry("models")

        @MODEL_REGISTRY.register("my_model")
        class MyModel(BaseModel):
            ...

        model = MODEL_REGISTRY.build("my_model", hidden_dim=256)
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: Optional[str] = None) -> Callable:
        def decorator(cls: Type) -> Type:
            registration_key = key or cls.__name__.lower()
            if registration_key in self._registry:
                raise KeyError(
                    f"'{registration_key}' already registered in {self.name}"
                )
            self._registry[registration_key] = cls
            return cls
        return decorator

    def build(self, key: str, **kwargs: Any) -> Any:
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{key}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[key](**kwargs)

    def get(self, key: str) -> Type:
        return self._registry[key]

    def list(self) -> list:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, keys={self.list()})"


# Global registries
MODEL_REGISTRY = Registry("models")
LOSS_REGISTRY = Registry("losses")
METRIC_REGISTRY = Registry("metrics")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")
DATAMODULE_REGISTRY = Registry("datamodules")
