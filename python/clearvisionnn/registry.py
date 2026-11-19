"""Named, swappable components.

Every family of interchangeable parts (degradation modules, networks, losses,
metrics, baselines) has one `Registry`. A component registers under the name
that configuration files use, so adding one never means editing a dispatcher.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps configuration names to components of one family."""

    def __init__(self, family: str) -> None:
        self._family = family
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Class or function decorator that registers its target as `name`."""

        def decorator(item: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._family} '{name}' is registered twice")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        """The component registered as `name`."""
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"unknown {self._family} '{name}' (known: {known})") from None

    def names(self) -> list[str]:
        """Every registered name, sorted."""
        return sorted(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
