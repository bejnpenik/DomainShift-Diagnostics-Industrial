from dataclasses import dataclass, field
from collections.abc import Mapping


@dataclass(frozen=True)
class Metadata(Mapping):
    _data: dict = field(default_factory=dict)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No such attribute: {name}")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return f"Metadata({self._data})"