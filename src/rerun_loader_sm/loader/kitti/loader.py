from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class CloudLoaderBase(ABC):
    @property
    @abstractmethod
    def path_original(self) -> Path: ...

    @property
    @abstractmethod
    def poses(self) -> np.ndarray: ...

    @abstractmethod
    def get_cloud(self, idx: int) -> np.ndarray | None: ...

    @abstractmethod
    def get_label(self, idx: int) -> tuple[np.ndarray, np.ndarray] | None: ...

    @abstractmethod
    def save_label(self, data: list[tuple[int, np.ndarray, np.ndarray]]) -> None: ...

    @abstractmethod
    def get_secondary_cloud(self, idx: int) -> np.ndarray | None: ...

    @abstractmethod
    def get_box_label(self, idx: int) -> np.ndarray | None: ...

    def optimize(self) -> None:  # noqa: B027, not all loaders need an optimize
        ...

    def close(self) -> None:  # noqa: B027, not all loaders need a close
        ...


class CloudLoader(CloudLoaderBase):
    def __init__(
        self,
        path: Path | str,
        cloud_folder: str,
        label_folder: str,
        secondary_cloud_folder: str,
        box_folder: str,
    ) -> None:
        self._path = Path(path)
        self._cloud_folder = cloud_folder
        self._label_folder = label_folder
        self._secondary_cloud_folder = secondary_cloud_folder
        self._box_folder = box_folder

    @property
    def path_original(self) -> Path:
        return self._path

    @staticmethod
    @abstractmethod
    def can_load(path: Path | str) -> bool: ...