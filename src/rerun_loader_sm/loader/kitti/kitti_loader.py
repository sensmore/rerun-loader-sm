from io import BytesIO, StringIO
from pathlib import Path
from zipfile import Path as ZipPath
from zipfile import ZipFile

import numpy as np

from .loader import CloudLoader


def _read_kitti_label(file: Path | ZipPath) -> tuple[np.ndarray, np.ndarray] | None:
    if not file.exists():
        return None
    x = np.frombuffer(file.read_bytes(), dtype=np.uint16).reshape((-1, 2))
    return x[:, 0], x[:, 1]


def read_kitti_bin(file: Path | ZipPath) -> np.ndarray | None:
    if not file.exists():
        return None
    return np.frombuffer(file.read_bytes(), dtype=np.float32).reshape((-1, 4))


def _read_box_label(file: Path | ZipPath) -> np.ndarray | None:
    if not file.exists():
        return None
    boxtxt = file.read_text()
    if len(boxtxt) == 0:
        return np.empty((0, 16))
    vals = np.genfromtxt(StringIO(boxtxt), dtype=np.float32, delimiter=" ")
    if vals.ndim == 1:
        vals = vals.reshape(-1, 16)
    return vals


def _parse_calibration(filename: Path | ZipPath) -> dict[str, np.ndarray]:
    """Read calibration file with given filename.

    Returns:
    -------
    dict
        Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    for line in filename.read_text().splitlines():
        key, content = map(str.strip, line.split(":"))
        values = np.array(content.split(), dtype=float)
        pose = np.eye(4)
        pose[:3, :4] = values.reshape(3, 4)

        calib[key] = pose

    return calib


def _parse_poses(filename: Path | ZipPath, calibration: dict[str, np.ndarray]) -> np.ndarray:
    """Read poses file with per-scan poses from given filename.

    Returns:
    -------
    np.ndarray
        Array of poses as 4x4 numpy arrays.
    """
    cab_tr = calibration["Tr"]
    tr_inv = np.linalg.inv(cab_tr)

    values = np.loadtxt(StringIO(filename.read_text()))

    poses = values.reshape(-1, 3, 4)
    # make 4x4 matrix
    poses = np.concatenate((poses, np.tile([0, 0, 0, 1], (poses.shape[0], 1, 1))), axis=1)

    poses = np.matmul(tr_inv, np.matmul(poses, cab_tr))

    return poses.astype(np.float32)


def _load_poses(folder: Path | ZipPath) -> np.ndarray:
    path_pose = folder / "poses.txt"
    if not path_pose.exists():
        msg = f"poses.txt missing at {path_pose}"
        raise Exception(msg)

    path_calib = folder / "calib.txt"
    if not path_pose.exists():
        msg = f"calib.txt missing at {path_calib}"
        raise Exception(msg)

    calib = _parse_calibration(path_calib)
    return _parse_poses(path_pose, calib)


class KittiLoader(CloudLoader):
    def __init__(
        self,
        path: Path,
        cloud_folder: str,
        label_folder: str,
        secondary_cloud_folder: str,
        box_folder: str,
    ) -> None:
        super().__init__(
            path,
            cloud_folder,
            label_folder,
            secondary_cloud_folder,
            box_folder,
        )
        assert path.exists(), f"{path} not found!"

        self.path = Path(path)
        self._zip_path = None
        self._zip_update: dict[str, bytes] = {}
        if path.is_file():
            assert path.suffix == ".zip", "Only zip files or folder supported!"
            self._zip_path = path
            self.path = ZipPath(path)
            children = list(self.path.iterdir())
            if len(children) == 1:
                self.path = children[0]

        self.label_folder = self.path / label_folder
        self.cloud_folder = self.path / cloud_folder
        self.secondary_cloud_folder = self.path / secondary_cloud_folder
        self.box_folder = self.path / box_folder

    @staticmethod
    def can_load(path: Path | str) -> bool:
        path = Path(path)
        if path.is_file() and path.suffix == ".zip":
            return True

        return bool(path.is_dir() and (path / "poses.txt").exists())

    @property
    def poses(self) -> np.ndarray:
        return _load_poses(self.path)

    def get_cloud(self, idx: int, cloud_folder:Path | None= None) -> np.ndarray | None:
        cloud_folder = cloud_folder if cloud_folder is not None else self.cloud_folder
        return read_kitti_bin(cloud_folder / f"{idx:06}.bin")

    def get_label(self, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        return _read_kitti_label(self.label_folder / f"{idx:06}.label")

    def save_label(self, data: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
        for idx, label, instance in data:
            assert label.shape == instance.shape, f"{label.shape} != {instance.shape}"

            data_np = np.stack(
                [label, instance],
                axis=1,
                dtype=np.uint16,
            )

            buffer = data_np.tobytes()
            path = self.label_folder / f"{idx:06}.label"
            if isinstance(path, Path):
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(buffer)
            elif isinstance(path, ZipPath):
                assert self._zip_update is not None
                assert isinstance(self.label_folder, ZipPath)
                self._zip_update[f"{self.label_folder.at}{idx:06}.label"] = buffer

    def get_secondary_cloud(self, idx: int) -> np.ndarray | None:
        return self.get_cloud(self, self.secondary_cloud_folder)

    def get_box_label(self, idx: int) -> np.ndarray | None:
        return _read_box_label(self.box_folder / f"{idx:06}.txt")

    def _rebuild_zip(self) -> None:
        if len(self._zip_update) == 0:
            return
        assert self._zip_path is not None
        write_buffer = BytesIO()
        with ZipFile(self._zip_path, "r") as zip_read, ZipFile(write_buffer, "w") as zip_write:
            for item in zip_read.infolist():
                data = self._zip_update.pop(item.filename, None)
                if data is None:
                    data = zip_read.read(item.filename)

                zip_write.writestr(item, data)

        self._zip_path.write_bytes(write_buffer.getbuffer())

    def close(self) -> None:
        if self._zip_path is not None:
            self._rebuild_zip()