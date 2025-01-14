import rerun as rr
import rerun.blueprint as rrb

import numpy as np
import argparse
import h5py
from pathlib import Path

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Type  # keep it backward compatible!
import sys
import logging
import yaml


from rerun_loader_sm.loader.kitti.kitti_loader import read_kitti_bin, KittiLoader

# Define the log file path
log_file = "/tmp/rerun-loader-sm-log.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # Log to stderr
        logging.FileHandler(log_file)      # Log to the specified file
    ]
)

@dataclass
class KittiLoaderConfig:
    cloud_start: float = 0
    cloud_end: float = float("inf")

# TODO: better parsing, e.g with pydantic?
def load_config_from_file(config_path: Path, klass: Type[KittiLoaderConfig] = KittiLoaderConfig) -> KittiLoaderConfig:
    with config_path.open('r') as file:
        config_data = yaml.safe_load(file)
    return klass(**config_data)

def write_config_to_file(config_path: Path, config_instance) -> None:
    with config_path.open('w') as file:
        yaml.safe_dump(asdict(config_instance), file)

# TODO: organize the config better, so we can load useful parts to and from file
@dataclass
class LogConfig:
    filepath: str
    application_id: str = "rerun_loader_sm"
    recording_id: Optional[str] = None
    entity_path_prefix: str = ""
    timeless: bool = False
    static: bool = False
    time: Optional[List[str]] = None
    sequence: Optional[List[str]] = None
    loader_kitti_config: KittiLoaderConfig = field(default_factory=KittiLoaderConfig)

@dataclass
class ConnectionConfig:
    address:str | None = None # address to connect to - None for default

def rr_set_time_from_config(log_config:LogConfig) -> None:
    if not log_config.timeless and log_config.time is not None:
        for time_str in log_config.time:
            parts = time_str.split("=")
            if len(parts) != 2:
                continue
            timeline_name, time = parts
            rr.set_time_nanos(timeline_name, int(time))

        for time_str in log_config.sequence:
            parts = time_str.split("=")
            if len(parts) != 2:
                continue
            timeline_name, time = parts
            rr.set_time_sequence(timeline_name, int(time))


class FileTypeError(Exception):
    """Custom exception for unsupported file types."""

    def __init__(self, message="Cannot Load this file"):
        self.message = message
        super().__init__(self.message)
        

def log_static_cloud(log_config:LogConfig, key:str, points_xyz:np.array, points_intensity:Optional[np.ndarray]=None):
    logging.debug(points_xyz)
    rr.log(
        get_path(log_config, key),
        rr.Points3D(points_xyz, radii=0.1, colors=[255, 0, 0]),
        static=True,
        timeless=True
    )

def log_dynamic_cloud_by_sequence_idx(log_config:LogConfig, key:str, seq_idx:int, points_xyz:np.array, points_intensity:Optional[np.ndarray]=None):
    logging.debug(f"Log cloud {points_xyz.shape} at sequence id {seq_idx}" )
    rr.set_time_sequence("frame_nr", seq_idx)
    rr.log(
        get_path(log_config, key),
        rr.Points3D(points_xyz, radii=0.1, colors=[255, 0, 0]),
    )
        
        
def is_single_kitti_cloud_file(filepath:Path):
    if not filepath.is_file():
        is_cloud_file = False
    else:
        try:
            data = read_kitti_bin(filepath)
            is_cloud_file = data is not None
        except Exception as e:
            logging.info(f"Error while trying to load as kitti file: {filepath} - {type(e)}")
            is_cloud_file =  False
    if not is_cloud_file:
        logging.info(f"Not a single kitti cloud file: {filepath}")
    return is_cloud_file

def load_single_kitti_cloud_file(log_config:LogConfig):
    file = Path(log_config.filepath)
    cloud = read_kitti_bin(file)
    log_static_cloud(log_config, 'cloud', cloud[:,0:3], cloud[:,3])



def _create_kitti_loader(path:Path)->KittiLoader:
    velodyne_path = 'velodyne'
    label_folder = 'labels'
    secondary_cloud_folder =  'radar'
    box_folder = 'boxes'
    kitti_loader = KittiLoader(path, 
                    cloud_folder = str(velodyne_path), 
                    label_folder = str(label_folder),
                    secondary_cloud_folder = str(secondary_cloud_folder),
                    box_folder = str(box_folder))
    logging.info("Created kitti loader")
    return kitti_loader


def is_kitti_dataset(filepath:Path):
    if not filepath.is_dir():
        logging.info("Not a kitti folder since no directory")
        is_loadable = False
    else:
        try:
            is_loadable = KittiLoader.can_load(filepath) 
            logging.info(f"Kitti Loader output can_load = {is_loadable}")
        except Exception as e:
            logging.info(f"Error while trying to call KittiLoader.can_load on folder:  {filepath} - {str(e)}")
            is_loadable =  False
    if not is_loadable:
        logging.info(f"Not a kitti folder: {filepath}")
    return is_loadable


def _get_numbers_from_numerated_files(folder_path:Path, suffix:str = ".bin"):
    # Get the numbers 
    numbers = sorted(
        int(file.stem) for file in Path(folder_path).iterdir() if file.suffix == suffix and file.is_file()
    )
    return numbers

def _cartesian_to_homogenous_rows(array):
    # transforms n x m to n x (m+1) by adding 1's in the last row
    return np.hstack((array, np.ones((array.shape[0], 1))))

def _homegenous_to_cartesian_rows(array):
    return array[:,0:3]

def _load_hdf5_clouds(path:Path)->list[tuple[str, np.ndarray]]:
    # Given one hdf5 file that contains all clouds for one frame, return
    # them as tuples of key and cloud
    with h5py.File(str(path), "r") as hf:
        clouds = []
        for key in hf.keys():
            data = hf[key][:]
            points_xyz = data[:, 0:4] # ensture that we have all components
            clouds.append((key, points_xyz))
    return clouds

def _load_kitti_hdf5_clouds(cloud_folder:Path, idx:int)->None | list[tuple[str, np.ndarray]]:
    clouds_files = [ cloud_folder / f"{idx:06}.npy", cloud_folder / f"{idx:06}.bin" ]
    clouds_files = sorted([f for f in clouds_files if f.exists()])
    if len(clouds_files) < 1:
        return None
    return _load_hdf5_clouds(clouds_files[0])


def is_npy_file(file_path):
    return file_path.is_file() and file_path.suffix == '.npy'

def find_subdirs_with_files(parent_dir, file_predicate)->list[Path]:
    """
    Finds subdirectories in the specified parent directory containing files 
    that satisfy the given predicate.
    """
    parent_path = Path(parent_dir)
    subdirs_with_files = []

    for subdir in parent_path.iterdir():
        if subdir.is_dir():
            contains_target_files = any(file_predicate(file) for file in subdir.iterdir() if file.is_file())
            if contains_target_files:
                subdirs_with_files.append(subdir)

    return subdirs_with_files


def _log_dynamic_cloud_with_pose(log_config, key:str,  seq_idx:int, pose:np.ndarray, points_xyzi:np.ndarray):
    points_xyz = points_xyzi[:,:3]
    # easy to read by unnessary double transpose
    points_xyz_transformed = _homegenous_to_cartesian_rows((pose @ _cartesian_to_homogenous_rows(points_xyz).T).T)
    log_dynamic_cloud_by_sequence_idx(log_config, key = key, seq_idx=seq_idx, points_xyz = points_xyz_transformed)

def load_kitti_dataset(log_config):
    filepath = Path(log_config.filepath)
    kitti_loader = _create_kitti_loader(filepath)
    logging.info("Loading KITTI dataset")
    
    
    poses = kitti_loader.poses
    indices = np.array(_get_numbers_from_numerated_files(kitti_loader.cloud_folder))
    indices = indices[ (indices>= log_config.loader_kitti_config.cloud_start) & (indices<=log_config.loader_kitti_config.cloud_end)]
    logging.info(f"Logging numpy subdirectories: {kitti_loader.cloud_folder} with {indices}")
    
    for idx in indices:
        pose = poses[idx, ...]
        
        # Load the "normal" cloud of the kitti data
        points_xyzi = kitti_loader.get_cloud(idx)
        if points_xyzi is not None:
            points_xyz = points_xyzi[:,:3]
            # easy to read by unnessary double transpose
            points_xyz_transformed = _homegenous_to_cartesian_rows((pose @ _cartesian_to_homogenous_rows(points_xyz).T).T)
            log_dynamic_cloud_by_sequence_idx(log_config, key = str(kitti_loader.cloud_folder), seq_idx=idx, points_xyz = points_xyz_transformed)

    hdf5_subdirs = find_subdirs_with_files(filepath, is_hdf5_file)
    logging.info(f"Logging HDF5 subdirectories {hdf5_subdirs} with indincess {indices}")
    for hdf5_subdir in hdf5_subdirs:
        for idx in indices:
            pose = poses[idx, ...]
            rel_path = hdf5_subdir.relative_to(filepath)
            
            # Load the "normal" cloud of the kitti data
            clouds = _load_kitti_hdf5_clouds(hdf5_subdir, idx)
            if clouds is None:
                continue
            for key, cloud_xyzi in clouds:
                _log_dynamic_cloud_with_pose(log_config, key=str(rel_path / key), seq_idx=idx, pose=pose, points_xyzi=cloud_xyzi)
    


def is_python_file(filepath:Path):
    return filepath.is_file() and filepath.suffix == ".py"

def get_path(log_config: LogConfig, path):
    if log_config.entity_path_prefix is None or log_config.entity_path_prefix == "":
        return path
    else:
        return f"{log_config.entity_path_prefix}/{path}"

def load_python_file(log_config: LogConfig):
    file = Path(log_config.filepath)
    logging.info(f"Load python file: {file}")
    with file.open(encoding="utf8") as f:
        body = f.read()
        text = f"""# Python code\n```python\n{body}\n```\n"""
        rr.log(
            get_path(log_config, file.stem), 
            rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), 
            static=log_config.static or log_config.timeless
        )


def set_time_from_args(args) -> None:
    if not args.timeless and args.time is not None:
        for time_str in args.time:
            parts = time_str.split("=")
            if len(parts) != 2:
                continue
            timeline_name, time = parts
            rr.set_time_nanos(timeline_name, int(time))

        for time_str in args.sequence:
            parts = time_str.split("=")
            if len(parts) != 2:
                continue
            timeline_name, time = parts
            rr.set_time_sequence(timeline_name, int(time))

def is_hdf5_file(filepath: Path):
    try:
        with h5py.File(str(filepath), "r") as _:
            return True
    except (OSError, IOError) as e:
        return False

def is_hdf5_data(filepath: Path):
    """
    Check if a given file is a valid HDF5 file.

    Parameters:
        filepath (str): The path to the file to check.

    Returns:
        bool: True if the file is a valid HDF5 file, False otherwise.
    """
    try:
        with h5py.File(str(filepath), "r") as _:
            logging.info(f"{filepath} is hdf5 file")
            return True
    except (OSError, IOError) as e:
        logging.debug(f"Error: {e}")
        logging.info(f"Not a hdf5 file: {str(filepath)}")
        return False
            

def log_hdf5_to_cloud(hf, key: str, log_config: LogConfig):
    """
    Load a dataset from an open HDF5 file.

    Parameters:
        hf (h5py.File): Open HDF5 file object.
        key (str): Key of the dataset to load.
    """
    if key in hf:
        data = hf[key][:]
        logging.info(
            f"Loading key {key} to cloud with shape {data.shape} to {type(data)}"
        )
        points_xyz = data[:, 0:3]
        log_static_cloud(log_config, key, points_xyz)
    else:
        raise KeyError(f"Dataset key '{key}' not found in the file.")


def load_hdf5_file(log_config: LogConfig):
    """
    Open an HDF5 file, iterate over its keys, and load datasets.
    """
    logging.info(f"Opening HDF5 file: {log_config.filepath}")
    try:
        with h5py.File(log_config.filepath, "r") as hf:
            keys = list(hf.keys())
            logging.info(f"HDF5 Keys: {list(hf.keys())}")
            for key in keys:
                log_hdf5_to_cloud(hf, key, log_config)
    except Exception as e:
        print(f"Error opening file: {e}")


def load_example_cloud(log_config):
    points = np.random.random((100,3))
    logging.info(f"Logging example points with shape {points.shape}")
    rr.log(
        get_path(log_config, "example_cloud"),
        rr.Points3D(points, radii=0.1, colors=[255, 0, 0]),
        static=True,
        timeless=True
    )


def load_file(log_config: LogConfig):
    file = Path(log_config.filepath)
    if not file.exists():
        logging.info(f"{log_config.filepath} does not exist.")
        raise FileNotFoundError(
            f"Cannot load file: {log_config.filepath} does not exist."
        )
    if is_python_file(file):
        load_python_file(log_config)
    elif is_kitti_dataset(file):
        load_kitti_dataset(log_config)
    elif is_single_kitti_cloud_file(file):
        load_single_kitti_cloud_file(log_config)
    elif is_hdf5_data(file):
        load_hdf5_file(log_config)
    else:
        logging.info(f"Cannot load this file type: {file}")
        raise FileTypeError("Cannot load this file")


def run(standalone=False):
    logging.info(f'Load rerun loader with standalone = {standalone}')
    # Exactly use CLI args in this way - this is the API specified by rerun
    # to use this loader
    parser = argparse.ArgumentParser(
        description="Load a dataset from an HDF5 file and print parameters."
    )
    parser.add_argument("filepath", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "--application-id", type=str, help="Optional recommended ID for the application"
    )
    parser.add_argument(
        "--recording-id", type=str, help="Optional recommended ID for the recording"
    )
    parser.add_argument(
        "--entity-path-prefix", type=str, help="Optional prefix for all entity paths"
    )
    parser.add_argument(
        "--timeless",
        action="store_true",
        default=False,
        help="Alias for `--static` (deprecated)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        default=False,
        help="Mark data to be logged as static",
    )
    parser.add_argument(
        "--time",
        type=str,
        action="append",
        help="Optional timestamps to log at (e.g., `--time sim_time=1709203426`)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        action="append",
        help="Optional sequences to log at (e.g., `--sequence sim_frame=42`)",
    )
    # Additonal parameters
    parser.add_argument(
        "--addr",
        help="Address to connect to",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=standalone,
        help="Use as standalone rerun viewer which starts rerun, i.e. not a dataloader",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        default=False,
        help="Log out an example",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Opional config to set parameters",
    )
    parser.add_argument(
        "--writeconfig",
        type=str,
        help="Path to write the config",
    )
    args = parser.parse_args()
    
    if args.config:
        kitti_config = load_config_from_file(Path(args.config))
    else:
        kitti_config = None

    default_log_config = LogConfig(
        filepath='.', 
        recording_id=f'sm-{np.random.randint(0,100)}'
    )
        
    log_config = LogConfig(
        filepath=args.filepath,
        application_id=args.application_id or default_log_config.application_id,
        recording_id=args.recording_id or default_log_config.recording_id,
        entity_path_prefix=args.entity_path_prefix or default_log_config.entity_path_prefix,
        timeless=args.timeless,
        static=args.static,
        time=args.time,
        sequence=args.sequence,
        loader_kitti_config=kitti_config or default_log_config.loader_kitti_config
    )
    
    con_config = ConnectionConfig(
        address=args.addr or None
    )
    
    logging.info(f"Parameters: {log_config}")
    if args.writeconfig:
        write_config_to_file(Path(args.writeconfig), log_config.loader_kitti_config)
    
    if args.standalone:
        logging.info("Starting rerun-viewer-sm in standalone mode")
        if con_config.address is None:
            logging.info("Spawn and init new rerun")
            rr.init(log_config.application_id, recording_id=log_config.recording_id, 
                    spawn=True)
        else:
            rr.init(log_config.application_id, recording_id=log_config.recording_id)
            logging.info(f"Connect to existing rerun at {con_config.address}")
            rr.connect_tcp(con_config.address)
    else:
        logging.info("Starting rerun-loader-sm in data loader mode")
        rr.init(log_config.application_id, recording_id=log_config.recording_id)
        # The most important part of this: log to standard output so the Rerun Viewer 
        # can ingest it!
        rr.stdout()
    
    set_time_from_args(args)
    
    # Handle the example case
    if args.example:
        load_example_cloud(log_config)
        sys.exit(0)  # Exit code 0 indicates success

    try:
        load_file(log_config)
        sys.exit(0)  # Exit code 0 indicates success
    except Exception as e:
        logging.error(f"Error loading file {e}")
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)
        
def main_loader():
    run(standalone=False)

def main_standalone():
    run(standalone=True)

if __name__ == "__main__":
    main_standalone()
