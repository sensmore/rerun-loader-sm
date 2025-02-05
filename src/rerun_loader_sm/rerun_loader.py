import rerun as rr
import rerun.blueprint as rrb

import numpy as np
import argparse
import h5py
from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Optional, Type  # keep it backward compatible!
import sys
import logging
import yaml
import traceback
import pickle
from abc import ABC, abstractmethod

from itertools import islice



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

# TODO: optimize: often file is already loaded during testing
# TODO: logs are very needed when making different checks on files but sometimes
# the cheks are needed internally without out -> provide optional logging instance to
# function 
# TODO: remove _config in Config attributes




class KittiLoaderConfig(BaseModel):
    cloud_start: float = 0
    cloud_end: float = float("inf")

class ConnectionConfig(BaseModel):
    address: Optional[str] = None  # Address to connect to, None for default

class LoaderConfig(BaseModel):
    forced_loader: Optional[str] = None
    loader_kitti_config: KittiLoaderConfig = Field(default_factory=KittiLoaderConfig)
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)

class LogConfig(BaseModel):
    filepath: str
    application_id: str = "rerun_loader_sm"
    recording_id: Optional[str] = None
    entity_path_prefix: str = ""
    timeless: bool = False
    static: bool = False
    time: Optional[List[str]] = None
    sequence: Optional[List[str]] = None
    loader_config: LoaderConfig = Field(default_factory=LoaderConfig)

# TODO: better parsing, e.g with pydantic?
def load_config_from_file(config_path: Path) -> LoaderConfig:
    with config_path.open('r') as file:
        yaml_data = yaml.safe_load(file)
        return LoaderConfig.model_validate(yaml_data)

def write_config_to_file(config_path: Path, config_instance) -> None:
    with config_path.open('w') as file:
        yaml.safe_dump(config_instance.dict(), file)



class DatasetRerunLoader(ABC):
    @staticmethod
    @abstractmethod
    def is_loadable(filepath:Path) -> bool: ...

    @abstractmethod
    def log_to_rerun(self, log_config:LogConfig) -> None : ...


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


def _get_numbers_from_numerated_files(folder_path:Path, suffix:str = ".bin"):
    # Get the numbers 
    numbers = sorted(
        int(file.stem) for file in Path(folder_path).iterdir() if file.suffix == suffix and file.is_file() and file.stem.isdigit()
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


def is_dir_containing_files_with(dir:Path, file_predicate)->bool:
    print(dir)
    # only test at most 10
    # TODO: improve - maybe first filter by extension
    return any(islice((file_predicate(file) for file in dir.iterdir() if file.is_file()), 10))


def find_subdirs_with_files(parent_dir, file_predicate)->list[Path]:
    """
    Finds subdirectories in the specified parent directory containing files 
    that satisfy the given predicate.
    """
    parent_path = Path(parent_dir)
    subdirs_with_files = []

    for subdir in parent_path.iterdir():
        if subdir.is_dir() and is_dir_containing_files_with(subdir, file_predicate):
                subdirs_with_files.append(subdir)
    return subdirs_with_files


def _log_dynamic_cloud_with_pose(log_config, key:str,  seq_idx:int, pose:np.ndarray, points_xyzi:np.ndarray):
    points_xyz = points_xyzi[:,:3]
    # easy to read by unnessary double transpose
    points_xyz_transformed = _homegenous_to_cartesian_rows((pose @ _cartesian_to_homogenous_rows(points_xyz).T).T)
    log_dynamic_cloud_by_sequence_idx(log_config, key = key, seq_idx=seq_idx, points_xyz = points_xyz_transformed)


def get_path(log_config: LogConfig, path, *more_path_parts):
    relative_path_str = '/'.join([path, *more_path_parts])
    if log_config.entity_path_prefix is None or log_config.entity_path_prefix == "":
        return relative_path_str
    else:
        return f"{log_config.entity_path_prefix}/{path}"



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

def is_python_file(filepath:Path):
    return filepath.is_file() and filepath.suffix == ".py"

def is_hdf5_file(filepath: Path):
    try:
        with h5py.File(str(filepath), "r") as _:
            return True
    except (OSError, IOError) as e:
        return False

def is_kitti_bin_file(filepath: Path):
    return filepath.is_file() and filepath.suffix == ".bin"

def is_npy_file(file_path):
    return file_path.is_file() and file_path.suffix == '.npy'

def _log_kitti_dir_to_rerun(log_config: LogConfig, poses_path:Path, cloud_dirs: List[Path] = None, 
                            hdf5_cloud_dirs:List[Path] = None):
        cloud_dirs = [ ] if cloud_dirs is None else cloud_dirs
        hdf5_cloud_dirs = [ ] if hdf5_cloud_dirs is None else hdf5_cloud_dirs
        assert isinstance(cloud_dirs, List)
        assert isinstance(hdf5_cloud_dirs, List)
        cloud_dirs = cloud_dirs or []
        hdf5_cloud_dirs = hdf5_cloud_dirs or []
        
        filepath = Path(poses_path.absolute().parent)
        kitti_loader = _create_kitti_loader(filepath)
        logging.info(f"Loading KITTI dataset {filepath}")
        
        
        poses = kitti_loader.poses
        translations = poses[:,0:3,3]
        assert translations.shape[1] == 3, "translations shape must be (n,3)"
        log_static_cloud(log_config, 'poses', translations)
        indices = np.arange(poses.shape[0])
        indices = indices[ (indices>= log_config.loader_config.loader_kitti_config.cloud_start) & (indices<=log_config.loader_config.loader_kitti_config.cloud_end)]
        logging.info(f"Logging kitti bin subdirectories: {cloud_dirs} with {indices}")
        
        # first iterate over index, then over clouds so we see all clouds at the same time
        for idx in indices:
            pose = poses[idx, ...]
            log_dynamic_cloud_by_sequence_idx(log_config, key = 'world/pose', seq_idx=idx, points_xyz = translations[idx])
            
            for cloud_dir in cloud_dirs:
            # Load the "normal" cloud of the kitti data
                points_xyzi = kitti_loader.get_cloud(idx, cloud_dir)
                if points_xyzi is not None:
                    points_xyz = points_xyzi[:,:3]
                    # easy to read by unnessary double transpose
                    points_xyz_transformed = _homegenous_to_cartesian_rows((pose @ _cartesian_to_homogenous_rows(points_xyz).T).T)
                    log_dynamic_cloud_by_sequence_idx(log_config, key = 'world/' + str(cloud_dir), seq_idx=idx, points_xyz = points_xyz_transformed)

        logging.info(f"Logging HDF5 subdirectories {hdf5_cloud_dirs} with indincess {indices}")
        for hdf5_subdir in hdf5_cloud_dirs:
            for idx in indices:
                pose = poses[idx, ...]
                rel_path = hdf5_subdir.relative_to(filepath)
                
                # Load the "normal" cloud of the kitti data
                clouds = _load_kitti_hdf5_clouds(hdf5_subdir, idx)
                if clouds is None:
                    continue
                for key, cloud_xyzi in clouds:
                    _log_dynamic_cloud_with_pose(log_config, key=str(rel_path / key), seq_idx=idx, pose=pose, points_xyzi=cloud_xyzi)
    

class KittiCloudDirectoryRerunLoader(DatasetRerunLoader):
    '''Specify a cloud direcoty in kitti subset and just log this'''
    
    @staticmethod
    def is_loadable(filepath:Path):
        logging.info(f"Checking if {filepath} is kitti cloud directory")
        if not filepath.is_dir():
            logging.info("Not a kitti cloud folder since no directory")
            return False
        kitti_cloud_dir = Path(filepath).absolute().parent
        logging.info(f"Checking if {kitti_cloud_dir} is kitti base dir")
        return KittiSequenceDatasetRerunLoader.is_loadable(kitti_cloud_dir)
    
    def log_to_rerun(self, log_config: LogConfig):
        cloud_dir = Path(log_config.filepath)
        kitti_cloud_dir = Path(cloud_dir).absolute().parent
        _log_kitti_dir_to_rerun(log_config, 
                                kitti_cloud_dir / 'poses.txt', cloud_dirs=[cloud_dir])

class KittiSequenceDatasetRerunLoader(DatasetRerunLoader):
    '''Log all the clouds in a kitti sequence dataset; not just a subdirectory with clouds'''
    
    @staticmethod
    def is_loadable(filepath:Path):
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
    
    def log_to_rerun(self, log_config: LogConfig):
        kitti_base_dir_path = Path(log_config.filepath)
        logging.info("Loading complete KITTI sequence")
        
        cloud_folders = find_subdirs_with_files(kitti_base_dir_path, is_kitti_bin_file)
        hdf5_cloud_folders = find_subdirs_with_files(kitti_base_dir_path, is_hdf5_file)
        logging.info(f"Logging kitti with cloud_folders={cloud_folders} and hdf5_cloud_folders={hdf5_cloud_folders}")
        _log_kitti_dir_to_rerun(log_config, kitti_base_dir_path / 'poses.txt', cloud_folders, hdf5_cloud_folders)
class KittiSingleCloudFileRerunLoader(DatasetRerunLoader):
    
    @staticmethod
    def is_loadable(filepath:Path):
        logging.info(f"Checking if is single kitti cloud file {filepath}")
        if not filepath.is_file():
            logging.info(f"Not kitti cloud file since not a file {filepath}")
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
        logging.info(f"Is single kitti cloud file: {is_cloud_file}")
        return is_cloud_file
    
    def log_to_rerun(self, log_config: LogConfig):
        file = Path(log_config.filepath)
        cloud = read_kitti_bin(file)
        log_static_cloud(log_config, 'cloud', cloud[:,0:3], cloud[:,3])


class PythonFileRerunLoader(DatasetRerunLoader):
    
    @staticmethod
    def is_loadable(filepath:Path):
        return is_python_file(filepath)
    
    def log_to_rerun(self, log_config: LogConfig):
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

def iterable_to_array(xs):
    return lambda xs : np.array(xs)

_trafos = {
    'idx': lambda x : x, 
    'positive_idxs': iterable_to_array,
    'negative_idx': iterable_to_array,
    'hard_idxs': iterable_to_array,
}

class PickledDictRerunLoader(DatasetRerunLoader):
    
    @staticmethod
    def is_loadable(filepath:Path):
        logging.info(f"Checking if it is pickle file {filepath}")
        if not filepath.is_file() and filepath.suffix != '.pickle':
            logging.info(f"Not picke file since not a file ending in .pickle {filepath}")
            is_expected_filetype = False
        try:
            with filepath.open('rb') as file:
                data = pickle.load(file)
                is_list = isinstance(data, list)
                is_dict = isinstance(data[0], dict)
                is_expected_filetype = is_list and is_dict
                logging.info(f"Is list dict in pickle is_list={is_list} is_dict={is_dict}")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, IsADirectoryError, PermissionError, ValueError):
            logging.info(f"Error during unpickling: {filepath}")
            is_expected_filetype = False
        logging.info(f"Is pickled dict {filepath}: {is_expected_filetype}")
        return is_expected_filetype
    
    def log_to_rerun(self, log_config: LogConfig):
        logging.info(f"Loading pickeled dict {log_config.filepath}")
        filepath = Path(log_config.filepath)
        with filepath.open('rb') as file:
            data = pickle.load(file)
        logging.info(f"Unpickled data of length {len(data)}")
        
        for i, dict_obj in enumerate(data):
            rr.set_time_sequence("frame_nr", i)
            for key in dict_obj:       
                val = str(dict_obj[key])
                rr.log(
                    get_path(log_config, log_config.filepath, key),
                    rr.TextLog(str(val))
                )
        logging.info("Loading pickeled dict done")

    
class HD5CloudRerunLoader(DatasetRerunLoader):
    
    @staticmethod
    def is_loadable(filepath:Path):
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

    def log_hdf5_to_cloud(self, hf:h5py.File, key: str, log_config: LogConfig):
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
    
    def log_to_rerun(self, log_config: LogConfig):
        """
        Open an HDF5 file, iterate over its keys, and load datasets.
        """
        logging.info(f"Opening HDF5 file: {log_config.filepath}")
        try:
            with h5py.File(log_config.filepath, "r") as hf:
                keys = list(hf.keys())
                logging.info(f"HDF5 Keys: {list(hf.keys())}")
                for key in keys:
                    self.log_hdf5_to_cloud(hf, key, log_config)
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
    # TODO: refactor; we need 
    # - priorities
    # - fallthrough (one fails - load from other)
    # - maybe parallel checks - only do easy checks first of all -> use coprocess (yield)
    rerun_data_loader_classes = [
        PythonFileRerunLoader,
        PickledDictRerunLoader,
        KittiCloudDirectoryRerunLoader,
        KittiSequenceDatasetRerunLoader,
        KittiSingleCloudFileRerunLoader,
        HD5CloudRerunLoader,
    ]
    loader_classes_by_name = {cls.__name__: cls for cls in rerun_data_loader_classes}
    logging.info(f"Consider rerun data loaders: {list(loader_classes_by_name.keys())}")
    
    selected_rerun_loader_class = None
    if log_config.loader_config.forced_loader is not None:
        logging.info(f"Selecting forced loader: {log_config.loader_config.forced_loader}")
        selected_rerun_loader_class = loader_classes_by_name[log_config.loader_config.forced_loader]
    else:
        for rerun_loader_class in rerun_data_loader_classes:
            if rerun_loader_class.is_loadable(Path(file)):
                selected_rerun_loader_class = rerun_loader_class
                break
    
    if selected_rerun_loader_class is not None:
        rerun_loader = selected_rerun_loader_class()
        rerun_loader.log_to_rerun(log_config)
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
    parser.add_argument(
        "--forcedloader",
        type=str,
        help="Name of loader to force instead of choosing best loader heuristically",
    )
    args = parser.parse_args()
    
    # TODO: reorganize this loading with jsonargparse but make sure CLI args stay
    # exaclty the same; CLI args are interface to rerun when used as loader
    
    loader_config = load_config_from_file(Path(args.config)) if args.config else None

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
        loader_config=loader_config or default_log_config.loader_config
    )
    
    
    
    if args.addr is not None:
        log_config.loader_config.connection.address = args.addr

    if args.forcedloader:
        log_config.loader_config.forced_loader = args.forcedloader
    
    logging.info(f"Parameters: {log_config}")
    
    if args.standalone:
        logging.info("Starting rerun-viewer-sm in standalone mode")
        if log_config.loader_config.connection.address is None:
            logging.info("Spawn and init new rerun")
            rr.init(log_config.application_id, recording_id=log_config.recording_id, 
                    spawn=True)
        else:
            rr.init(log_config.application_id, recording_id=log_config.recording_id)
            address = log_config.loader_config.connection
            logging.info(f"Connect to existing rerun at {address}")
            rr.connect_tcp(address)
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

    if args.example:
        load_example_cloud(log_config)
        sys.exit(0)  # Exit code 0 indicates success


    try:
        load_file(log_config)
        sys.exit(0)  # Exit code 0 indicates success
    except Exception as e:
        logging.error(f"Error loading file: {e} {traceback.format_exc()})")
        logging.error("Error during loading. Exiting with exitcode to signal load of this filetype is not possible")
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)
        
def main_loader():
    run(standalone=False)

def main_standalone():
    run(standalone=True)

if __name__ == "__main__":
    main_standalone()
