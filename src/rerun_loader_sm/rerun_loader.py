import rerun as rr
import rerun.blueprint as rrb

import argparse
import h5py
from pathlib import Path

from dataclasses import dataclass
from typing import List, Optional # keep it backward compatible!
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class LogConfig:
    filepath: str
    application_id: str = "rerun_example_external_data_loader"
    recording_id: Optional[str] = None
    entity_path_prefix: str = ""
    timeless: bool = False
    static: bool = False
    time: Optional[List[str]] = None
    sequence: Optional[List[str]] = None

class FileTypeError(Exception):
    """Custom exception for unsupported file types."""
    def __init__(self, message="Cannot Load this file"):
        self.message = message
        super().__init__(self.message)


def is_hdf5_file(filepath:Path):
    """
    Check if a given file is a valid HDF5 file.
    
    Parameters:
        filepath (str): The path to the file to check.
    
    Returns:
        bool: True if the file is a valid HDF5 file, False otherwise.
    """
    try:
        with h5py.File(str(filepath), 'r') as _:
            logging.info(f"{filepath} is hdf5 file")
            return True
    except (OSError, IOError) as e:
        logging.debug(f"Error: {e}")
        logging.info(f"Not a hdf5 file: {str(filepath)}")
        return False


def load_hdf5_to_cloud(hf, key: str, log_config: LogConfig):
    """
    Load a dataset from an open HDF5 file.

    Parameters:
        hf (h5py.File): Open HDF5 file object.
        key (str): Key of the dataset to load.
    """
    if key in hf:
        data = hf[key][:]
        logging.info(f"Loading key {key} to cloud with shape {data.shape} to {type(data)}")
        points_xyz = data[:,0:3]
        points_label = data[:,3]
        rr.log(f"{log_config.entity_path_prefix}/{key}", rr.Points3D(points_xyz, radii=2.0, colors=[255, 0, 0]))
    else:
        raise KeyError(f"Dataset key '{key}' not found in the file.")

def load_hdf5_file(log_config: LogConfig):
    """
    Open an HDF5 file, iterate over its keys, and load datasets.
    """
    logging.info(f"Opening HDF5 file: {log_config.filepath}")
    try:
        with h5py.File(log_config.filepath, 'r') as hf:
            keys = list(hf.keys())
            logging.info(f"HDF5 Keys: {list(hf.keys())}")
            for key in keys:
                load_hdf5_to_cloud(hf, key, log_config)
    except Exception as e:
        print(f"Error opening file: {e}")

def load_dataset(log_config:LogConfig):
    if not Path(log_config.filepath).exists():
        logging.info(f"{log_config.filepath} does not exist.")
        raise FileNotFoundError(f"Cannot load file: {log_config.filepath} does not exist.")
    if is_hdf5_file(log_config.filepath):
        load_hdf5_file(log_config)
    else:
        raise FileTypeError("Cannot load this file")
    
def main():
    parser = argparse.ArgumentParser(description="Load a dataset from an HDF5 file and print parameters.")
    parser.add_argument("filepath", type=str, help="Path to the HDF5 file")
    parser.add_argument("--application-id", type=str, help="Optional recommended ID for the application")
    parser.add_argument("--recording-id", type=str, help="Optional recommended ID for the recording")
    parser.add_argument("--entity-path-prefix", type=str, help="Optional prefix for all entity paths")
    parser.add_argument("--timeless", action="store_true", default=False, help="Alias for `--static` (deprecated)")
    parser.add_argument("--static", action="store_true", default=False, help="Mark data to be logged as static")
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
    args = parser.parse_args()

    log_config = LogConfig(
        filepath=args.filepath,
        application_id=args.application_id,
        recording_id=args.recording_id,
        entity_path_prefix=args.entity_path_prefix,
        timeless=args.timeless,
        static=args.static,
        time=args.time,
        sequence=args.sequence,
    )

    rr.init(app_id, recording_id=log_config.recording_id)
    # The most important part of this: log to standard output so the Rerun Viewer can ingest it!
    rr.stdout()

    # Print all parameters
    logging.info(f"Parameters: {log_config}")

    try:
        load_dataset(log_config)
        sys.exit(0)  # Exit code 0 indicates success
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

if __name__ == "__main__":
    main()
    