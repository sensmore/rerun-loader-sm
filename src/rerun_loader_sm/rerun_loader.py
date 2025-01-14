import rerun as rr
import rerun.blueprint as rrb

import numpy as np
import argparse
import h5py
from pathlib import Path

from dataclasses import dataclass
from typing import List, Optional  # keep it backward compatible!
import sys
import logging

# Define the log file path
log_file = "/tmp/rerun-loader-sm-log.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.StreamHandler(sys.stderr),  # Log to stderr
        logging.FileHandler(log_file)      # Log to the specified file
    ]
)


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


def is_hdf5_file(filepath: Path):
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


def load_hdf5_to_cloud(hf, key: str, log_config: LogConfig):
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
        points_label = data[:, 3]
        print(points_xyz)
        rr.log(
            f"{log_config.entity_path_prefix}/{key}",
            rr.Points3D(points_xyz, radii=2.0, colors=[255, 0, 0]),
        )
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
                load_hdf5_to_cloud(hf, key, log_config)
    except Exception as e:
        print(f"Error opening file: {e}")


def load_dataset(log_config: LogConfig):
    if not Path(log_config.filepath).exists():
        logging.info(f"{log_config.filepath} does not exist.")
        raise FileNotFoundError(
            f"Cannot load file: {log_config.filepath} does not exist."
        )
    if is_hdf5_file(log_config.filepath):
        load_hdf5_file(log_config)
    else:
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
        "--standalone",
        action="store_true",
        default=standalone,
        help="Use as standalone rerun viewer which starts rerun, i.e. not a dataloader",
    )
    args = parser.parse_args()

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
    )
    
    logging.info(f"Parameters: {log_config}")
    
    if args.standalone:
        logging.info("Starting rerun-viewer-sm in standalone mode")
        rr.init(log_config.application_id, recording_id=log_config.recording_id, 
                spawn=True)
    else:
        logging.info("Starting rerun-loader-sm in data loader mode")
        rr.init(log_config.application_id, recording_id=log_config.recording_id)
        # The most important part of this: log to standard output so the Rerun Viewer 
        # can ingest it!
        rr.stdout()

    try:
        load_dataset(log_config)
        sys.exit(0)  # Exit code 0 indicates success
    except Exception as e:
        logging.error("Starting rerun-loader-sm in data loader mode")
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)
        
def main_loader():
    run(standalone=False)

def main_standalone():
    run(standalone=True)

if __name__ == "__main__":
    main_standalone()
