import argparse
import sys
import re
import os
from dataclasses import field

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train a Dec2Vec speech representation learning model.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="The name of configuration file to use.",
    )
    args = parser.parse_args()

    return args

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1  


def find_speaker_gender(speaker_dir, speaker_id):
    """
    Recursively search for speaker ID in directory structure and determine gender for TIMIT
    
    Args:
        speaker_dir (str): Root directory to search
        speaker_id (str): Speaker ID to find
    
    Returns:
        str: 'M' or 'F' indicating speaker gender
    """
            
    for root, dirs, files in os.walk(speaker_dir):
        # Look for speaker ID in current directory name
        if speaker_id in root:
            # For TIMIT, speaker gender is encoded in directory path:
            # .../TRAIN/DR1/FCJF0/... where F indicates female speaker
            speaker_folder = os.path.basename(root)
            return 'F' if speaker_folder.startswith('F') else 'M'
                    
    return None

