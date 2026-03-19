"""Serialization and deserialization utilities for MRP objects.

All functions use numpy's .npz format for efficient storage of multiple arrays.
"""

import numpy as np
from pathlib import Path
from typing import Union

from fhetch.data import MRP, Vector


def save_mrp(mrp: MRP, path: Union[str, Path]) -> None:
    """Save a single MRP to disk in .npz format.
    
    Args:
        mrp: The MRP object to save
        path: Output file path (will be created/overwritten)
        
    File format:
        - 'moduli': array of prime moduli (preserves ordering)
        - 'limb_0', 'limb_1', ...: coefficient/evaluation arrays for each modulus
    """
    path = Path(path)
    
    # Extract moduli and preserve ordering
    moduli = np.array(list(mrp.values.keys()))
    
    # Create save dictionary with moduli and all limbs
    save_dict = {'moduli': moduli}
    for i, mod in enumerate(moduli):
        save_dict[f'limb_{i}'] = mrp.values[mod].value.astype(np.int64)
    
    np.savez(path, **save_dict)


def load_mrp(path: Union[str, Path]) -> MRP:
    """Load a single MRP from disk.
    
    Args:
        path: Path to the .npz file to load
        
    Returns:
        The reconstructed MRP object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the file is missing required keys
    """
    path = Path(path)
    
    # Load the .npz file
    data = np.load(path)
    
    # Extract moduli
    moduli = data['moduli']
    
    # Reconstruct the values dictionary
    values = {int(moduli[i]): Vector(data[f'limb_{i}']) for i in range(len(moduli))}
    
    return MRP(values)


def save_ksk(ksk: list[tuple[MRP, MRP]], directory: Union[str, Path], prefix: str) -> None:
    """Save a key-switching key (list of MRP pairs) to disk.

    Each pair is saved as {prefix}_d{i}_0.npz and {prefix}_d{i}_1.npz.
    """
    directory = Path(directory)
    for i, (ksk_0, ksk_1) in enumerate(ksk):
        save_mrp(ksk_0, directory / f"{prefix}_d{i}_0.npz")
        save_mrp(ksk_1, directory / f"{prefix}_d{i}_1.npz")

