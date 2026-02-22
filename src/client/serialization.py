"""Serialization and deserialization utilities for MRP and Ciphertext objects.

This module provides a unified interface for saving and loading:
- Individual MRP (Multi-Residue Polynomial) objects
- Complete Ciphertext objects (containing multiple MRPs + scale)

All functions use numpy's .npz format for efficient storage of multiple arrays.
"""

import numpy as np
from pathlib import Path
from typing import Union

from fhetch.data import MRP, Vector
from client.crypto import Ciphertext


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
        save_dict[f'limb_{i}'] = mrp.values[mod].value
    
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

