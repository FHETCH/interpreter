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

#TODO: remove this method use only the mrp one
def save_ciphertext(ct: Ciphertext, path: Union[str, Path]) -> None:
    """Save a complete Ciphertext (2 MRPs + scale) to disk in .npz format.
    
    Args:
        ct: The Ciphertext object to save
        path: Output file path (will be created/overwritten)
        
    File format:
        - 'moduli': array of prime moduli (shared by both polynomials)
        - 'scale': ciphertext scale
        - 'ct_0_limb_0', 'ct_0_limb_1', ...: limbs of first polynomial
        - 'ct_1_limb_0', 'ct_1_limb_1', ...: limbs of second polynomial
    """
    path = Path(path)
    
    # Extract the two polynomials
    ct_0 = ct.polynomials[0]
    ct_1 = ct.polynomials[1]
    
    # Get moduli (should be the same for both polynomials)
    moduli = np.array(list(ct_0.values.keys()))
    
    # Create save dictionary with moduli, scale, and all limbs
    save_dict = {'moduli': moduli, 'scale': ct.scale}
    for i, mod in enumerate(moduli):
        save_dict[f'ct_0_limb_{i}'] = ct_0.values[mod].value
        save_dict[f'ct_1_limb_{i}'] = ct_1.values[mod].value
    
    np.savez(path, **save_dict)


def load_ciphertext(path: Union[str, Path]) -> Ciphertext:
    """Load a complete Ciphertext from disk.
    
    Args:
        path: Path to the .npz file to load
        
    Returns:
        The reconstructed Ciphertext object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the file is missing required keys
    """
    path = Path(path)
    
    # Load the .npz file
    ct_data = np.load(path)
    
    # Extract metadata
    moduli = ct_data['moduli']
    scale = ct_data['scale'].item() if 'scale' in ct_data else 2**30
    
    # Reconstruct MRP objects from limbs
    ct_0_values = {int(moduli[i]): Vector(ct_data[f'ct_0_limb_{i}']) for i in range(len(moduli))}
    ct_1_values = {int(moduli[i]): Vector(ct_data[f'ct_1_limb_{i}']) for i in range(len(moduli))}
    
    ct_0 = MRP(ct_0_values)
    ct_1 = MRP(ct_1_values)
    
    return Ciphertext(scale=scale, polynomials=[ct_0, ct_1])
