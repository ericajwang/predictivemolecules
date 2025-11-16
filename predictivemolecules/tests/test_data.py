"""
Unit tests for data loading and preprocessing.
"""

import pytest
import numpy as np
from data.preprocessing import (
    MoleculePreprocessor,
    GraphBuilder,
    FingerprintGenerator,
)
from rdkit import Chem


def test_molecule_preprocessor():
    """Test molecule preprocessor."""
    preprocessor = MoleculePreprocessor()
    
    # Valid SMILES
    mol = preprocessor.smiles_to_mol("CCO")  # Ethanol
    assert mol is not None
    
    # Invalid SMILES
    mol = preprocessor.smiles_to_mol("InvalidSMILES")
    assert mol is None
    
    # Validate SMILES
    assert preprocessor.validate_smiles("CCO") == True
    assert preprocessor.validate_smiles("Invalid") == False


def test_graph_builder():
    """Test graph builder."""
    builder = GraphBuilder()
    
    mol = Chem.MolFromSmiles("CCO")
    graph = builder.mol_to_graph(mol)
    
    assert graph.x.shape[0] == mol.GetNumAtoms()
    assert graph.edge_index.shape[0] == 2


def test_fingerprint_generator():
    """Test fingerprint generator."""
    generator = FingerprintGenerator(fp_type='ECFP', n_bits=2048)
    
    mol = Chem.MolFromSmiles("CCO")
    fp = generator.generate_fingerprint(mol)
    
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32

