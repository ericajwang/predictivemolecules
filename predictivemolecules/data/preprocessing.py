import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data


class MoleculePreprocessor:
    def __init__(self, sanitize: bool = True, remove_hs: bool = False):
        """
        sanitize: Whether to sanitize molecules
        remove_hs: Whether to remove hydrogens
        """
        self.sanitize = sanitize
        self.remove_hs = remove_hs
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.sanitize:
                Chem.SanitizeMol(mol)
            
            if self.remove_hs:
                mol = Chem.RemoveHs(mol)
            
            return mol
        except Exception as e:
            print(f"Error processing: {smiles}: {e}")
            return None
    
    def validate_smiles(self, smiles: str) -> bool:
        mol = self.smiles_to_mol(smiles)
        return mol is not None
    
    def get_molecular_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        if mol is None:
            return {}
        
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
            'num_hba': rdMolDescriptors.CalcNumHBA(mol),
        }


class GraphBuilder:
    def __init__(self, use_edge_features: bool = True):
        self.use_edge_features = use_edge_features
        self.atom_features_dim = 44  # dim of atom features
        self.bond_features_dim = 10  # dim of bond features
    
    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        features = []
        
        features.append(atom.GetAtomicNum())
        features.append(atom.GetDegree())
        features.append(atom.GetFormalCharge())
        features.append(atom.GetHybridization())
        features.append(atom.GetNumRadicalElectrons())
        features.append(atom.GetIsAromatic())
        features.append(atom.GetMass())
        features.append(atom.GetNumImplicitHs())
        
        element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        element_onehot = [1 if atom.GetAtomicNum() == e else 0 for e in element_list]
        features.extend(element_onehot)
        
        hybridization_list = [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
        ]
        hybridization_onehot = [
            1 if atom.GetHybridization() == h else 0 
            for h in hybridization_list
        ]
        features.extend(hybridization_onehot)
        
        features.append(atom.IsInRing())
        features.append(atom.GetChiralTag())
        features.append(atom.GetTotalNumHs())
        
        while len(features) < self.atom_features_dim:
            features.append(0)
        
        return np.array(features[:self.atom_features_dim], dtype=np.float32)
    
    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        features = []
        
        features.append(bond.GetBondTypeAsDouble())
        features.append(bond.GetIsAromatic())
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        
        bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]
        bond_type_onehot = [
            1 if bond.GetBondType() == bt else 0 
            for bt in bond_types
        ]
        features.extend(bond_type_onehot)
        
        features.append(bond.GetStereo())
        
        while len(features) < self.bond_features_dim:
            features.append(0)
        
        return np.array(features[:self.bond_features_dim], dtype=np.float32)
    
    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        if mol is None:
            return Data(
                x=torch.zeros((1, self.atom_features_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, self.bond_features_dim)) if self.use_edge_features else None
            )
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float32)
        
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            if self.use_edge_features:
                bond_features = self.get_bond_features(bond)
                edge_features.append(bond_features)
                edge_features.append(bond_features)  
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        if self.use_edge_features and edge_features:
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_attr = None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class FingerprintGenerator:
    def __init__(self, fp_type: str = 'ECFP', radius: int = 2, n_bits: int = 2048):
        """
        fp_type: Type of fingerprint ('ECFP', 'MACCS', 'RDKit')
        radius: Radius for ECFP
        n_bits: Number of bits for fingerprint
        """
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
    
    def generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        
        if self.fp_type == 'ECFP':
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.n_bits
            )
        elif self.fp_type == 'MACCS':
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_array = np.array(fp, dtype=np.float32)
            if len(fp_array) < self.n_bits:
                fp_array = np.pad(fp_array, (0, self.n_bits - len(fp_array)))
            else:
                fp_array = fp_array[:self.n_bits]
            return fp_array
        elif self.fp_type == 'RDKit':
            fp = Chem.RDKFingerprint(mol)
        return np.array(fp, dtype=np.float32)
    
    def generate_multiple_fingerprints(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        fingerprints = {}
        
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprints['ECFP'] = np.array(ecfp, dtype=np.float32)
        
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.array(maccs, dtype=np.float32)
        maccs_array = np.pad(maccs_array, (0, 2048 - len(maccs_array)))
        fingerprints['MACCS'] = maccs_array[:2048]
        
        rdkfp = Chem.RDKFingerprint(mol)
        rdkfp_array = np.array(rdkfp, dtype=np.float32)
        rdkfp_array = np.pad(rdkfp_array, (0, 2048 - len(rdkfp_array)))
        fingerprints['RDKit'] = rdkfp_array[:2048]
        
        return fingerprints


class FeatureExtractor:
    def __init__(self):
        self.preprocessor = MoleculePreprocessor()
        self.graph_builder = GraphBuilder()
        self.fp_generator = FingerprintGenerator()
    
    def extract_all_features(self, smiles: str) -> Dict:
        mol = self.preprocessor.smiles_to_mol(smiles)
        
        if mol is None:
            return {
                'graph': None,
                'fingerprint': np.zeros(2048, dtype=np.float32),
                'properties': {},
                'valid': False
            }
        
        graph = self.graph_builder.mol_to_graph(mol)
        fingerprint = self.fp_generator.generate_fingerprint(mol)
        properties = self.preprocessor.get_molecular_properties(mol)
        
        return {
            'graph': graph,
            'fingerprint': fingerprint,
            'properties': properties,
            'valid': True,
            'mol': mol
        }

