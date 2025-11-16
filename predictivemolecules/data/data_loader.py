import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from .preprocessing import (
    MoleculePreprocessor,
    GraphBuilder,
    FingerprintGenerator,
    FeatureExtractor
)


class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        smiles_column: str = 'SMILES',
        target_column: str = 'binding_affinity',
        mode: str = 'graph',
        transform: Optional[Callable] = None,
        preprocess: bool = True,
    ):
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.mode = mode
        self.transform = transform
        self.preprocess = preprocess
        
        self.preprocessor = MoleculePreprocessor()
        self.graph_builder = GraphBuilder()
        self.fp_generator = FingerprintGenerator()
        self.feature_extractor = FeatureExtractor()
        
        if data_path is not None:
            self.data = pd.read_csv(data_path)
            self._validate_data()
        
        if preprocess and len(self.data) > 0:
            self._preprocess_all()
    
    def _validate_data(self):
        """Validate that required columns exist."""
        required_columns = [self.smiles_column, self.target_column]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _preprocess_all(self):
        """Preprocess all molecules in the dataset."""
        self.processed_data = []
        
        for idx in range(len(self.data)):
            smiles = self.data.iloc[idx][self.smiles_column]
            target = self.data.iloc[idx][self.target_column]
            
            processed = self._process_single(smiles, target)
            if processed is not None:
                self.processed_data.append(processed)
    
    def _process_single(self, smiles: str, target: float) -> Optional[Dict]:
        mol = self.preprocessor.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        data_dict = {
            'smiles': smiles,
            'target': float(target),
            'valid': True
        }
        
        if self.mode in ['graph', 'both']:
            graph = self.graph_builder.mol_to_graph(mol)
            data_dict['graph'] = graph
        
        if self.mode in ['fingerprint', 'both']:
            fingerprint = self.fp_generator.generate_fingerprint(mol)
            data_dict['fingerprint'] = fingerprint
        
        if self.mode == 'both':
            properties = self.preprocessor.get_molecular_properties(mol)
            data_dict['properties'] = properties
        
        return data_dict
    
    def __len__(self) -> int:
        if self.preprocess:
            return len(self.processed_data)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.preprocess:
            sample = self.processed_data[idx].copy()
        else:
            row = self.data.iloc[idx]
            sample = self._process_single(
                row[self.smiles_column],
                row[self.target_column]
            )
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_graphs(batch: List[Dict]) -> Dict:
    graphs = [item['graph'] for item in batch]
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float32)
    
    batched_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batched_graph,
        'target': targets,
        'smiles': [item['smiles'] for item in batch]
    }


def collate_fingerprints(batch: List[Dict]) -> Dict:
    fingerprints = torch.stack([
        torch.tensor(item['fingerprint'], dtype=torch.float32)
        for item in batch
    ])
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float32)
    
    return {
        'fingerprint': fingerprints,
        'target': targets,
        'smiles': [item['smiles'] for item in batch]
    }


def collate_both(batch: List[Dict]) -> Dict:
    graphs = [item['graph'] for item in batch]
    fingerprints = torch.stack([
        torch.tensor(item['fingerprint'], dtype=torch.float32)
        for item in batch
    ])
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float32)
    
    batched_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batched_graph,
        'fingerprint': fingerprints,
        'target': targets,
        'smiles': [item['smiles'] for item in batch]
    }


class MoleculeDataLoader:
    @staticmethod
    def create_loader(
        dataset: MoleculeDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        if dataset.mode == 'graph':
            collate_fn = collate_graphs
        elif dataset.mode == 'fingerprint':
            collate_fn = collate_fingerprints
        elif dataset.mode == 'both':
            collate_fn = collate_both

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

