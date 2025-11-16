import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # r value: tuple of (train_df, val_df, test_df)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # first split: train vs (val + test)
    train_df, temp_df = train_test_split(
        data,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=stratify,
    )
    
    # second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=stratify if stratify is not None else None,
    )
    
    return train_df, val_df, test_df


def normalize_targets(
    targets: np.ndarray,
    method: str = 'standard',
    fit_on: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any]:
    if fit_on is None:
        fit_on = targets
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    scaler.fit(fit_on.reshape(-1, 1))
    normalized = scaler.transform(targets.reshape(-1, 1)).flatten()
    
    return normalized, scaler


def denormalize_targets(
    normalized_targets: np.ndarray,
    scaler: Any,
) -> np.ndarray:
    return scaler.inverse_transform(normalized_targets.reshape(-1, 1)).flatten()


def validate_smiles_batch(smiles_list: list) -> list:
    from rdkit import Chem
    
    valid = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            valid.append(mol is not None)
        except:
            valid.append(False)
    
    return valid

