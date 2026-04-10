"""
Helper module for efficiently loading and processing FireProtDB data.
Handles large CSV files with chunking and filtering capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class FireProtDBLoader:
    """
    Efficient loader for FireProtDB CSV files with chunking support.
    """
    
    def __init__(self, csv_path: str, chunk_size: int = 10000):
        """
        Initialize FireProtDB loader.
        
        Args:
            csv_path: Path to FireProtDB CSV file
            chunk_size: Number of rows to process per chunk
        """
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FireProtDB file not found: {csv_path}")
    
    def get_total_rows(self) -> int:
        """Get total number of rows in CSV file."""
        count = 0
        for _ in pd.read_csv(self.csv_path, chunksize=self.chunk_size, usecols=[0]):
            count += len(_)
        return count
    
    def load_with_filters(
        self,
        required_columns: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        max_rows: Optional[int] = None,
        exclude_indices: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Load FireProtDB data with optional filtering.
        
        Args:
            required_columns: List of column names that must be present
            filters: Dictionary of column:value filters (e.g., {'DDG': lambda x: x.notna()})
            max_rows: Maximum number of rows to load
            exclude_indices: Set of row indices to exclude (e.g., training set indices)
            
        Returns:
            Filtered DataFrame
        """
        chunks = []
        rows_loaded = 0
        row_index_offset = 0
        
        self.logger.info(f"Loading data from {self.csv_path}...")
        
        for chunk_idx, chunk in enumerate(pd.read_csv(
            self.csv_path,
            chunksize=self.chunk_size,
            low_memory=False
        )):
            # Check required columns
            if required_columns:
                missing = set(required_columns) - set(chunk.columns)
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
            
            # Apply filters
            if filters:
                mask = pd.Series([True] * len(chunk), index=chunk.index)
                for col, filter_func in filters.items():
                    if col not in chunk.columns:
                        self.logger.warning(f"Filter column '{col}' not found, skipping")
                        continue
                    mask = mask & filter_func(chunk[col])
                chunk = chunk[mask].copy()
            
            # Exclude specific indices
            if exclude_indices:
                chunk_indices = set(range(row_index_offset, row_index_offset + len(chunk)))
                exclude_mask = chunk_indices & exclude_indices
                if exclude_mask:
                    exclude_relative = {idx - row_index_offset for idx in exclude_mask}
                    chunk = chunk.drop(chunk.index[list(exclude_relative)])
            
            if len(chunk) > 0:
                chunks.append(chunk)
                rows_loaded += len(chunk)
            
            row_index_offset += self.chunk_size
            
            # Check max_rows limit
            if max_rows and rows_loaded >= max_rows:
                break
            
            if (chunk_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {chunk_idx + 1} chunks, loaded {rows_loaded} rows")
        
        if not chunks:
            self.logger.warning("No data loaded after filtering")
            return pd.DataFrame()
        
        result = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Loaded {len(result)} rows total")
        
        return result
    
    def get_validation_set(
        self,
        exclude_indices: Set[int],
        min_ddg_samples: int = 100,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get validation set excluding training/val/test indices.
        
        Args:
            exclude_indices: Set of indices to exclude (training/val/test)
            min_ddg_samples: Minimum number of samples with DDG values required
            max_samples: Maximum number of samples to return
            
        Returns:
            DataFrame with validation sequences
        """
        filters = {
            'sequence': lambda x: x.notna() & (x.str.len() >= 10) & (x.str.len() <= 5000),
            'DDG': lambda x: x.notna()
        }
        
        validation_df = self.load_with_filters(
            required_columns=['sequence', 'DDG'],
            filters=filters,
            exclude_indices=exclude_indices,
            max_rows=max_samples
        )
        
        if len(validation_df) < min_ddg_samples:
            self.logger.warning(
                f"Only {len(validation_df)} validation samples found "
                f"(minimum {min_ddg_samples} requested)"
            )
        
        return validation_df
    
    def sample_random_sequences(
        self,
        n_samples: int,
        exclude_indices: Optional[Set[int]] = None,
        require_ddg: bool = True
    ) -> pd.DataFrame:
        """
        Sample random sequences from database.
        
        Args:
            n_samples: Number of samples to retrieve
            exclude_indices: Indices to exclude
            require_ddg: Whether to require DDG values
            
        Returns:
            DataFrame with sampled sequences
        """
        filters = {
            'sequence': lambda x: x.notna() & (x.str.len() >= 10) & (x.str.len() <= 5000)
        }
        
        if require_ddg:
            filters['DDG'] = lambda x: x.notna()
        
        df = self.load_with_filters(
            required_columns=['sequence', 'DDG'] if require_ddg else ['sequence'],
            filters=filters,
            exclude_indices=exclude_indices,
            max_rows=n_samples * 2  # Load extra to account for filtering
        )
        
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        
        return df.reset_index(drop=True)


def load_training_indices(splits_path: str) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Load training/validation/test indices from splits file.
    
    Args:
        splits_path: Path to data_splits.npz file
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices) as sets
    """
    splits = np.load(splits_path)
    train_indices = set(splits['train_indices'])
    val_indices = set(splits['val_indices'])
    test_indices = set(splits['test_indices'])
    
    return train_indices, val_indices, test_indices


def get_all_training_indices(splits_path: str) -> Set[int]:
    """
    Get all indices used in training (train + val + test).
    
    Args:
        splits_path: Path to data_splits.npz file
        
    Returns:
        Set of all training indices
    """
    train_indices, val_indices, test_indices = load_training_indices(splits_path)
    return train_indices | val_indices | test_indices

