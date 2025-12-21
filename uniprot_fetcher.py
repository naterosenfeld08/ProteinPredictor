"""
UniProt sequence fetcher for FireProtDB training pipeline
Fetches protein sequences from UniProt API using UniProt IDs
"""

import time
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Set
import logging
from collections import defaultdict
import json

# UniProt REST API endpoints
UNIPROT_API_BASE = "https://rest.uniprot.org"
UNIPROT_SEQUENCE_URL = f"{UNIPROT_API_BASE}/uniprotkb/{{uniprot_id}}.fasta"
UNIPROT_BATCH_URL = f"{UNIPROT_API_BASE}/uniprotkb/search"

# Rate limiting: UniProt allows 3 requests per second
MIN_REQUEST_INTERVAL = 0.34  # Slightly more than 1/3 second


class UniProtSequenceFetcher:
    """
    Fetches protein sequences from UniProt API with caching and rate limiting
    """
    
    def __init__(self, cache_dir: str = "./uniprot_cache", max_retries: int = 3):
        """
        Initialize UniProt sequence fetcher
        
        Args:
            cache_dir: Directory to cache fetched sequences
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.last_request_time = 0
        self.request_count = 0
        self.cache_file = self.cache_dir / "sequence_cache.json"
        self.sequence_cache = self._load_cache()
        self.logger = logging.getLogger(__name__)
        
    def _load_cache(self) -> Dict[str, str]:
        """Load cached sequences from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save sequence cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.sequence_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting for UniProt API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def fetch_sequence(self, uniprot_id: str) -> Optional[str]:
        """
        Fetch a single protein sequence from UniProt
        
        Args:
            uniprot_id: UniProt accession ID (e.g., "P00698")
            
        Returns:
            Protein sequence string, or None if not found
        """
        # Clean up UniProt ID (remove version numbers, etc.)
        uniprot_id = uniprot_id.strip().split('.')[0].split('-')[0]
        
        # Check cache first
        if uniprot_id in self.sequence_cache:
            return self.sequence_cache[uniprot_id]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{uniprot_id}.fasta"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    lines = f.readlines()
                    # FASTA format: first line is header, rest is sequence
                    sequence = ''.join(line.strip() for line in lines[1:])
                    self.sequence_cache[uniprot_id] = sequence
                    return sequence
            except Exception as e:
                self.logger.debug(f"Could not read cached file {cache_file}: {e}")
        
        # Fetch from UniProt API
        self._rate_limit()
        
        url = UNIPROT_SEQUENCE_URL.format(uniprot_id=uniprot_id)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Parse FASTA format
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        # Skip header line, join sequence lines
                        sequence = ''.join(line.strip() for line in lines[1:])
                        # Cache it
                        self.sequence_cache[uniprot_id] = sequence
                        # Save to disk cache
                        try:
                            with open(cache_file, 'w') as f:
                                f.write(response.text)
                        except:
                            pass
                        return sequence
                    else:
                        self.logger.warning(f"Empty response for {uniprot_id}")
                        return None
                elif response.status_code == 404:
                    self.logger.debug(f"UniProt ID not found: {uniprot_id}")
                    # Cache None to avoid repeated failed requests
                    self.sequence_cache[uniprot_id] = None
                    return None
                else:
                    self.logger.warning(f"UniProt API error {response.status_code} for {uniprot_id}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request error for {uniprot_id} (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # All retries failed
        self.sequence_cache[uniprot_id] = None
        return None
    
    def fetch_sequences_batch(
        self,
        uniprot_ids: pd.Series,
        progress_callback: Optional[callable] = None
    ) -> pd.Series:
        """
        Fetch sequences for multiple UniProt IDs
        
        Args:
            uniprot_ids: Series of UniProt IDs
            progress_callback: Optional callback function(completed, total, current_id)
            
        Returns:
            Series of sequences (same index as input)
        """
        sequences = pd.Series(index=uniprot_ids.index, dtype=object)
        unique_ids = uniprot_ids.dropna().unique()
        
        self.logger.info(f"Fetching {len(unique_ids)} unique UniProt sequences...")
        
        for idx, uniprot_id in enumerate(unique_ids):
            if pd.isna(uniprot_id) or uniprot_id == '':
                continue
            
            sequence = self.fetch_sequence(str(uniprot_id))
            
            # Assign to all rows with this UniProt ID
            mask = uniprot_ids == uniprot_id
            sequences[mask] = sequence
            
            if progress_callback:
                progress_callback(idx + 1, len(unique_ids), str(uniprot_id))
            
            # Save cache periodically
            if (idx + 1) % 100 == 0:
                self._save_cache()
                self.logger.info(f"Fetched {idx + 1}/{len(unique_ids)} sequences, cached {len([s for s in self.sequence_cache.values() if s is not None])} sequences")
        
        # Final cache save
        self._save_cache()
        
        self.logger.info(f"Sequence fetching complete. {sequences.notna().sum()} sequences retrieved out of {len(sequences)} rows")
        
        return sequences
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached sequences"""
        total_cached = len(self.sequence_cache)
        valid_sequences = len([s for s in self.sequence_cache.values() if s is not None])
        return {
            'total_cached': total_cached,
            'valid_sequences': valid_sequences,
            'failed_fetches': total_cached - valid_sequences
        }


def fetch_sequences_for_fireprot(
    csv_path: str,
    output_csv_path: Optional[str] = None,
    cache_dir: str = "./uniprot_cache",
    chunk_size: int = 10000,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch sequences for FireProtDB entries and create enriched dataset
    
    Args:
        csv_path: Path to fireprotdb.csv
        output_csv_path: Path to save enriched CSV (if None, adds _with_sequences suffix)
        cache_dir: Directory for UniProt cache
        chunk_size: Process in chunks of this size
        max_rows: Maximum rows to process (None for all)
        
    Returns:
        DataFrame with added 'sequence' column
    """
    fetcher = UniProtSequenceFetcher(cache_dir=cache_dir)
    logger = logging.getLogger(__name__)
    
    if output_csv_path is None:
        output_csv_path = csv_path.replace('.csv', '_with_sequences.csv')
    
    output_path = Path(output_csv_path)
    
    # Check if output already exists
    if output_path.exists():
        logger.info(f"Output file {output_path} already exists. Loading...")
        return pd.read_csv(output_path, low_memory=False)
    
    logger.info(f"Processing {csv_path} to fetch sequences...")
    
    chunks_with_sequences = []
    total_processed = 0
    
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        if max_rows and total_processed >= max_rows:
            break
        
        logger.info(f"Processing chunk {chunk_idx + 1} ({len(chunk)} rows)...")
        
        # Check if UNIPROTKB column exists
        if 'UNIPROTKB' not in chunk.columns:
            logger.warning("UNIPROTKB column not found. Cannot fetch sequences.")
            break
        
        # Fetch sequences for this chunk
        sequences = fetcher.fetch_sequences_batch(
            chunk['UNIPROTKB'],
            progress_callback=lambda completed, total, current_id: None
        )
        
        # Add sequence column
        chunk['sequence'] = sequences
        
        # Keep only rows with valid sequences and DDG values
        chunk = chunk[chunk['sequence'].notna() & chunk['DDG'].notna()].copy()
        
        if len(chunk) > 0:
            chunks_with_sequences.append(chunk)
            total_processed += len(chunk)
            logger.info(f"  Chunk {chunk_idx + 1}: {len(chunk)} rows with sequences and DDG")
        
        # Save progress periodically
        if (chunk_idx + 1) % 10 == 0:
            if chunks_with_sequences:
                temp_df = pd.concat(chunks_with_sequences, ignore_index=True)
                temp_df.to_csv(output_path, index=False)
                logger.info(f"  Saved progress: {len(temp_df)} rows with sequences")
    
    if not chunks_with_sequences:
        logger.error("No rows with sequences found!")
        return pd.DataFrame()
    
    # Combine all chunks
    df_enriched = pd.concat(chunks_with_sequences, ignore_index=True)
    
    # Save final result
    df_enriched.to_csv(output_path, index=False)
    logger.info(f"Saved enriched dataset to {output_path}")
    logger.info(f"Total rows with sequences: {len(df_enriched)}")
    
    # Cache stats
    cache_stats = fetcher.get_cache_stats()
    logger.info(f"UniProt cache stats: {cache_stats}")
    
    return df_enriched


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch UniProt sequences for FireProtDB")
    parser.add_argument("--csv", type=str, default="fireprotdb.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--cache-dir", type=str, default="./uniprot_cache", help="Cache directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum rows to process")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    fetch_sequences_for_fireprot(
        args.csv,
        args.output,
        args.cache_dir,
        max_rows=args.max_rows
    )

