"""
Efficient batched loading of large parquet files for redshift estimation.

This module handles loading 100k+ source parquet files in batches with:
1. COLUMN FILTERING - primary memory optimization (loads only 10 of 50+ columns)
2. Row-based batching - manages workflow by loading sources in chunks
3. Support for subsetting and filtering

MEMORY OPTIMIZATION EXPLAINED:
==============================
For large L3 parquet files (e.g., 45GB, 865k sources, 50+ columns):

WITHOUT column filtering:
  - Loading 1000 sources with ALL columns: ~1-5 GB RAM
  - Loading 10k sources: ~10-50 GB RAM (often exceeds GPU memory!)

WITH column filtering (load only 10 required columns):
  - Loading 1000 sources: ~50-200 MB RAM 
  - Loading 10k sources: ~500 MB - 2 GB RAM (fits comfortably!)
  - Memory reduction: 70-95%!

Key insight: Most columns in L3 files are not needed for redshift estimation
(e.g., photometry in unused bands, quality flags, extended metadata, etc.)

ROW STREAMING vs COLUMN FILTERING:
===================================
- Row streaming: Only effective when file has many small row groups
- Column filtering: Always effective, especially for wide tables
- For single-row-group files (common in L3): column filtering is THE solution

Usage:
    # Load with column filtering (recommended for large files)
    loader = ParquetBatchLoader(
        parquet_file,
        batch_size=1000,
        columns=['SPHERExRefID', 'flux_dered_fiducial', 'flux_err_dered_fiducial', 'ra', 'dec']
    )
    
    for batch_data, start_idx, end_idx in loader:
        # Process batch_data with your pipeline
        ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import sys


class ParquetBatchLoader:
    """
    Memory-efficient batch loader for large parquet files using column filtering.
    
    PRIMARY OPTIMIZATION: Loads only specified columns, not entire table.
    This is critical for L3 parquet files with 50+ columns where you only need ~10.
    
    Parameters
    ----------
    parquet_file : str or Path
        Path to parquet file
    batch_size : int
        Number of sources to process per batch (default: 10000)
        For 10 columns × 306 bands × float32: ~1k sources ≈ 12 MB
    columns : list of str, optional
        Specific columns to load (HIGHLY RECOMMENDED for memory efficiency!)
        If None, loads ALL columns (not recommended for large files)
        Example: ['SPHERExRefID', 'flux_dered_fiducial', 'flux_err_dered_fiducial']
    filter_fn : callable, optional
        Function to filter sources: filter_fn(df) -> boolean mask
        Applied after loading each batch
    
    Examples
    --------
    >>> # Memory-efficient: load only 10 columns from file with 50+ columns
    >>> loader = ParquetBatchLoader(
    ...     'large_file.parquet',
    ...     batch_size=1000,
    ...     columns=['SPHERExRefID', 'flux_dered_fiducial', 'flux_err_dered_fiducial']
    ... )
    >>> 
    >>> for batch_df, start_idx, end_idx in loader:
    ...     print(f"Processing sources {start_idx}-{end_idx}")
    ...     # batch_df contains only 3 columns, not 50+!
    """
    
    def __init__(
        self,
        parquet_file: str,
        batch_size: int = 10000,
        columns: Optional[List[str]] = None,
        filter_fn: Optional[callable] = None,
        row_offset: Optional[int] = None,
        num_rows: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        row_offset : int, optional
            Global start row in the parquet file for this task's slice.
            When provided together with num_rows, the entire task slice is
            read ONCE at init time and cached in memory.  Subsequent batch
            iterations slice the in-memory DataFrame — zero additional I/O
            per batch, no concurrent Lustre contention across job-array tasks.
        num_rows : int, optional
            Number of rows to read starting from row_offset.
            Automatically clamped to the file boundary.
        """
        self.parquet_file = Path(parquet_file)
        self.batch_size = batch_size
        self.columns = columns
        self.filter_fn = filter_fn
        self.row_offset = row_offset if row_offset is not None else 0

        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.parquet_file)
        file_total = pf.metadata.num_rows
        n_row_groups = pf.num_row_groups

        if row_offset is not None:
            # ── FAST PATH ────────────────────────────────────────────────────
            # Read only this task's row range once into memory.
            # 22k sources × 306 bands × float32 ≈ 27 MB — trivially fits in RAM.
            # Eliminates per-batch O(N) re-scanning and concurrent Lustre hammering.
            effective_num_rows = min(
                num_rows if num_rows is not None else (file_total - row_offset),
                file_total - row_offset
            )
            print(f"ParquetBatchLoader: reading task slice "
                  f"[{row_offset:,}:{row_offset + effective_num_rows:,}] "
                  f"({effective_num_rows:,} rows) from {self.parquet_file.name} ...", flush=True)
            table = pq.read_table(str(self.parquet_file), columns=self.columns)
            table_slice = table.slice(row_offset, effective_num_rows)
            del table
            self._task_df_raw = table_slice.to_pandas()
            del table_slice
            self.total_sources = len(self._task_df_raw)   # == effective_num_rows
            self.n_batches = (self.total_sources + batch_size - 1) // batch_size
            self.dataset = None
            print(f"  Cached {self.total_sources:,} rows in memory. "
                  f"{self.n_batches} batches of {batch_size:,}.", flush=True)
            if self.columns is not None:
                print(f"  Columns loaded: {len(self.columns)}", flush=True)
        else:
            # ── LEGACY PATH ──────────────────────────────────────────────────
            # Iterate over the full file; caller is responsible for skipping
            # batches outside their assigned source range.
            self._task_df_raw = None
            self.total_sources = file_total
            self.n_batches = (self.total_sources + batch_size - 1) // batch_size
            avg_rows_per_rg = file_total / n_row_groups if n_row_groups > 0 else 0
            print(f"ParquetBatchLoader initialized (legacy full-file mode):")
            print(f"  File: {self.parquet_file.name}")
            print(f"  Total sources: {self.total_sources:,}")
            print(f"  Batch size: {batch_size:,}")
            print(f"  Number of batches: {self.n_batches}")
            print(f"  Row groups: {n_row_groups} (avg {avg_rows_per_rg:,.0f} rows/group)")
            if n_row_groups == 1 and self.total_sources > 100000:
                print(f"  ⚠ WARNING: single row group — use row_offset/num_rows for task slicing")
            import pyarrow.dataset as ds
            self.dataset = ds.dataset(str(self.parquet_file), format="parquet")
            if self.columns is not None:
                print(f"  Column filter: {len(self.columns)} columns")
            else:
                print(f"  No column filter - loading ALL columns")

        self._current_batch = 0
    
    def __iter__(self):
        """Iterate over batches."""
        self._current_batch = 0
        return self
    
    def __next__(self):
        """Get next batch.  Returns (df_batch, global_start_idx, global_end_idx)."""
        if self._current_batch >= self.n_batches:
            raise StopIteration

        local_start = self._current_batch * self.batch_size
        local_end   = min(local_start + self.batch_size, self.total_sources)

        if self._task_df_raw is not None:
            # ── FAST PATH: slice the cached in-memory DataFrame ───────────────
            df_batch = self._task_df_raw.iloc[local_start:local_end].copy()
            # Return global parquet row indices so callers and file names stay
            # consistent with the full-file addressing scheme.
            global_start = self.row_offset + local_start
            global_end   = self.row_offset + local_end
        else:
            # ── LEGACY PATH: scan from row 0 on every call ────────────────────
            global_start = local_start
            global_end   = local_end
            n_rows = local_end - local_start
            try:
                scanner = self.dataset.scanner(
                    columns=self.columns,
                    batch_size=self.batch_size
                )
                rows_seen = 0
                df_batch = None
                for record_batch in scanner.to_batches():
                    batch_start = rows_seen
                    batch_end   = rows_seen + record_batch.num_rows
                    if batch_end > global_start and batch_start < global_end:
                        ls = max(0, global_start - batch_start)
                        le = min(record_batch.num_rows, global_end - batch_start)
                        sliced = record_batch.slice(ls, le - ls).to_pandas()
                        df_batch = sliced if df_batch is None else pd.concat(
                            [df_batch, sliced], ignore_index=True)
                        del sliced
                        if len(df_batch) >= n_rows:
                            del record_batch
                            break
                    rows_seen = batch_end
                    del record_batch
                    if rows_seen >= global_end:
                        break
                if df_batch is None or len(df_batch) == 0:
                    print(f"\nWarning: no data found at indices {global_start}-{global_end}")
                    df_batch = pd.DataFrame()
            except Exception as e:
                print(f"\nWarning: scanner failed: {e}; falling back to direct read")
                import pyarrow.parquet as pq
                table = pq.read_table(str(self.parquet_file), columns=self.columns)
                df_batch = table.slice(global_start, n_rows).to_pandas()
                del table

        # Apply row filter (e.g. drop NaN z_specz)
        if self.filter_fn is not None and len(df_batch) > 0:
            mask = self.filter_fn(df_batch)
            df_batch = df_batch[mask].copy()

        self._current_batch += 1
        return df_batch, global_start, global_end
    
    def __len__(self):
        """Number of batches."""
        return self.n_batches
    
    def load_batch(self, batch_idx: int) -> Tuple[pd.DataFrame, int, int]:
        """
        Load a specific batch by index.
        
        Parameters
        ----------
        batch_idx : int
            Batch index (0-based)
        
        Returns
        -------
        df_batch : DataFrame
            Batch data
        start_idx : int
            Start index in full dataset
        end_idx : int
            End index in full dataset
        """
        if batch_idx < 0 or batch_idx >= self.n_batches:
            raise ValueError(f"Batch index {batch_idx} out of range [0, {self.n_batches})")
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_sources)
        
        df = pd.read_parquet(self.parquet_file, columns=self.columns)
        df_batch = df.iloc[start_idx:end_idx]
        
        if self.filter_fn is not None:
            mask = self.filter_fn(df_batch)
            df_batch = df_batch[mask]
        
        return df_batch, start_idx, end_idx
    
    def load_all(self) -> pd.DataFrame:
        """
        Load entire dataset (use only for small files or when memory allows).
        
        Returns
        -------
        df : DataFrame
            Full dataset
        """
        print(f"Loading all {self.total_sources:,} sources...")
        df = pd.read_parquet(self.parquet_file, columns=self.columns)
        
        if self.filter_fn is not None:
            mask = self.filter_fn(df)
            df = df[mask]
            print(f"After filtering: {len(df):,} sources")
        
        return df


def load_fiducial_fluxes_batch(
    parquet_file: str,
    filter_set_name: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    weight_soft: float = 5e-4,
    abs_norm: bool = True,
    max_normflux: float = 100,
    filter_fn: Optional[callable] = None,
    use_weighted_mean: bool = False,
):
    """
    Load fiducial fluxes from parquet file for a specific range of sources.
    
    This is designed for integration with redshift_job.py and similar pipelines.
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file (e.g., '/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet')
    filter_set_name : str
        Name of filter set (e.g., 'SPHEREx_filter_306')
    start_idx : int
        Starting source index (default: 0)
    end_idx : int, optional
        Ending source index (default: None = all)
    weight_soft : float
        Soft weighting parameter for normalization
    abs_norm : bool
        Whether to use absolute normalization
    max_normflux : float
        Maximum normalized flux value
    filter_fn : callable, optional
        Function to filter sources: filter_fn(df) -> boolean mask
        Example: lambda df: (df['z_specz'] > 0) & (df['z_specz'] < 2.0)
    
    Returns
    -------
    dat_obs : spec_data_jax
        Observed data object with loaded photometry
    property_cat_df : DataFrame
        Property catalog with source IDs, redshifts, etc.
    central_wavelengths : ndarray
        Central wavelengths for rest frame (linearly spaced)
    wave_obs : ndarray
        Observed wavelengths from filters
    
    Example
    -------
    # Load first 1000 sources with spec-z > 0
    dat, props, _, wave = load_fiducial_fluxes_batch(
        '/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet',
        'SPHEREx_filter_306',
        start_idx=0,
        end_idx=1000,
        filter_fn=lambda df: df['z_specz'] > 0
    )
    """
    from models.pae_jax import load_filter_central_wavelengths
    from data_proc.dataloader_jax import load_real_spherex_parquet
    
    # Load filter central wavelengths
    wave_obs, _ = load_filter_central_wavelengths(filter_set_name, filtfiles=None)
    nbands = len(wave_obs)
    
    print(f"\nLoading sources {start_idx} to {end_idx or 'end'} from {Path(parquet_file).name}")
    print(f"Filter set: {filter_set_name} ({nbands} bands)")
    
    # Load parquet (slice if needed)
    df = pd.read_parquet(parquet_file)
    
    if end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]
    
    # Apply filter
    if filter_fn is not None:
        mask = filter_fn(df)
        n_before = len(df)
        df = df[mask]
        print(f"Filter applied: {len(df):,}/{n_before:,} sources retained")
    
    if len(df) == 0:
        raise ValueError("No sources remaining after filtering!")
    
    # Use existing loading function
    dat_obs, property_cat_df = load_real_spherex_parquet(
        parquet_file=None,  # Not used since we pass df directly
        filter_set_name=filter_set_name,
        wave_obs=wave_obs,
        weight_soft=weight_soft,
        abs_norm=abs_norm,
        max_normflux=max_normflux,
        df=df,  # Pass pre-loaded dataframe
        use_weighted_mean=use_weighted_mean
    )
    
    # Rest frame wavelengths
    central_wavelengths = np.linspace(0.1, 5, 500)
    
    return dat_obs, property_cat_df, central_wavelengths, wave_obs


# Predefined filter functions
def filter_has_specz(df: pd.DataFrame) -> np.ndarray:
    """Filter sources with spectroscopic redshift."""
    return df['z_specz'].notna() & (df['z_specz'] > 0)


def filter_redshift_range(z_min: float, z_max: float):
    """Create filter for redshift range (uses spec-z if available, else photo-z)."""
    def _filter(df: pd.DataFrame) -> np.ndarray:
        z = df['z_specz'].fillna(df['z_best_gals'])
        return (z >= z_min) & (z <= z_max)
    return _filter


def filter_good_fits(chi2_red_max: float = 3.0):
    """Filter sources with good SED fits."""
    def _filter(df: pd.DataFrame) -> np.ndarray:
        chi2_red = df['minchi2_minchi2_gals'] / (df['Nsamples'] - 1)
        return chi2_red < chi2_red_max
    return _filter


def combine_filters(*filters):
    """Combine multiple filter functions with AND logic."""
    def _combined_filter(df: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        for filt in filters:
            mask &= filt(df)
        return mask
    return _combined_filter


if __name__ == '__main__':
    # Example usage
    parquet_file = '/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet'
    
    print("Example 1: Iterate over batches")
    print("=" * 60)
    loader = ParquetBatchLoader(parquet_file, batch_size=10000)
    for batch_df, start, end in loader:
        print(f"Batch {start:,}-{end:,}: {len(batch_df):,} sources")
        if start >= 20000:  # Just show first few
            break
    
    print("\n" + "=" * 60)
    print("Example 2: Load specific batch with filtering")
    print("=" * 60)
    # Only sources with spec-z and good fits
    filter_func = combine_filters(
        filter_has_specz,
        filter_good_fits(chi2_red_max=2.0),
        filter_redshift_range(0.5, 1.5)
    )
    
    loader_filtered = ParquetBatchLoader(
        parquet_file, 
        batch_size=5000,
        filter_fn=filter_func
    )
    
    batch_df, start, end = loader_filtered.load_batch(0)
    print(f"Loaded batch 0: {len(batch_df):,} sources after filtering")
    print(f"Redshift range: {batch_df['z_specz'].min():.3f} - {batch_df['z_specz'].max():.3f}")
