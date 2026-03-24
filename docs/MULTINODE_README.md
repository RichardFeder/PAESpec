# Multi-Node PAE Redshift Estimation

This directory contains scripts for running PAE redshift estimation across multiple GPU nodes at NERSC.

## Files

- **`redshift_job_batched.py`** - Core processing script with multi-node support
- **`submit_multinode_job.sh`** - SLURM job array template (single file, multiple nodes)
- **`submit_multifile_jobs.py`** - Python helper for multi-file submissions

## Usage Scenarios

### 1. Single File, Single Node (baseline)

Process one parquet file on one GPU node:

```bash
python scripts/redshift_job_batched.py \
  --parquet-file /pscratch/sd/r/rmfeder/data/selection_wf.parquet \
  --batch-size 800 \
  --sampling-batch-size 200 \
  --filter-specz \
  --z-max 2.0 \
  --datestr test_$(date +%m%d%y) \
  --use-multicore \
  --n-devices 4
```

### 2. Single Large File, Multiple Nodes (job array)

Split one large file across N nodes using SLURM job array:

#### Method A: Edit and submit template script

1. Edit `scripts/submit_multinode_job.sh`:
   - Set `PARQUET_FILE`, `SOURCES_PER_TASK`, etc.
   - Calculate `N_TASKS = ceil(total_sources / SOURCES_PER_TASK)`
   - Set `#SBATCH --array=0-(N_TASKS-1)%max_concurrent`

2. Submit:
   ```bash
   sbatch scripts/submit_multinode_job.sh
   ```

#### Method B: Use Python helper (recommended)

```bash
python scripts/submit_multifile_jobs.py \
  --files /pscratch/sd/r/rmfeder/data/selection_wf.parquet \
  --split-into 10 \
  --sources-per-task 10000 \
  --filter-specz \
  --z-max 2.0
```

This will:
- Count sources in the file
- Generate job array script splitting work into 10 chunks
- Submit the job array

### 3. Multiple Files, One Node Each

Process multiple parquet files (one job per file):

```bash
python scripts/submit_multifile_jobs.py \
  --files file1.parquet file2.parquet file3.parquet \
  --filter-specz \
  --z-max 2.0
```

### 4. Multiple Files, Each Split Across Nodes

Process multiple files, each split across nodes:

```bash
python scripts/submit_multifile_jobs.py \
  --files file1.parquet file2.parquet \
  --split-into 5 \
  --max-concurrent 3 \
  --filter-specz
```

This creates 2 job arrays (one per file), each with 5 tasks, max 3 running concurrently.

## How It Works

### Source Range Splitting

The `redshift_job_batched.py` script now accepts:

- `--start-source N` / `--end-source M` - Process sources [N, M)
- `--job-array-task-id ID` - Auto-compute range from task ID
- `--sources-per-task N` - Sources per task (used with job-array-task-id)

Example:
```bash
# Task 0 processes sources 0-10000
# Task 1 processes sources 10000-20000
# etc.
python scripts/redshift_job_batched.py \
  --parquet-file data.parquet \
  --job-array-task-id ${SLURM_ARRAY_TASK_ID} \
  --sources-per-task 10000
```

### Job Array Indexing

SLURM job arrays use `SLURM_ARRAY_TASK_ID` (0, 1, 2, ...) to identify each task.
Each task:
1. Calculates its source range: `[task_id * sources_per_task, (task_id+1) * sources_per_task)`
2. Loads and processes only that chunk
3. Saves results with task ID in filename: `PAE_results_batch0_start10000_...npz`

### Result Files

Each task creates separate result files:
```
PAE_results_batch0_start0_taskdate.npz
PAE_results_batch0_start10000_taskdate.npz
PAE_results_batch0_start20000_taskdate.npz
...
```

Collate them afterward (see below).

## Monitoring Jobs

```bash
# View job array status
squeue -u $USER --array

# View specific job
squeue -j JOBID

# View all tasks of a job array
squeue -j JOBID --array

# Cancel entire job array
scancel JOBID

# Cancel specific tasks
scancel JOBID_2,JOBID_5

# View completed job info
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

## Collating Results

After all tasks complete, combine per-task result files:

### Method 1: Manual collation

```python
import numpy as np
from pathlib import Path

result_dir = Path('/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/multinode_123025')
pattern = 'PAE_results_*.npz'
files = sorted(result_dir.glob(pattern))

all_dicts = [dict(np.load(f, allow_pickle=True)) for f in files]
keys = set().union(*[d.keys() for d in all_dicts])

merged = {}
for k in keys:
    parts = [d[k] for d in all_dicts if k in d]
    try:
        merged[k] = np.concatenate(parts, axis=0)
    except:
        merged[k] = np.array(parts, dtype=object)

np.savez_compressed(result_dir / 'PAE_results_combined.npz', **merged)
```

### Method 2: Use built-in collation

Add `--collate-results` flag to the Python script (but note: this currently only collates results **within one task**, not across tasks).

For cross-task collation, use Method 1 above.

## Resource Planning

### Sizing job arrays

Given:
- Total sources: `N_total`
- Sources per GPU node: `N_per_node` (recommended: 5,000-20,000)
- GPU nodes needed: `N_nodes = ceil(N_total / N_per_node)`

Example:
- `selection_wf.parquet`: ~80,000 sources (after filtering)
- 10,000 sources/node
- → 8 nodes needed
- → `--array=0-7%4` (8 tasks, max 4 concurrent)

### Walltime estimation

Based on single-node runs:
- ~0.15s per galaxy per batch (800 sources)
- 10,000 sources ≈ 25 minutes
- Add 20% buffer for I/O and startup
- → Request `--time=00:30:00` for 10k sources/node

### Cost estimation

GPU node-hours:
- 1 node × 0.5 hours = 0.5 node-hours per 10k sources
- 80k sources = 4 node-hours total
- At NERSC Perlmutter GPU rates: ~$3/node-hour
- → ~$12 for 80k sources (8 nodes × 0.5 hours)

## Troubleshooting

### "No such file or directory: /pscratch/..."
- Check paths are absolute
- Verify files exist from compute node: `srun -C gpu -t 10 ls -lh /pscratch/...`

### Memory issues
- Reduce `--batch-size` (outer batch)
- Reduce `--sampling-batch-size` (inner batch, sources per GPU)
- Check current implementation includes all memory fixes

### Job array task fails
- Check logs in `logs/multinode_JOBID_TASKID.{out,err}`
- Resubmit failed tasks: `sbatch --array=3,7,9 submit_multinode_job.sh`

### Duplicate source processing
- Ensure `--start-source` / `--end-source` ranges don't overlap
- Verify `sources_per_task` calculation is correct

### Missing results files
- Check output directory permissions (should be 700/owner-only)
- Verify job completed successfully (check logs)
- Look for Python errors in stderr logs

## Examples

### Quick test (dry run)

```bash
python scripts/submit_multifile_jobs.py \
  --files /pscratch/sd/r/rmfeder/data/selection_wf.parquet \
  --split-into 2 \
  --dry-run
# Check generated scripts in job_scripts/ directory
```

### Production run (100k sources, 10 nodes)

```bash
python scripts/submit_multifile_jobs.py \
  --files /pscratch/sd/r/rmfeder/data/selection_wf.parquet \
  --split-into 10 \
  --sources-per-task 10000 \
  --max-concurrent 5 \
  --filter-specz \
  --z-max 2.0 \
  --time 01:00:00
```

### Process deep field + wide field (separate jobs)

```bash
python scripts/submit_multifile_jobs.py \
  --files \
    /pscratch/sd/r/rmfeder/data/selection_df.parquet \
    /pscratch/sd/r/rmfeder/data/selection_wf.parquet \
  --split-into 5 \
  --filter-specz \
  --z-max 2.0
```

## Best Practices

1. **Always test with `--dry-run` first** to verify scripts are generated correctly
2. **Use `--max-concurrent` to limit concurrent nodes** (avoid overwhelming scheduler)
3. **Start with small `--split-into` values** (2-3 nodes) to test end-to-end
4. **Check one task's output before submitting many** (verify file paths, permissions, etc.)
5. **Set conservative walltime estimates** (add 30% buffer to measured times)
6. **Monitor jobs actively during first hour** (catch configuration errors early)
7. **Keep job array size ≤ 50 tasks** (scheduler-friendly, easier to manage)
8. **Name jobs clearly** (helps when viewing queue: `pae_wf`, `pae_df`, etc.)
