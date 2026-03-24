#!/bin/bash
# Run PAE batched redshift estimation on 4 GPUs in production mode
# Optionally generate summary plots after completion

set -e  # Exit on error

# ============================================================
# CONFIGURATION
# ============================================================
# Set to "true" to automatically collate multi-task results
# Set to "false" to skip collation (can run manually later)
COLLATE_RESULTS="true"
OVERWRITE_COLLATED_RES="true"

# Set to "true" to automatically generate plots after redshift estimation
# Set to "false" to skip plotting (can run manually with run_plot_afterburner.sh)
GENERATE_PLOTS="true"
GENERATE_SOURCE_PLOTS="false"
N_GOOD_SOURCES=10
N_BAD_SOURCES=10

# Set to "true" to skip redshift estimation and only run plotting
# (Useful for re-running plots on existing results)
SKIP_REDSHIFT_RUN="true"

# Set to "true" to resume a prematurely terminated run
# Will detect existing batch files and continue from where it left off
RESUME_RUN="false"

# Set to "true" to fix redshift to spec-z values (requires spec-z in parquet)
FIX_REDSHIFT="false"

# Set to "true" to disable absolute normalization (use raw fluxes/errors)
# Relies on amplitude marginalization to handle flux scaling
NO_ABS_NORM="true"

# Set to "true" to use inverse-variance weighted mean for normalization
# More robust for noisy data, downweights channels with high uncertainties
USE_WEIGHTED_MEAN="false"

# Set to integer for reproducible runs, or leave empty for random initialization
# To test convergence, run twice with different seeds (e.g., 42 and 123)
RANDOM_SEED="0"  # Empty = random, or set to number like "42"

# ============================================================
# CHANNEL MASKING
# ============================================================
# Exclude SPHEREx band 4 (channels with 2.30 < lambda < 3.82 micron).
# Lower boundary extended from 2.42 to 2.30 to also mask the CO bandhead region
# (adds ch 44 ~2.327 µm and ch 46 ~2.380 µm to the masked set).
# Set to "true" to exclude these channels from the likelihood.
MASK_BAND4="true"

# Optionally specify custom wavelength exclusion ranges in addition to (or instead of)
# MASK_BAND4. Format: space-separated list of "lmin,lmax" pairs (micron).
# Example: MASK_WAVELENGTH_RANGES="2.42,3.82 4.5,5.0"
# He I 10830 (1.083 µm): mask ch 11 (1.067 µm) and ch 15 (1.092 µm).
MASK_WAVELENGTH_RANGES="1.06,1.10"

# Baseline run to compare against when generating plots.
# Comparison figures (CDF, metric bars, bias histograms) are saved under
# <figures_dir>/<DATESTR>/comparison/. Set empty to skip the comparison step.
COMPARISON_DATESTR="multinode_validation_run_022126"
# COMPARISON_DATESTR="validation_set_865k_010826_no_band4"


# Set to "true" to force-recompute the broadband SNR cache from parquet before
# running plots, even if a cache file already exists.  Useful when the existing
# cache was built on a subset of sources and the combined result file has grown.
# The old cache is deleted so generate_redshift_plots.py triggers a fresh build.
RECOMPUTE_SNR_CACHE="true"

# Set datestr (used for both redshift estimation and plot directory)
# DATESTR="multicore_test_16k_wf_123025"
# DATESTR="multicore_8k_wf_8cps_v0_123125"
# DATESTR="test_serial_verbose_010226_v2"
# DATESTR="test_df_32k_010326"
# DATESTR="test_wf_32k_010526_nfalpha=1"
# DATESTR="test_wf_16k_010326_nfalpha=1"
# DATESTR="test_wf_32k_010526_nfalpha=1_noZprior"
# DATESTR="test_wf_32k_010726_nfalpha=1_withZprior_meannorm_randomseed${RANDOM_SEED}"
# DATESTR="test_wf_32k_010726_nfalpha=1_withZprior_noabsnorm_randomseed${RANDOM_SEED}"
# DATESTR="test_865k_validation_mediumtest_$(date +%m%d%y)"
# DATESTR="test_865k_validation_minitest2_$(date +%m%d%y)"
# DATESTR="single_test_170826"
# DATESTR="multinode_validation_run_022126"

DATESTR="multinode_validation_run_022126"
# DATESTR="validation_set_865k_010826_no_band4_dichroic_HeI_mask"
# DATESTR="debug_no_band4"


# DATESTR="test_wf_32k_010726_nfalpha=1_withZprior_meannorm"

# ============================================================
# SAVE RUN METADATA
# ============================================================
save_run_metadata() {
    local results_dir="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
    
    # Create results directory if it doesn't exist
    mkdir -p "$results_dir"
    
    # Save script configuration
    cat > "$results_dir/run_script_config.txt" << EOF
# Run Script Configuration - $(date)
# Script: $(readlink -f "$0")
# Working Directory: $(pwd)
# User: $(whoami)
# Host: $(hostname)
# Command: $0 $@

# Configuration Variables:
DATESTR="$DATESTR"
COLLATE_RESULTS="$COLLATE_RESULTS"
GENERATE_PLOTS="$GENERATE_PLOTS"
GENERATE_SOURCE_PLOTS="$GENERATE_SOURCE_PLOTS"
N_GOOD_SOURCES=$N_GOOD_SOURCES
N_BAD_SOURCES=$N_BAD_SOURCES
SKIP_REDSHIFT_RUN="$SKIP_REDSHIFT_RUN"
RESUME_RUN="$RESUME_RUN"
FIX_REDSHIFT="$FIX_REDSHIFT"
USE_WEIGHTED_MEAN="$USE_WEIGHTED_MEAN"
NO_ABS_NORM="$NO_ABS_NORM"
RANDOM_SEED="$RANDOM_SEED"
MASK_BAND4="$MASK_BAND4"
MASK_WAVELENGTH_RANGES="$MASK_WAVELENGTH_RANGES"
COMPARISON_DATESTR="$COMPARISON_DATESTR"

# Environment:
CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_JOB_NAME=$SLURM_JOB_NAME
SLURM_NODELIST=$SLURM_NODELIST

EOF

    # Save copy of entire script for full traceability
    cp "$(readlink -f "$0")" "$results_dir/run_script_copy.sh"
    
    echo "Script metadata saved to: $results_dir/run_script_config.txt"
    echo "Script copy saved to: $results_dir/run_script_copy.sh"
}

# ============================================================
# RUN REDSHIFT ESTIMATION
# ============================================================

# Save run metadata before starting
save_run_metadata

# ============================================================
# SET UP LOGGING
# ============================================================
# Create log file in results directory to capture all output
RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
LOG_FILE="$RESULTS_DIR/run_log_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================"
echo "LOGGING SETUP"
echo "============================================================"
echo "All output will be logged to: $LOG_FILE"
echo "Log includes stdout and stderr from all pipeline steps"
echo "============================================================"
echo ""

# Disable Python output buffering to ensure print() statements appear immediately
export PYTHONUNBUFFERED=1

# Start logging - redirect all subsequent output to both terminal and log file
# Use process substitution to tee both stdout and stderr
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Logging started (Python unbuffered mode enabled)"
echo ""

if [ "$SKIP_REDSHIFT_RUN" = "true" ]; then
    echo "============================================================"
    echo "SKIPPING REDSHIFT ESTIMATION (SKIP_REDSHIFT_RUN=true)"
    echo "============================================================"
    echo "Using existing results for: $DATESTR"
    echo ""
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUNNING REDSHIFT ESTIMATION"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================================"
    echo "Date string: $DATESTR"
    echo "Generate plots after: $GENERATE_PLOTS"
    echo "Fix redshift: $FIX_REDSHIFT"
    echo "Log file: $LOG_FILE"
    echo ""

    # Uncomment one of the following configurations:

# Configuration 1: Full production run
# python scripts/redshift_job_batched.py \
#   --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
#   --datestr production_run_122925 \
#   --batch-size 5000 \
#   --sampling-batch-size 200 \
#   --use-multicore \
#   --n-devices 4 \
#   --collate-results \
#   --filter-set SPHEREx_filter_306

# Configuration 2: Mini test run
# python scripts/redshift_job_batched.py \
#   --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
#   --batch-size 40 \
#   --datestr mini_prod_run_122925 \
#   --sampling-batch-size 20 \
#   --max-batches 1 \
#   --collate-results \
#   --filter-set SPHEREx_filter_306

# Configuration 3: Medium test with spec-z filter (CURRENTLY ACTIVE)
# Build command with optional --fix-z flag
FIX_Z_FLAG=""
if [ "$FIX_REDSHIFT" = "true" ]; then
    FIX_Z_FLAG="--fix-z"
fi

# python scripts/redshift_job_batched.py \
#   --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
#   --datestr $DATESTR \
#   --batch-size 800 \
#   --sampling-batch-size 400 \
#   --filter-specz \
#   --z-max 2.0 \
#   --max-batches 10 \
#   --use-multicore \
#   --n-devices 4 \
#   --init_reinit \
#   --collate-results \
#   --filter-set SPHEREx_filter_306 \
#   $FIX_Z_FLAG

# Test configuration: 1600 sources total, 800 per task (2 tasks)
# Sampling batch size of 200 per core (4 cores × 200 = 800 samples per batch in multicore)

# ============================================================
# RESUME LOGIC: Detect existing batches and calculate resume point
# ============================================================
START_TASK_ARG=""
if [ "$RESUME_RUN" = "true" ]; then
    RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
    
    if [ -d "$RESULTS_DIR" ]; then
        echo "============================================================"
        echo "RESUME MODE: Detecting existing batch files"
        echo "============================================================"
        
        # Count existing batch result files across all task directories
        N_EXISTING_BATCHES=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | wc -l)
        
        echo "Found $N_EXISTING_BATCHES existing batch files"
        
        if [ $N_EXISTING_BATCHES -gt 0 ]; then
            # Extract batch numbers and find the highest completed batch
            # Look for patterns like "batch0_start0", "batch1_start800", etc.
            HIGHEST_START=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | \
                sed -n 's/.*_start\([0-9]*\)_.*/\1/p' | \
                sort -n | tail -1)
            
            if [ -n "$HIGHEST_START" ]; then
                # Calculate task ID from source index
                # Task ID = source_index / sources_per_task
                SOURCES_PER_TASK=8000
                LAST_COMPLETED_TASK=$((HIGHEST_START / SOURCES_PER_TASK))
                RESUME_TASK=$((LAST_COMPLETED_TASK + 1))
                
                echo "Last completed batch started at source: $HIGHEST_START"
                echo "Last completed task ID: $LAST_COMPLETED_TASK"
                echo "Resuming from task ID: $RESUME_TASK"
                
                START_TASK_ARG="--start-task $RESUME_TASK"
                
                echo "============================================================"
            else
                echo "Warning: Could not parse batch file names to determine resume point"
                echo "Starting from beginning"
            fi
        else
            echo "No existing batch files found. Starting from beginning."
        fi
    else
        echo "Results directory does not exist. Starting fresh run."
    fi
    echo ""
fi

# Define the redshift estimation command
# NOTE: For large parquet files (40GB+), run_tile_serial_multicore.py now uses
# PyArrow metadata for source counting (doesn't load full file into memory)
REDSHIFT_COMMAND=(
    python scripts/run_tile_serial_multicore.py
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/full_validation_sz_0-1000.0_z_0-1000.0.parquet
    --datestr "$DATESTR"
    --sources-per-task 5000
    --max-tasks 4
    --batch-size 1000
    --no-robust-reinit
    --sampling-batch-size 1000
    --filter-set SPHEREx_filter_306
    --nf-alpha 1.0
)

# Add use-weighted-mean flag if enabled
if [ "$USE_WEIGHTED_MEAN" = "true" ]; then
    REDSHIFT_COMMAND+=(--use-weighted-mean)
fi

# Add no-abs-norm flag if enabled
if [ "$NO_ABS_NORM" = "true" ]; then
    REDSHIFT_COMMAND+=(--no-abs-norm)
fi

# Add random seed if specified
if [ -n "$RANDOM_SEED" ]; then
    REDSHIFT_COMMAND+=(--random-seed "$RANDOM_SEED")
fi

# Add channel masking flags if enabled.
# --mask-wavelength-ranges uses nargs='*', so ALL ranges must be passed as tokens
# after a SINGLE flag invocation.  Collect all ranges first, then emit one flag.
_ALL_MASK_RANGES=()
if [ "$MASK_BAND4" = "true" ]; then
    _ALL_MASK_RANGES+=(2.30,3.82)
fi
if [ -n "$MASK_WAVELENGTH_RANGES" ]; then
    read -ra _CUSTOM_RANGES <<< "$MASK_WAVELENGTH_RANGES"
    _ALL_MASK_RANGES+=("${_CUSTOM_RANGES[@]}")
fi
if [ "${#_ALL_MASK_RANGES[@]}" -gt 0 ]; then
    REDSHIFT_COMMAND+=(--mask-wavelength-ranges "${_ALL_MASK_RANGES[@]}")
fi

# Add resume parameter if set
if [ -n "$START_TASK_ARG" ]; then
    REDSHIFT_COMMAND+=($START_TASK_ARG)
fi

# Save command to metadata
echo "" >> "/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/run_script_config.txt"
echo "# Executed Command:" >> "/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/run_script_config.txt"
echo "${REDSHIFT_COMMAND[*]}" >> "/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/run_script_config.txt"
echo "" >> "/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/run_script_config.txt"

# Execute the command
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing: ${REDSHIFT_COMMAND[*]}"
echo ""
"${REDSHIFT_COMMAND[@]}"

# Capture exit code from redshift estimation
REDSHIFT_EXIT_CODE=$?
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Redshift estimation completed with exit code: $REDSHIFT_EXIT_CODE"

if [ $REDSHIFT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ Redshift estimation FAILED with exit code $REDSHIFT_EXIT_CODE"
    echo "Skipping plot generation."
    exit $REDSHIFT_EXIT_CODE
fi

echo ""
echo "✓ Redshift estimation completed successfully!"
echo ""
echo "Results saved to: /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"

fi  # End of SKIP_REDSHIFT_RUN check

# ============================================================
# COLLATE RESULTS (if enabled and multi-task run)
# ============================================================
COMBINED_FILE="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/PAE_results_combined_${DATESTR}.npz"

if [ "$SKIP_REDSHIFT_RUN" = "false" ] && [ "$COLLATE_RESULTS" = "true" ]; then
    # Just finished redshift run, collate results
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] COLLATING BATCH RESULTS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================================"
    
    # Count batch files before collation
    RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
    N_BATCH_FILES=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | wc -l)
    echo "Number of batch files to collate: $N_BATCH_FILES"
    
    python scripts/collate_batched_results.py "$DATESTR"
    
    COLLATE_EXIT_CODE=$?
    
    if [ $COLLATE_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ WARNING: Result collation failed with exit code $COLLATE_EXIT_CODE"
        echo "Individual batch files are still valid."
        echo "Skipping plot generation (requires combined file)."
        GENERATE_PLOTS="false"
        GENERATE_SOURCE_PLOTS="false"
    else
        echo ""
        echo "✓ Results collated successfully!"
        echo "Combined file: $COMBINED_FILE"
    fi
elif [ "$SKIP_REDSHIFT_RUN" = "true" ]; then
    # Skipped redshift run, check if we need combined file for plots
    if [ "$GENERATE_PLOTS" = "true" ] || [ "$GENERATE_SOURCE_PLOTS" = "true" ]; then
        if [ ! -f "$COMBINED_FILE" ] || [ "$OVERWRITE_COLLATED_RES" = "true" ]; then
            if [ "$COLLATE_RESULTS" = "true" ]; then
                # Combined file doesn't exist and user wants to collate
                echo ""
                echo "============================================================"
                echo "COLLATING BATCH RESULTS"
                echo "============================================================"
                echo "Combined file not found, running collation..."
                
                # Count batch files before collation
                RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
                N_BATCH_FILES=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | wc -l)
                echo "Number of batch files to collate: $N_BATCH_FILES"
                
                python scripts/collate_batched_results.py "$DATESTR"
                
                COLLATE_EXIT_CODE=$?
                if [ $COLLATE_EXIT_CODE -ne 0 ]; then
                    echo "⚠ WARNING: Result collation failed"
                    echo "Skipping plot generation."
                    GENERATE_PLOTS="false"
                    GENERATE_SOURCE_PLOTS="false"
                else
                    echo "✓ Results collated successfully!"
                    echo "Combined file: $COMBINED_FILE"
                fi
            else
                # User doesn't want to collate but combined file is missing
                echo ""
                echo "⚠ WARNING: Combined results file not found: $COMBINED_FILE"
                echo "Set COLLATE_RESULTS=\"true\" to generate it, or run manually:"
                echo "  python scripts/collate_batched_results.py $DATESTR"
                echo "Skipping plot generation."
                GENERATE_PLOTS="false"
                GENERATE_SOURCE_PLOTS="false"
            fi
        else
            echo ""
            echo "Using existing combined results: $COMBINED_FILE"
        fi
    fi
fi

# ============================================================
# GENERATE PLOTS (if enabled)
# ============================================================
if [ "$GENERATE_PLOTS" = "true" ]; then
    echo ""
    echo "============================================================"
    echo "GENERATING SUMMARY PLOTS"
    echo "============================================================"

    # Optionally wipe the existing SNR cache so it is rebuilt from parquet.
    if [ "$RECOMPUTE_SNR_CACHE" = "true" ]; then
        SNR_CACHE_FILE="/pscratch/sd/r/rmfeder/figures/redshift_validation/${DATESTR}/snr_diagnostics/snr_cache.npz"
        if [ -f "$SNR_CACHE_FILE" ]; then
            echo "Removing stale SNR cache: $SNR_CACHE_FILE"
            rm -f "$SNR_CACHE_FILE"
            echo "✓ Cache removed — will be recomputed from parquet during plot generation."
        else
            echo "(RECOMPUTE_SNR_CACHE=true but no cache found at $SNR_CACHE_FILE — nothing to remove.)"
        fi
    fi

    PLOT_CMD=(python scripts/generate_redshift_plots.py --datestr "$DATESTR")
    if [ -n "$COMPARISON_DATESTR" ]; then
        PLOT_CMD+=(--compare-datestr "$COMPARISON_DATESTR")
    else
        PLOT_CMD+=(--compare-datestr "")
    fi
    "${PLOT_CMD[@]}"
    
    PLOT_EXIT_CODE=$?
    
    if [ $PLOT_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ WARNING: Summary plot generation failed with exit code $PLOT_EXIT_CODE"
        echo "Redshift results are still valid and saved."
        echo "Continuing to per-source plots..."
    else
        echo ""
        echo "✓ Summary plots completed successfully!"
        echo "Figures saved to: /pscratch/sd/r/rmfeder/figures/redshift_validation/$DATESTR"
        (cd /pscratch/sd/r/rmfeder/figures/redshift_validation && zip -r "${DATESTR}_figures.zip" "$DATESTR")
    fi
fi

# ============================================================
# GENERATE PER-SOURCE DIAGNOSTIC PLOTS (if enabled)
# ============================================================
if [ "$GENERATE_SOURCE_PLOTS" = "true" ]; then
    echo ""
    echo "============================================================"
    echo "GENERATING PER-SOURCE DIAGNOSTIC PLOTS"
    echo "============================================================"
    echo "Configuration auto-loaded from run_params.npz"
    echo "N good sources: $N_GOOD_SOURCES"
    echo "N bad sources: $N_BAD_SOURCES (worst by |zscore|)"
    echo ""
    
    python scripts/generate_source_reconstructions.py \
        --datestr "$DATESTR" \
        --n-good $N_GOOD_SOURCES \
        --n-bad $N_BAD_SOURCES \
        --rhat-max 1.5 \
        --chi2-max 5.0 \
        --verbose
    
    SOURCE_PLOT_EXIT_CODE=$?
    
    if [ $SOURCE_PLOT_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ WARNING: Source plot generation failed with exit code $SOURCE_PLOT_EXIT_CODE"
        echo "Summary plots and redshift results are still valid."
        exit $SOURCE_PLOT_EXIT_CODE
    fi
    
    echo ""
    echo "✓ Per-source diagnostic plots completed successfully!"
    echo "  Figures saved to: /pscratch/sd/r/rmfeder/figures/redshift_validation/$DATESTR"
    echo "    - good_fits/ (${N_GOOD_SOURCES} sources)"
    echo "    - bad_fits/ (${N_BAD_SOURCES} sources)"
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All pipeline steps completed"
echo "Run configuration saved to: run_params.npz"
echo "Results directory: /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
echo "Full log saved to: $LOG_FILE"
if [ "$GENERATE_PLOTS" = "true" ] || [ "$GENERATE_SOURCE_PLOTS" = "true" ]; then
    echo "Figures directory: /pscratch/sd/r/rmfeder/figures/redshift_validation/$DATESTR"
fi
echo ""
echo "To view saved configuration:"
echo "  python -c \"import numpy as np; c=np.load('/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR/run_params.npz', allow_pickle=True); print({k: c[k] for k in c.files})\""
echo ""
echo "To view log file:"
echo "  less $LOG_FILE"
echo "============================================================"