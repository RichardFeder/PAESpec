#!/bin/bash
# Run PAE batched redshift estimation on mock data (single GPU)
# This script handles mock data in the old format (not parquet)
# Optionally generates summary plots after completion

set -e  # Exit on error

# ============================================================
# CONFIGURATION
# ============================================================
# Set to "true" to automatically collate multi-task results
# Set to "false" to skip collation (can run manually later)
COLLATE_RESULTS="${COLLATE_RESULTS:-true}"

# Set to "true" to automatically generate plots after redshift estimation
# Set to "false" to skip plotting (can run manually with run_plot_afterburner.sh)
GENERATE_PLOTS="${GENERATE_PLOTS:-true}"

# Set to "true" to skip redshift estimation and only run plotting
# (Useful for re-running plots on existing results)
# SKIP_REDSHIFT_RUN="${SKIP_REDSHIFT_RUN:-false}"
SKIP_REDSHIFT_RUN="true"

# Set to "true" to reprocess existing sample files with a reduced burn-in,
# then re-collate before plotting.  Requires --save-samples to have been used
# in the original run (PAE_samples_batch*.npz must exist).
# Set REPROCESS_BURNIN_VALUE to the number of chain steps to discard.
REPROCESS_BURNIN="${REPROCESS_BURNIN:-false}"
REPROCESS_BURNIN_VALUE="${REPROCESS_BURNIN_VALUE:-200}"

# Model run directory under modl_runs/.
# Override at runtime, e.g.:
#   MODEL_RUN_NAME="fp_nlatent=5_013126" bash scripts/run_redshift_job_mock.sh
# MODEL_RUN_NAME="${MODEL_RUN_NAME:-fp_nlatent=10_031326}"
MODEL_RUN_NAME="fp_nlatent=10_031426"
# MODEL_RUN_NAME="pae_marg_nlatent5_sig001"

# MODEL_RUN_NAME=fp_nlatent=10_withsimloss_v2 \

# Prior toggles for matrix runs
# NF_ALPHA=1.0 enables NF prior, NF_ALPHA=0.0 disables it.
NF_ALPHA=1.0
# BPZ_PRIOR_ENABLED=true -> prior-type 2 + BPZ prior JSON
# BPZ_PRIOR_ENABLED=false -> prior-type 0 (no redshift prior)
BPZ_PRIOR_ENABLED=true
BPZ_PRIOR_JSON="${BPZ_PRIOR_JSON:-figures/redshift_priors/binned_dz02/redshift_prior_fit_results.json}"
SAMPLE_LOG_REDSHIFT="${SAMPLE_LOG_REDSHIFT:-true}"

# Set datestr for output
# DATESTR="mock_validation_siglevelnorm0.01_lr1e-3_091325_flow092225"
# DATESTR="mock_validation_011225"


# DATESTR="mock_validation_011326_20k_nfalpha=1_zpriorGauss0p4"

# BASE_DATESTR="mock_validation_011326_20k_nfalpha=0_noprior"
# DATESTR="mock_validation_011326_20k_nfalpha=1_noprior"

# DATESTR="mock_validation_011326_20k_nfalpha=1_zpriorGauss0.5"
# DATESTR="mock_validation_021126_1k_debug_nfalpha=1_zpriorGauss0.5"
# DATESTR="mock_20ktest_bpz_fixed_021326"
# DATESTR="mock_20ktest_021626_nfalpha=1_zpriorGauss0.5"
# DATESTR="mock_20ktest_logz_bpz_nfalpha=1_022726"

# DATESTR="mock_20ktest_logz_unifz_nfalpha=1_022726"
# DATESTR="mock_20ktest_bpz_nfalpha=0_nfburnin=1_030126"

# DATESTR="mock_20ktest_logz_bpz_nfalpha=1_030226"
# DATESTR="mock_20ktest_unifz_nfalpha=0_nfburnin=1_030226"

# DATESTR="mock_2ktest_bpz_nfalpha=1_sbi_compare_030226"

# DATESTR="${DATESTR:-mock_20ktest_logz_bpz_nfalpha=1_nlatent=10_031426}"
# DATESTR=mock_nf1_bpz1_simlossv2_$(date +%m%d%y_%H%M%S)
# DATESTR=mock_nf1_bpz1_simlossv2_031426_173346
# DATESTR=mock_redshift_4case_031426_163845_nf0_bpz0
# DATESTR=mock_redshift_031526_margpae_nf0_bpz0

# DATESTR=mock_redshift_nlatent20_031426_nf1_bpz1

DATESTR=mock_20ktest_logz_bpz_nfalpha=1_nlatent=10_031426

# DATESTR=mock_redshift_031526__nf0_bpz0


# ============================================================
# TIMING DIAGNOSTICS (persistent run profiling for forecasting)
# ============================================================
RUN_START_EPOCH=$(date +%s)
RUN_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

PHASE_REDSHIFT_SEC=-1
PHASE_COLLATE_SEC=-1
PHASE_REPROCESS_SEC=-1
PHASE_PLOT_SEC=-1
PHASE_ZIP_SEC=-1

REDSHIFT_STATUS="not-run"
COLLATE_STATUS="not-run"
REPROCESS_STATUS="not-run"
PLOT_STATUS="not-run"
ZIP_STATUS="not-run"

TIMING_WRITTEN="false"

write_timing_summary() {
        if [ "$TIMING_WRITTEN" = "true" ]; then
                return
        fi
        TIMING_WRITTEN="true"

        local exit_code="$1"
        local run_end_epoch run_end_iso total_wall_sec
        run_end_epoch=$(date +%s)
        run_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        total_wall_sec=$((run_end_epoch - RUN_START_EPOCH))

        local base_results_dir run_results_dir timing_json timing_csv
        base_results_dir="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched"
        run_results_dir="${base_results_dir}/${DATESTR}"
        timing_json="${run_results_dir}/timing_summary_${DATESTR}.json"
        timing_csv="${base_results_dir}/timing_registry_mock.csv"

        mkdir -p "$run_results_dir"

        cat > "$timing_json" <<EOF
{
    "datestr": "${DATESTR}",
    "model_run_name": "${MODEL_RUN_NAME}",
    "run_start_utc": "${RUN_START_ISO}",
    "run_end_utc": "${run_end_iso}",
    "total_wallclock_sec": ${total_wall_sec},
    "exit_code": ${exit_code},
    "config": {
        "skip_redshift_run": "${SKIP_REDSHIFT_RUN}",
        "collate_results": "${COLLATE_RESULTS}",
        "generate_plots": "${GENERATE_PLOTS}",
        "reprocess_burnin": "${REPROCESS_BURNIN}",
        "reprocess_burnin_value": ${REPROCESS_BURNIN_VALUE},
        "nf_alpha": "${NF_ALPHA}",
        "bpz_prior_enabled": "${BPZ_PRIOR_ENABLED}",
        "sample_log_redshift": "${SAMPLE_LOG_REDSHIFT}"
    },
    "phase_durations_sec": {
        "redshift": ${PHASE_REDSHIFT_SEC},
        "collate": ${PHASE_COLLATE_SEC},
        "reprocess": ${PHASE_REPROCESS_SEC},
        "plot": ${PHASE_PLOT_SEC},
        "zip": ${PHASE_ZIP_SEC}
    },
    "phase_status": {
        "redshift": "${REDSHIFT_STATUS}",
        "collate": "${COLLATE_STATUS}",
        "reprocess": "${REPROCESS_STATUS}",
        "plot": "${PLOT_STATUS}",
        "zip": "${ZIP_STATUS}"
    },
    "slurm": {
        "job_id": "${SLURM_JOB_ID:-}",
        "array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
        "array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
        "node": "${SLURMD_NODENAME:-$(hostname)}"
    }
}
EOF

        if [ ! -f "$timing_csv" ]; then
                echo "datestr,run_start_utc,run_end_utc,total_wallclock_sec,exit_code,redshift_sec,collate_sec,reprocess_sec,plot_sec,zip_sec,redshift_status,collate_status,reprocess_status,plot_status,zip_status,model_run_name,nf_alpha,bpz_prior_enabled,sample_log_redshift,skip_redshift_run,collate_results,generate_plots,reprocess_burnin,reprocess_burnin_value,slurm_job_id,slurm_array_job_id,slurm_array_task_id,node" > "$timing_csv"
        fi

        echo "${DATESTR},${RUN_START_ISO},${run_end_iso},${total_wall_sec},${exit_code},${PHASE_REDSHIFT_SEC},${PHASE_COLLATE_SEC},${PHASE_REPROCESS_SEC},${PHASE_PLOT_SEC},${PHASE_ZIP_SEC},${REDSHIFT_STATUS},${COLLATE_STATUS},${REPROCESS_STATUS},${PLOT_STATUS},${ZIP_STATUS},${MODEL_RUN_NAME},${NF_ALPHA},${BPZ_PRIOR_ENABLED},${SAMPLE_LOG_REDSHIFT},${SKIP_REDSHIFT_RUN},${COLLATE_RESULTS},${GENERATE_PLOTS},${REPROCESS_BURNIN},${REPROCESS_BURNIN_VALUE},${SLURM_JOB_ID:-},${SLURM_ARRAY_JOB_ID:-},${SLURM_ARRAY_TASK_ID:-},${SLURMD_NODENAME:-$(hostname)}" >> "$timing_csv"

        echo ""
        echo "Timing diagnostics saved: $timing_json"
        echo "Timing registry updated: $timing_csv"
}

trap 'write_timing_summary "$?"' EXIT



# OUTPUT_CATGRID="/pscratch/sd/r/rmfeder/data/phot/catgrid_sphxonly_COSMOS_zlt20_sbi_comparison"
# OUTPUT_SELECT_INFO="/pscratch/sd/r/rmfeder/data/select_catgrid_info/catgrid_info_COSMOS_zlt20_sbi_comparison"


# To use NF prior during burn-in only: set --nf-alpha-burnin 1.0 and --nf-alpha 0.0

# ============================================================
# RUN REDSHIFT ESTIMATION
# ============================================================

if [ "$SKIP_REDSHIFT_RUN" = "true" ]; then
    REDSHIFT_STATUS="skipped"
    echo "============================================================"
    echo "SKIPPING REDSHIFT ESTIMATION (SKIP_REDSHIFT_RUN=true)"
    echo "============================================================"
    echo "Using existing results for: $DATESTR"
    echo ""
else
    echo "============================================================"
    echo "RUNNING MOCK DATA REDSHIFT ESTIMATION"
    echo "============================================================"
    echo "Date string: $DATESTR"
    echo "Model run: $MODEL_RUN_NAME"
    echo "NF alpha: $NF_ALPHA"
    echo "BPZ prior enabled: $BPZ_PRIOR_ENABLED"
    echo "Collate results: $COLLATE_RESULTS"
    echo "Generate plots after: $GENERATE_PLOTS"
    echo ""

    PRIOR_TYPE=0
    EXTRA_PRIOR_ARGS=()
    BPZ_ENABLED_LC=$(echo "$BPZ_PRIOR_ENABLED" | tr '[:upper:]' '[:lower:]')
    if [ "$BPZ_ENABLED_LC" = "true" ] || [ "$BPZ_ENABLED_LC" = "1" ] || [ "$BPZ_ENABLED_LC" = "yes" ]; then
        PRIOR_TYPE=2
        EXTRA_PRIOR_ARGS+=(--bpz-prior-json "$BPZ_PRIOR_JSON")
    fi


    # log-z BPZ (sample_log_redshift, prior-type=2, 1 task, 2 batches of 1000)
    # python scripts/redshift_job_mock_batched.py \
    #   --start-task 0 \
    #   --sources-per-task 5000 \
    #   --max-tasks 4 \
    #   --batch-size 1000 \
    #   --sampling-batch-size 1000 \
    #   --datestr "$DATESTR" \
    #   --filter-set spherex_filters102/ \
    #   --sig-level-norm 0.01 \
    #   --collate-results \
    #   --use-multicore \
    #   --n-devices 4 \
    #   --nf-alpha 0.0 \
    #   --prior-type 2 \
    #   --bpz-prior-json figures/redshift_priors/binned_dz02/redshift_prior_fit_results.json \
    #   --sample-log-redshift
    
    # log-z uniform z (sample_log_redshift, prior-type=2, 1 task, 2 batches of 1000)
    # python scripts/redshift_job_mock_batched.py \
    #   --start-task 0 \
    #   --sources-per-task 2000 \
    #   --max-tasks 1 \
    #   --batch-size 1000 \
    #   --sampling-batch-size 1000 \
    #   --datestr "$DATESTR" \
    #   --filter-set spherex_filters102/ \
    #   --sig-level-norm 0.01 \
    #   --collate-results \
    #   --use-multicore \
    #   --n-devices 4 \
    #   --nf-alpha 1.0 \
    #   --nf-alpha-burnin 1.0 \
    #   --prior-type 0 \
    #   --sample-log-redshift


    # Previous 20k run (no log-z, Gaussian prior)
    # python scripts/redshift_job_mock_batched.py \
    #   --start-task 0 \
    #   --sources-per-task 5000 \
    #   --max-tasks 4 \
    #   --batch-size 1000 \
    #   --sampling-batch-size 1000 \
    #   --datestr "$DATESTR" \
    #   --filter-set spherex_filters102/ \
    #   --sig-level-norm 0.01 \
    #   --collate-results \
    #   --use-multicore \
    #   --n-devices 4 \
    #   --nf-alpha 1.0 \
    #   --prior-type 1 \
    #   --sigma-prior 0.5


        CMD=(python scripts/redshift_job_mock_batched.py
            --start-task 0
            --sources-per-task 5000
            --max-tasks 4
            --batch-size 1000
            --sampling-batch-size 1000
            --datestr "$DATESTR"
            --filter-set spherex_filters102/
            --sig-level-norm 0.01
            --run-name "$MODEL_RUN_NAME"
            --use-multicore
            --n-devices 4
            --nf-alpha "$NF_ALPHA"
            --prior-type "$PRIOR_TYPE")

        if [ "$COLLATE_RESULTS" = "true" ]; then
            CMD+=(--collate-results)
        fi

        if [ "$SAMPLE_LOG_REDSHIFT" = "true" ]; then
            CMD+=(--sample-log-redshift)
        fi

        if [ ${#EXTRA_PRIOR_ARGS[@]} -gt 0 ]; then
            CMD+=("${EXTRA_PRIOR_ARGS[@]}")
        fi

        REDSHIFT_STATUS="running"
        REDSHIFT_PHASE_START=$(date +%s)
        set +e
        "${CMD[@]}"
        REDSHIFT_EXIT_CODE=$?
        set -e
        PHASE_REDSHIFT_SEC=$(( $(date +%s) - REDSHIFT_PHASE_START ))
        if [ $REDSHIFT_EXIT_CODE -eq 0 ]; then
            REDSHIFT_STATUS="success"
        else
            REDSHIFT_STATUS="failed"
        fi
    #   --data-fpath "$OUTPUT_CATGRID" \

    # Save this script to the results directory for reproducibility
    RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
    if [ -d "$RESULTS_DIR" ]; then
        cp "$0" "$RESULTS_DIR/run_script.sh"
        echo "Saved run script to: $RESULTS_DIR/run_script.sh"
    fi

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
    COLLATE_STATUS="running"
    COLLATE_PHASE_START=$(date +%s)
    # Just finished redshift run, collate results
    echo ""
    echo "============================================================"
    echo "COLLATING BATCH RESULTS"
    echo "============================================================"
    
    # Count batch files before collation
    RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
    N_BATCH_FILES=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | wc -l)
    echo "Number of batch files to collate: $N_BATCH_FILES"
    
    set +e
    python scripts/collate_batched_results.py "$DATESTR"
    COLLATE_EXIT_CODE=$?
    set -e
    PHASE_COLLATE_SEC=$(( $(date +%s) - COLLATE_PHASE_START ))
    
    if [ $COLLATE_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ WARNING: Result collation failed with exit code $COLLATE_EXIT_CODE"
        echo "Individual batch files are still valid."
        echo "Skipping plot generation (requires combined file)."
        GENERATE_PLOTS="false"
        COLLATE_STATUS="failed"
    else
        echo ""
        echo "✓ Results collated successfully!"
        echo "Combined file: $COMBINED_FILE"
        COLLATE_STATUS="success"
    fi
elif [ "$SKIP_REDSHIFT_RUN" = "true" ]; then
    # Skipped redshift run, check if we need combined file for plots
    if [ "$GENERATE_PLOTS" = "true" ]; then
        if [ "$COLLATE_RESULTS" = "true" ]; then
            COLLATE_STATUS="running"
            COLLATE_PHASE_START=$(date +%s)
            echo ""
            echo "============================================================"
            echo "COLLATING BATCH RESULTS"
            echo "============================================================"
            if [ -f "$COMBINED_FILE" ]; then
                echo "Overwriting existing combined file: $COMBINED_FILE"
            else
                echo "Combined file not found, running collation..."
            fi

            # Count batch files before collation
            RESULTS_DIR="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/$DATESTR"
            N_BATCH_FILES=$(find "$RESULTS_DIR" -name "PAE_results_batch*.npz" | wc -l)
            echo "Number of batch files to collate: $N_BATCH_FILES"

            set +e
            python scripts/collate_batched_results.py "$DATESTR"
            COLLATE_EXIT_CODE=$?
            set -e
            PHASE_COLLATE_SEC=$(( $(date +%s) - COLLATE_PHASE_START ))
            if [ $COLLATE_EXIT_CODE -ne 0 ]; then
                echo "⚠ WARNING: Result collation failed"
                echo "Skipping plot generation."
                GENERATE_PLOTS="false"
                COLLATE_STATUS="failed"
            else
                echo "✓ Results collated successfully!"
                echo "Combined file: $COMBINED_FILE"
                COLLATE_STATUS="success"
            fi
        else
            # User doesn't want to collate
            if [ ! -f "$COMBINED_FILE" ]; then
                echo ""
                echo "⚠ WARNING: Combined results file not found: $COMBINED_FILE"
                echo "Set COLLATE_RESULTS=\"true\" to generate it, or run manually:"
                echo "  python scripts/collate_batched_results.py $DATESTR"
                echo "Skipping plot generation."
                GENERATE_PLOTS="false"
            else
                echo ""
                echo "Using existing combined results: $COMBINED_FILE"
            fi
            if [ "$COLLATE_STATUS" = "not-run" ]; then
                COLLATE_STATUS="skipped"
            fi
        fi
    fi
fi

# ============================================================
# REPROCESS WITH REDUCED BURN-IN (if enabled)
# ============================================================
if [ "$REPROCESS_BURNIN" = "true" ]; then
    REPROCESS_STATUS="running"
    REPROCESS_PHASE_START=$(date +%s)
    echo ""
    echo "============================================================"
    echo "REPROCESSING SAMPLES WITH REDUCED BURN-IN = $REPROCESS_BURNIN_VALUE"
    echo "============================================================"

        set +e
        python scripts/reprocess_with_burnin.py "$DATESTR" \
            --burn-in "$REPROCESS_BURNIN_VALUE"
        REPROCESS_EXIT_CODE=$?
        set -e
        PHASE_REPROCESS_SEC=$(( $(date +%s) - REPROCESS_PHASE_START ))
    if [ $REPROCESS_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ WARNING: Reprocessing failed with exit code $REPROCESS_EXIT_CODE"
        echo "Continuing with existing combined file."
                REPROCESS_STATUS="failed"
    else
        echo ""
        echo "✓ Reprocessing complete. Combined file updated with burn_in=$REPROCESS_BURNIN_VALUE"
                REPROCESS_STATUS="success"
    fi
else
        REPROCESS_STATUS="skipped"
fi

# ============================================================
# GENERATE PLOTS (if enabled)
# ============================================================
if [ "$GENERATE_PLOTS" = "true" ]; then
    PLOT_STATUS="running"
    PLOT_PHASE_START=$(date +%s)
    echo ""
    echo "============================================================"
    echo "GENERATING SUMMARY PLOTS"
    echo "============================================================"
    
    # Create output directory if it doesn't exist
    FIGURE_DIR="/pscratch/sd/r/rmfeder/figures/redshift_validation/$DATESTR"
    mkdir -p "$FIGURE_DIR"
    
    # Use --datestr for automatic file detection (generates mock-specific plots)
        set +e
        python3 scripts/generate_mock_plots.py --datestr "$DATESTR" \
      --output-dir "$FIGURE_DIR" \
      --uncertainty-histogram \
      --chi2-comparison \
      --raw-chi2-histogram \
      --ztf-zpae-comparison \
      --rhat-zscore-analysis \
      --rhat-max 100 \
      --chi2-max 2.0 \
      --hexbin-comparison \
      --hexbin-gridsize 50 \
    --zerr-binned-corr \
      --cross-selection \
      --sigz-binned-plots \
      --cross-selection-sigz-binned \
      --cross-selection-sigz-ratio \
      --cross-selection-sigz-fixed \
      --cross-selection-sigz-bin-edges 0 0.03 0.1 0.2 \
      --convergence-diagnostics
    

    PLOT_STATUS="success"
    
    echo ""
    echo "✓ Plot generation completed successfully!"
    echo "Figures saved to: $FIGURE_DIR"

    # Zip all figures for easy download/transfer
    ZIP_STATUS="running"
    ZIP_PHASE_START=$(date +%s)
    ZIP_FILE="${FIGURE_DIR}/../${DATESTR}_figures.zip"
    ZIP_FILE=$(realpath "$ZIP_FILE")
    echo ""
    echo "Creating figures zip: $ZIP_FILE"
    set +e
    zip -j "$ZIP_FILE" "$FIGURE_DIR"/*.png "$FIGURE_DIR"/*.pdf 2>/dev/null || \
        zip -j "$ZIP_FILE" "$FIGURE_DIR"/*.png 2>/dev/null || \
        zip -r "$ZIP_FILE" "$FIGURE_DIR"
    ZIP_EXIT_CODE=$?
    set -e
    PHASE_ZIP_SEC=$(( $(date +%s) - ZIP_PHASE_START ))
    if [ $ZIP_EXIT_CODE -eq 0 ]; then
        ZIP_STATUS="success"
    else
        ZIP_STATUS="failed"
    fi
    echo "✓ Figures zipped to: $ZIP_FILE"
else
    PLOT_STATUS="skipped"
    ZIP_STATUS="skipped"
    echo ""
    echo "Plot generation skipped (GENERATE_PLOTS=$GENERATE_PLOTS)"
    echo "To generate plots later, run:"
    echo "  python3 scripts/generate_mock_plots.py --datestr $DATESTR --output-dir /pscratch/sd/r/rmfeder/figures/redshift_validation/$DATESTR"
fi

echo ""
echo "============================================================"
echo "✓ ALL TASKS COMPLETED"
echo "============================================================"
