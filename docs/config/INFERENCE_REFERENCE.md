# Mock Redshift Argument Reference

Script: `scripts/redshift_job_mock_batched.py`

## High-priority run control

- `--config-yaml`: Path to YAML defaults file
- `--run-name`: Trained model run name
- `--datestr`: Output run tag for saved files/directories
- `--collate-results`: Combine per-batch files after completion
- `--stop-on-error`: Stop remaining tasks on first failure

## Data and model selection

- `--filter-set`: Filter-set subdirectory name
- `--nlatent`: Latent dimension expected by trained model
- `--sig-level-norm`: Noise/normalization level matching training setup
- `--sel-str`: Selection label passed to loader logic
- `--with-ext-phot`: Include external photometry if present
- `--data-fpath`: Optional explicit data-file override
- `--filename-flow`: Flow model filename stem

## Task partitioning and throughput

- `--sources-per-task`: Number of sources per task (required)
- `--start-task`: Starting task index
- `--max-tasks`: Number of tasks to process
- `--batch-size`: Sources per model-evaluation batch
- `--sampling-batch-size`: Chains/sources sampled concurrently

## Sample filtering

- `--z-min`, `--z-max`: Redshift filter bounds
- `--snr-min`, `--snr-max`: SNR filter bounds

## MCLMC controls

- `--num-steps`: Total MCMC steps
- `--burn-in`: Burn-in steps
- `--nsamp-init`: Initialization sample count
- `--chi2-red-threshold`: Reduced-chi2 threshold for quality/reinit logic
- `--gr-threshold`: Gelman-Rubin threshold
- `--fix-z`: Fix redshift instead of sampling

## Prior and parameterization controls

- `--nf-alpha`: Flow-prior strength in final sampling
- `--nf-alpha-burnin`: Flow-prior strength during burn-in
- `--prior-type`: Prior mode (for example none/Gaussian/BPZ)
- `--z0-prior`, `--sigma-prior`: Prior-shape controls
- `--bpz-prior-json`: BPZ prior fit JSON path
- `--sample-log-redshift`: Sample in log-redshift space
- `--sample-log-amplitude`: Sample log-amplitude explicitly
- `--log-amplitude-prior-std`: Log-amplitude prior width

## Multi-device execution

- `--use-multicore`: Enable multi-device mode
- `--n-devices`: Number of devices in multi-device mode

## SNR-prefit initialization

- `--use-snr-prefit-init`: Enable SNR-based initialization from prefit mapping
- `--snr-prefit-json`: SNR-prefit parameter JSON path
- `--snr-prefit-column`: Column name used for SNR lookup
