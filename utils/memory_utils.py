"""
Utility functions for monitoring GPU and host memory usage.

These functions provide detailed memory diagnostics but may not capture
all GPU memory usage (e.g., neural network models). Use with caution.
"""

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_gpu_memory_limit():
    """Detect GPU memory limit in GB.
    
    Returns:
        float: GPU memory limit in GB (e.g., 40 or 80 for A100)
    """
    try:
        import jax
        devices = jax.devices()
        if devices:
            # Try to get memory limit from device
            device = devices[0]
            # NVIDIA A100 comes in 40GB and 80GB variants
            # Try to infer from device name if available
            device_name = str(device).lower()
            if 'a100' in device_name and '80' in device_name:
                return 80.0
            elif 'a100' in device_name:
                return 40.0
            # Default assumption for modern GPUs
            return 80.0
    except:
        pass
    return 80.0  # Conservative default


def estimate_batch_size_memory(n_chains, n_steps=2000, n_bands=102, n_latent=5):
    """Estimate GPU memory usage for a given batch size.
    
    This is a rough approximation based on empirical observations:
    - Each chain requires storage for samples (n_steps × n_latent floats)
    - Model parameters and intermediate computations add overhead
    - vmap creates arrays for all chains simultaneously
    
    NOTE: This does NOT account for neural network model memory,
    which can be substantial. Use as a rough guide only.
    
    Args:
        n_chains: Number of parallel chains (sampling_batch_size)
        n_steps: Number of MCMC steps per chain
        n_bands: Number of filter bands
        n_latent: Latent dimensionality
    
    Returns:
        float: Estimated GPU memory in GB
    """
    # Bytes per float32
    bytes_per_float = 4
    
    # Sample storage: n_chains × n_steps × (n_latent + 1 for z)
    sample_memory = n_chains * n_steps * (n_latent + 1) * bytes_per_float
    
    # Model forward passes are vmapped, creating temporary arrays
    # Rough estimate: 3-5x the sample storage for intermediate computations
    # This includes gradients, likelihoods, model evaluations
    overhead_multiplier = 4.0
    
    # Additional overhead for PAE model, flow model, etc. (~ 2-3 GB)
    model_overhead_gb = 2.5
    
    total_bytes = sample_memory * overhead_multiplier
    total_gb = (total_bytes / 1024**3) + model_overhead_gb
    
    return total_gb


def print_batch_size_guide(current_batch_size=800, gpu_limit_gb=80.0):
    """Print a guide for choosing batch sizes based on GPU memory.
    
    Args:
        current_batch_size: Current sampling batch size per GPU
        gpu_limit_gb: GPU memory limit in GB
    """
    print(f"\n{'='*70}")
    print("BATCH SIZE OPTIMIZATION GUIDE")
    print(f"{'='*70}")
    print(f"GPU Memory Limit: {gpu_limit_gb:.0f} GB")
    print(f"Current batch size: {current_batch_size} chains/GPU")
    print(f"\nEstimated memory usage for different batch sizes:")
    print(f"{'Chains/GPU':<12} {'Est. Memory (GB)':<18} {'Utilization':<15} {'Recommendation'}")
    print("-" * 70)
    
    batch_sizes = [200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200]
    
    for batch_size in batch_sizes:
        est_mem = estimate_batch_size_memory(batch_size)
        utilization = (est_mem / gpu_limit_gb) * 100
        
        if utilization > 95:
            recommendation = "❌ TOO HIGH - Risk OOM"
        elif utilization > 85:
            recommendation = "⚠️  Near limit"
        elif utilization > 70:
            recommendation = "✓ Good target"
        elif utilization > 50:
            recommendation = "✓ Safe, can increase"
        else:
            recommendation = "✓ Conservative"
        
        marker = " <-- CURRENT" if batch_size == current_batch_size else ""
        print(f"{batch_size:<12} {est_mem:<18.1f} {utilization:>5.1f}%          {recommendation}{marker}")
    
    print(f"\n{'='*70}")
    print("NOTES:")
    print("  • These are ESTIMATES - actual usage varies by workload")
    print("  • Does NOT include neural network model memory (can be significant)")
    print("  • Watch for 'Peak' memory, not just 'Current'")
    print("  • Performance degrades when memory > 85% (swapping to host)")
    print("  • Multicore mode: Total chains = batch_size × n_devices")
    print(f"  • With 4 GPUs: {current_batch_size} × 4 = {current_batch_size * 4} total parallel chains")
    print(f"{'='*70}\n")


def get_detailed_memory_info():
    """Get detailed memory breakdown.
    
    Returns:
        dict or None: Dictionary with memory info or None if psutil not available
    """
    if not HAS_PSUTIL:
        return None
    
    proc = psutil.Process()
    mem_info = proc.memory_info()
    
    return {
        'rss_gb': mem_info.rss / 1024**3,  # Resident Set Size (actual RAM)
        'vms_gb': mem_info.vms / 1024**3,  # Virtual Memory Size
        'shared_gb': getattr(mem_info, 'shared', 0) / 1024**3,
    }


def print_memory_status(label, batch_idx=None, gpu_limit_gb=80.0):
    """Print basic memory status (host RAM only).
    
    NOTE: Does not report detailed GPU memory as it doesn't capture
    neural network model memory which is the main GPU memory consumer.
    
    Args:
        label: Description of current state
        batch_idx: Optional batch number
        gpu_limit_gb: GPU memory limit in GB (unused, kept for compatibility)
    """
    mem = get_detailed_memory_info()
    if mem is None:
        return
    
    prefix = f"[MEM] Batch {batch_idx} {label}" if batch_idx is not None else f"[MEM] {label}"
    print(f"\n{prefix}")
    print(f"  Host RAM - RSS (actual): {mem['rss_gb']:.2f} GB")
    print(f"  Host RAM - VMS (virtual): {mem['vms_gb']:.2f} GB")
    print(f"  Host RAM - Shared:        {mem['shared_gb']:.2f} GB")
