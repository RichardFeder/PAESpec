import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_default_project_path = _this_dir + '/'
_default_base_path = os.path.dirname(_this_dir.rstrip('/')) + '/'

# Base paths are environment-configurable for portability.
sphx_base_path = os.environ.get('SPAE_BASE_PATH', _default_base_path)
if not sphx_base_path.endswith('/'):
	sphx_base_path += '/'

sphx_project_path = os.environ.get('SPAE_PROJECT_PATH', _default_project_path)
if not sphx_project_path.endswith('/'):
	sphx_project_path += '/'

# Scratch base remains configurable for HPC users, but defaults locally.
scratch_basepath = os.environ.get('SPAE_SCRATCH_BASEPATH', sphx_base_path)
if not scratch_basepath.endswith('/'):
	scratch_basepath += '/'

# Data paths
# sphx_dat_path = sphx_base_path+'data/'
sphx_dat_path = os.environ.get('SPAE_DATA_PATH', os.path.join(scratch_basepath, 'data'))
if not sphx_dat_path.endswith('/'):
	sphx_dat_path += '/'
filt_basepath = sphx_dat_path+'filters/'

# Model runs directory (where trained models are stored)
_modl_runs_default_scratch = os.path.join(scratch_basepath, 'sed_vae', 'modl_runs')
_modl_runs_default_home = os.path.join(sphx_base_path, 'modl_runs')
modl_runs_path = os.environ.get('SPAE_MODEL_RUNS_PATH', _modl_runs_default_scratch)
if not os.path.isdir(modl_runs_path):
    modl_runs_path = _modl_runs_default_home
if not modl_runs_path.endswith('/'):
    modl_runs_path += '/'

# Profile likelihood data directory
profile_like_path = sphx_dat_path + 'profile_like/'

# External filter directory
ext_filt_path = sphx_dat_path + 'filters/ext_filters/'



