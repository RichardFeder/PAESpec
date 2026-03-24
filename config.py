import os

sphx_base_path = '/global/homes/r/rmfeder/sed_vae/'
sphx_project_path = sphx_base_path + 'sp-ae-herex/'
scratch_basepath = '/pscratch/sd/r/rmfeder/' # needed for later

# Data paths
# sphx_dat_path = sphx_base_path+'data/'
sphx_dat_path = scratch_basepath+'data/'
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



