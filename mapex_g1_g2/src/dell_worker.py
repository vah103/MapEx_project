"""Full-loop fresh exploratory MapEx Native2D + R4.1 run, guarded externally; no QA claim."""
import os,sys,time,types,hashlib,traceback
from pathlib import Path
ROOT=Path(os.environ['DELL_NATIVE2D_ROOT'])
NATIVE=Path('/home/dell/MapEx')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(NATIVE/'scripts'))
sys.path.insert(0,str(NATIVE))
os.chdir(str(NATIVE/'scripts'))
import numpy as np
import torch
import cv2
torch.set_num_threads(int(os.environ.get('DELL_NATIVE2D_THREADS','2')))
try:torch.set_num_interop_threads(int(os.environ.get('DELL_NATIVE2D_THREADS','2')))
except Exception:pass
from scripts import simple_mask_utils
import raycast_paper_patch
raycast_paper_patch.install(simple_mask_utils)
import importlib.util
code=ROOT/'explore_room_r41_lama3_smallroi_v1.py'
mod=types.ModuleType('room_r41_full_run')
# Disable graphical figure rendering for stable headless execution; does not
# affect native scoring or Room Node outputs. This deviation is logged.
src=code.read_text(encoding="utf8")
assert src.count("matplotlib.use('TkAgg')")==1
src=src.replace("matplotlib.use('TkAgg')","matplotlib.use('Agg')")
assert src.count("            if show_plt:")==1,src.count("            if show_plt:")
src=src.replace("            if show_plt:","            if False and show_plt:")
compiled=compile(src,str(code),'exec')
exec(compiled,mod.__dict__)
opts=mod.get_options_dict_from_yml('base.yaml')
opts.lama_device='cpu'
opts.modes_to_test=['visvarprob']
opts.mission_time=1000000
opts.show_plt_freq=100000
mod.collect_opts=opts
mod.device='cpu'
output=ROOT/'native_opt_r41_lama3small_run04'
if output.exists():raise RuntimeError("full run output already exists")
output.mkdir()
mod.output_root_dir=str(output)
os.chdir(str(output))
print("MAX_CPU_PROFILE",{"torch_threads":torch.get_num_threads(),"interop":torch.get_num_interop_threads(),"affinity":len(os.sched_getaffinity(0)),"OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS"),"OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS")},flush=True)
print('PRELOAD_START',time.time(),flush=True)
models=[]
for name in ['lama_ensemble/train_1','lama_ensemble/train_2','lama_ensemble/train_3']:
    path='/home/dell/MapEx/pretrained_models/weights/'+name
    model=mod.load_lama_model(path,device='cpu')
    models.append(model)
    print('MODEL_LOADED',name,time.time(),flush=True)
big=None  # Removed redundant 4th big-G model from optimized variant
print('THREE_G1_G2_G3_MODELS_READY',time.time(),flush=True)
mod.model_list=models
transform=mod.get_lama_transform(opts.lama_transform_variant,opts.lama_out_size)
p1=np.load('/home/dell/Workspace/02_Research/MX_Experiments/MX072_COM1_20261012/fixtures/p1_raw.npy',allow_pickle=False)
assert p1.shape==(203,263)
observed=np.pad(p1,500,mode='constant',constant_values=0)
if os.environ.get('DELL_NATIVE2D_MODEL_SMOKE')=='1':
    print('DELL_NATIVE2D_MODEL_SMOKE_PASS',p1.shape,len(models),flush=True)
    sys.exit(0)
print('NATIVE_SOURCE_TICK_START',time.time(),flush=True)
mod.run_exploration_for_map(observed,'R41_OPT_LAMA3_SMALLROI_RUN04',models,big,transform,
    opts.pred_vis_configs,opts.lidar_sim_configs,'visvarprob',
    np.array([631,631],dtype=np.int64),opts.unknown_as_occ,
    opts.use_distance_transform_for_planning,upen_config=None)
print('RUN_FUNCTION_RETURNED',time.time(),flush=True)
import json
states=list(output.rglob('mission_status.json'))
if not states:raise RuntimeError('missing native mission status')
state=json.loads(states[0].read_text())
print('FULL_FINAL_STATE',json.dumps(state,sort_keys=True),flush=True)
print('R41_REPORT_FILES',len(list(output.rglob('decision_*_report.json'))),flush=True)
assert isinstance(state.get('mission_complete'),bool)
assert len(list(output.rglob('decision_*.json')))>=1
print('OPTIMIZED_RUN_FUNCTION_EXIT',state.get('mission_complete'),state.get('fail_reason'),flush=True)
