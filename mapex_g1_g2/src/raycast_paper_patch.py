"""Frozen MapEx PAPER raycast delta: reset occupancy sum ONCE PER RAY.
Runtime applies the corrected Numba function to the original MapEx module.
No change to original MapEx files, WFD, local controller, LaMa, or A*.
"""
from __future__ import print_function
import inspect,hashlib
EXACT_OFFICIAL_SOURCE_SHA="74f13cc8f013b26c733067fc99811cc001d37ac356bece86e53fcccf4a5ec576"
def install(module):
    source=inspect.getsource(module.get_free_points.py_func)
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest!=EXACT_OFFICIAL_SOURCE_SHA:
        raise RuntimeError("RAYCAST_ORIGINAL_SOURCE_CHANGED "+digest)
    marker="        bresenham_points = bresenham_points_list[endpoint_i]"
    reset="            accum_hit_prob = 0\n"
    assert source.count(marker)==1 and source.count(reset)==1
    fixed=source.replace(marker,"        accum_hit_prob = 0\n"+marker,1).replace(reset,"",1)
    assert fixed.count("accum_hit_prob = 0")==1
    namespace=dict(module.__dict__)
    exec(compile(fixed,"<paper-cumulative-get_free_points>","exec"),namespace)
    old=module.get_free_points
    module.get_free_points=namespace["get_free_points"]
    if module.get_free_points is old:raise RuntimeError("PATCH_NOT_APPLIED")
    print("MAPEX_RAYCAST_PAPER_CORRECTION_ACTIVE",
      {"original_sha256":digest,
       "patch":"single accum_hit_prob=0 before bresenham cell loop, within each ray",
       "source_file_unmodified":True},flush=True)
