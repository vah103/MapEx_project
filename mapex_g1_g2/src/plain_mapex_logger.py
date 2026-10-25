"""Read-only evidence for PLAIN MapEx 2D; no Room Node imports or decisions."""
import os,json,time,hashlib
import numpy as np
def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
def atomic(path,obj):
    if os.path.exists(path):raise RuntimeError("output collision "+path)
    tmp=path+".partial"
    with open(tmp,"w") as f:
        json.dump(obj,f,sort_keys=True,indent=2)
        f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)
class PlainRecorder:
    def __init__(self,exp_dir):
        self.root=os.path.join(exp_dir,"plain_mapex_evidence")
        if os.path.exists(self.root):raise RuntimeError("prior baseline evidence exists")
        os.mkdir(self.root)
        self.steps=os.path.join(self.root,"steps")
        self.decisions=os.path.join(self.root,"decisions")
        os.mkdir(self.steps);os.mkdir(self.decisions)
        self.n=0;self.start=time.time()
        self.total_distance_m=0.
        self.last_pose=None
        self.last_decision=None
        atomic(os.path.join(self.root,"identity.json"),
            {"controller":"Original MapEx GLOBAL frontier selection only",
             "room_node_disabled":True,"frontier_scope":"GLOBAL_ONLY",
             "no_GT_in_scoring":True,"GT_read_only_native_sim_collision":True})
    def decision(self,t,pose,obs,mean,var,candidates,scores,pred_duration,pad,padtop,padleft,resolution):
        self.n+=1
        p=np.asarray(pose,dtype=np.int32)
        pts=np.asarray(candidates,dtype=np.int32).reshape((-1,2))
        costs=np.asarray(scores,dtype=np.float64).reshape(-1)
        if not len(pts) or len(pts)!=len(costs) or not np.isfinite(costs).all():
            raise RuntimeError("baseline candidate and score mismatch")
        full=np.asarray(obs)
        h,w=full.shape[0]-2*pad,full.shape[1]-2*pad
        assert h>0 and w>0
        core=np.asarray(full[pad:pad+h,pad:pad+w])
        m=np.asarray(mean)
        v=var.detach().cpu().numpy() if hasattr(var,"detach") else np.asarray(var)
        y=pad+padtop;x=pad+padleft
        mm=m[y:y+h,x:x+w]
        vv=v[y:y+h,x:x+w]
        if mm.shape!=core.shape or vv.shape!=core.shape:raise RuntimeError("baseline crop mismatch")
        base=os.path.join(self.decisions,"decision_%06d"%self.n)
        np.savez_compressed(base+"_input.npz",observed=core.astype(np.float32),
                            mean=mm.astype(np.float32),variance=vv.astype(np.float32))
        first=int(np.argmin(costs))
        atomic(base+".json",{"decision":self.n,"native_tick":int(t),"mode":"GLOBAL",
            "robot_rc":p.tolist(),"frontier_candidate_count":len(pts),
            "candidates_rc":pts.tolist(),"costs":costs.tolist(),
            "first_selected_index":first,"first_selected_rc":pts[first].tolist(),
            "prediction_time_s":float(pred_duration),"source":"MapEx original GLOBAL score_frontiers",
            "no_room_controller":True,"decision_input_npz":base+"_input.npz",
            "input_observed_sha256":sha(core),"input_mean_sha256":sha(mm),
            "input_var_sha256":sha(vv)})
        self.last_decision=self.n
        if self.last_pose is None:self.last_pose=p.astype(float)
    def plan(self,t,pose,goal,path,ppm):
        p=np.asarray(path,dtype=np.int32)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)==0:raise RuntimeError("native path invalid")
        atomic(os.path.join(self.steps,"tick_%06d_plan.json"%int(t)),
            {"native_tick":int(t),"decision":self.last_decision,
             "mode":"GLOBAL","pose_rc":np.asarray(pose).tolist(),
             "actual_selected_goal_rc":np.asarray(goal).tolist(),
             "astar_path_rc":p.tolist(),"astar_path_length_m":float(np.abs(np.diff(p,axis=0)).sum())/float(ppm)})
    def scan(self,t,pose,obs,step_duration_s,ppm):
        p=np.asarray(pose,dtype=float)
        if self.last_pose is not None:
            self.total_distance_m+=float(np.linalg.norm(p-self.last_pose))/float(ppm)
        self.last_pose=p.copy()
        atomic(os.path.join(self.steps,"tick_%06d_summary.json"%int(t)),
          {"native_tick":int(t),"decision":self.last_decision,"mode":"GLOBAL",
           "postscan_pose_rc":p.astype(int).tolist(),"cumulative_distance_m":self.total_distance_m,
           "step_duration_s":float(step_duration_s),"postscan_sha256":sha(obs)})
