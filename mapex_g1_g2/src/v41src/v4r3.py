"""V4-R3: bounded door-aware separation preprocessor above frozen V4-R2.

Inputs: ONLY one decision's observed occupancy, LaMa mean/variance and resolution.
Baseline V4-R2 is imported unchanged. No geometry templates, GT, region labels,
known building, user-drawn ROI or future snapshots enter the decision function.

Experimental: proposed virtual doorway caps are not occupied walls or verified doors.
Their provenance stays separate from observed and LaMa wall evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import numpy as np
import cv2
from scipy import ndimage as ndi

from v4r2 import _t_roles, _graph_roles, RoleParams
from mapex_room_v4.structure_engine import (
    _extract_structural, Params, ROOM, CORRIDOR, WALL, OBSTACLE
)
from mapex_room_v4.geometry_helpers import nested_bay_hypotheses

@dataclass(frozen=True)
class DoorCapParams:
    min_gap_m: float = 0.5
    max_gap_m: float = 2.6
    min_wall_arm_m: float = 1.8
    min_perpendicular_support_m: float = 1.2
    min_small_region_m2: float = 2.0
    max_small_region_m2: float = 180.0
    min_wall_boundary_fraction: float = 0.50
    min_observed_or_pred_free: float = 0.75
    max_candidates: int = 100
    max_accepted: int = 20


def _components_lines(binary: np.ndarray, orientation: str, min_length: int):
    n,lab,stat,_=cv2.connectedComponentsWithStats(binary.astype('uint8'),8)
    ans=[]
    for k in range(1,n):
        x,y,w,h,area=(int(z) for z in stat[k])
        if orientation=='H' and w>=min_length and w>=h*3.0 and h<=8:
            ans.append((x,y,w,h))
        elif orientation=='V' and h>=min_length and h>=w*3.0 and w<=8:
            ans.append((x,y,w,h))
    return ans


def _runs1d(row):
    coords=np.flatnonzero(row)
    if not len(coords):return []
    return [g for g in np.split(coords, np.where(np.diff(coords)>1)[0]+1) if len(g)]


def _candidate_geometries(st,obs,mean, cfg:DoorCapParams, res):
    H,W=obs.shape
    lo=max(2,int(round(cfg.min_gap_m/res)))
    hi=max(lo+1,int(round(cfg.max_gap_m/res)))
    arm=max(12,int(round(cfg.min_wall_arm_m/res)))
    support=max(7,int(round(cfg.min_perpendicular_support_m/res)))
    hlines=st['horizontal'];vlines=st['vertical']
    candidates=[]
    # Work twice, with a transpose of the SAME occupancy grid, to avoid treating
    # horizontal/vertical structural openings differently.
    for transposed in (False,True):
        hp=(vlines.T if transposed else hlines)
        vp=(hlines.T if transposed else vlines)
        ob=(obs.T if transposed else obs)
        pr=(mean.T if transposed else mean)
        hh,ww=ob.shape
        for x,y,w,h in _components_lines(hp,'H',arm):
            ry=y+h//2
            if ry<4 or ry>hh-5:continue
            for endpoint,side in ((x,-1),(x+w-1,1)):
                for d in range(lo,hi+1):
                    tx=endpoint+side*d
                    if tx<4 or tx>=ww-4:break
                    if not vp[max(0,ry-2):min(hh,ry+3),tx].any():continue
                    # Candidate closes a line arm to a real, supported perpendicular.
                    above=vp[max(0,ry-35):ry+1,max(0,tx-2):min(ww,tx+3)]
                    below=vp[ry:min(hh,ry+36),max(0,tx-2):min(ww,tx+3)]
                    sup_above=int(np.max(above.sum(axis=0))) if above.size else 0
                    sup_below=int(np.max(below.sum(axis=0))) if below.size else 0
                    longest=max(sup_above,sup_below)
                    if longest<support:continue
                    x0,x1=sorted((endpoint,tx))
                    a=ob[ry,x0+1:x1]
                    b=pr[ry,x0+1:x1]
                    if not a.size:continue
                    traversable=((a==0)|((a==1)&(b<.65))).mean()
                    if traversable<cfg.min_observed_or_pred_free:continue
                    # Exclude already-complete cuts.
                    if st['barrier'].T[ry,x0+1:x1].all() if transposed else st['barrier'][ry,x0+1:x1].all():continue
                    coords=((x0,ry,x1,ry) if not transposed else (ry,x0,ry,x1))
                    candidates.append({'line_xyxy':coords,'gap_m':round(float(d*res),3),
                                       'perpendicular_support_m':round(float(longest*res),3),
                                       'gap_traversable':round(float(traversable),3),
                                       'axis':'V' if transposed else 'H'})
                    break
    # deduplicate multi-row or multi-column versions of the same segment
    candidates.sort(key=lambda c:(-c['perpendicular_support_m'],c['gap_m'],c['line_xyxy']))
    filtered=[]
    for c in candidates:
        a=c['line_xyxy']
        if any(c['axis']==d['axis'] and abs(a[0]-d['line_xyxy'][0])<5 and
               abs(a[1]-d['line_xyxy'][1])<5 and abs(a[2]-d['line_xyxy'][2])<5 and
               abs(a[3]-d['line_xyxy'][3])<5 for d in filtered):continue
        filtered.append(c)
    return filtered[:cfg.max_candidates]


def _accepted_caps(st,obs,mean,cfg:DoorCapParams,res):
    candidates=_candidate_geometries(st,obs,mean,cfg,res)
    H,W=obs.shape
    barrier=st['barrier'].copy()
    accepted=[]
    diagnostics=[]
    base_count,base_lab,_,_=cv2.connectedComponentsWithStats((~barrier).astype('uint8'),4)
    boundary_wall=ndi.binary_dilation(st['wall'],iterations=2)
    maxsmall=int(cfg.max_small_region_m2/(res*res));minsmall=int(cfg.min_small_region_m2/(res*res))
    for c in candidates:
        x0,y0,x1,y1=c['line_xyxy']
        proposal=barrier.copy()
        if y0==y1:
            proposal[max(0,y0-1):min(H,y0+2),x0:x1+1]=True
            p=(max(0,y0-4),min(H-1,y0+4),(x0+x1)//2)
            coords1=(p[0],p[2]);coords2=(p[1],p[2])
        else:
            proposal[y0:y1+1,max(0,x0-1):min(W,x0+2)]=True
            p=(max(0,x0-4),min(W-1,x0+4),(y0+y1)//2)
            coords1=(p[2],p[0]);coords2=(p[2],p[1])
        if not st['floor'][coords1] or not st['floor'][coords2]:
            diagnostics.append({**c,'accepted':False,'reason':'NO_FLOOR_ON_BOTH_SIDES'});continue
        n,lbl,stats,_=cv2.connectedComponentsWithStats((~proposal).astype('uint8'),4)
        a=int(lbl[coords1]);b=int(lbl[coords2])
        if a==0 or b==0 or a==b:
            diagnostics.append({**c,'accepted':False,'reason':'NO_SPATIAL_SEPARATION'});continue
        areas=[int(stats[i,cv2.CC_STAT_AREA]) for i in (a,b)]
        small=a if areas[0]<=areas[1] else b
        ar=min(areas)
        if not (minsmall<=ar<=maxsmall):
            diagnostics.append({**c,'accepted':False,'reason':'REGION_AREA_OUTSIDE_RANGE','small_area_cells':ar});continue
        mask=lbl==small
        edge=mask & ~ndi.binary_erosion(mask,iterations=1)
        wall_support=float((boundary_wall & edge).sum()/max(1,int(edge.sum())))
        if wall_support<cfg.min_wall_boundary_fraction:
            diagnostics.append({**c,'accepted':False,'reason':'INSUFFICIENT_STRUCTURAL_BOUNDARY','wall_fraction':round(wall_support,3)});continue
        xy=np.where(mask)
        sy,sx=xy;box=[int(sx.min()),int(sy.min()),int(sx.max()+1),int(sy.max()+1)]
        # Guard against promoting narrow corridor corridors by bounding shape.
        bx,by,ex,ey=box
        aspect=max(ex-bx,ey-by)/max(1,min(ex-bx,ey-by))
        if aspect>5.0:
            diagnostics.append({**c,'accepted':False,'reason':'NARROW_LONG_REGION','aspect':round(aspect,3)});continue
        barrier=proposal
        proposal_info={**c,'accepted':True,'reason':'ROOMLIKE_CLOSED_FACE','small_area_cells':ar,
                       'wall_fraction':round(wall_support,3),'detached_bbox_xyxy':box,'aspect':round(aspect,3)}
        accepted.append(proposal_info)
        diagnostics.append(proposal_info)
        if len(accepted)>=cfg.max_accepted:break
    return barrier,accepted,diagnostics


def infer(observed,mean,variance,*,resolution_m=.1,door_config=DoorCapParams(),
          role_config=RoleParams(),snapshot_id=''):
    """Runs frozen V4-R2 roles on augmented geometry for one snapshot only."""
    obs=np.asarray(observed);mean=np.asarray(mean);var=np.asarray(variance)
    if obs.ndim!=2 or obs.shape!=mean.shape or obs.shape!=var.shape:raise ValueError('shape mismatch')
    if obs.size>1_000_000 or resolution_m<=0:raise ValueError('invalid resolution')
    if not np.isin(obs,[0,1,2]).all():raise ValueError('observed values 0,1,2 required')
    if not np.isfinite(mean).all() or not np.isfinite(var).all():raise ValueError('nonfinite map')
    if mean.min()<0 or mean.max()>1 or var.min()<0 or var.max()>1:raise ValueError('mean/variance range')
    before=hashlib.sha256(obs.tobytes()).hexdigest()
    st=_extract_structural(obs,mean,var,resolution_m,Params(virtual_gap_m=2.5,junction_reach_m=1.4))
    new_barrier,accepted,audit=_accepted_caps(st,obs,mean,door_config,resolution_m)
    st={**st,'barrier':new_barrier,'virtual':(new_barrier&~st['solid'])}
    base,lab,faces,adj,corr=_t_roles(st,obs,resolution_m)
    base,graph=_graph_roles(st,obs,resolution_m,base,role_config)
    sem=np.zeros_like(obs,dtype=np.uint8)
    sem[st['floor']]=base[st['floor']]
    sem[st['solid']&~st['furniture']]=WALL
    sem[st['furniture']]=OBSTACLE
    bays=nested_bay_hypotheses(st['horizontal'],st['vertical'],sem,st['floor'],obs)
    nested_added=[]
    for score,(x0,y0,x1,y1),left,right,obsfrac in bays:
        sub=sem[y0+2:y1-2,x0+2:x1-2]
        replace=int((sub==CORRIDOR).sum())
        sub[sub==CORRIDOR]=ROOM
        if replace:nested_added.append({'bbox_xyxy':[int(x0),int(y0),int(x1),int(y1)],'promoted_floor_cells':replace})
    # Conservative R3 invariant: accepted door caps may add ROOM evidence only.
    # Preserve original V4-R2 labels outside the newly separated, room-like
    # faces, and NEVER demote existing R2 ROOM to CORRIDOR.
    from v4r2 import infer as infer_frozen_r2
    original=infer_frozen_r2(obs,mean,var,resolution_m=resolution_m,snapshot_id=snapshot_id)
    original_sem=original['semantic']
    local=np.zeros(obs.shape,dtype=bool)
    for cap in accepted:
        x0,y0,x1,y1=cap['detached_bbox_xyxy']
        local[y0:y1,x0:x1]=True
    promoted=(sem==ROOM)&(original_sem==CORRIDOR)&local
    sem=original_sem.copy()
    sem[promoted]=ROOM
    local_added=int(promoted.sum())
    if not np.isin(sem[st['floor']],[ROOM,CORRIDOR]).all():raise AssertionError('unclassified floor')
    if before!=hashlib.sha256(obs.tobytes()).hexdigest():raise AssertionError('mutated observed')
    d={'version':'V4R3_DOOR_AWARE_PROTOTYPE','snapshot_id':snapshot_id,
       'status':'OFFLINE_SHADOW_DIAGNOSTIC_NOT_BLIND','room_floor_cells':int((sem==ROOM).sum()),
       'corridor_floor_cells':int((sem==CORRIDOR).sum()),'floor_cells':int(st['floor'].sum()),
       'proposals_count':len(audit),'caps_accepted_count':len(accepted),
       'guarded_hall_to_room_cells':local_added,
       'conservative_no_r2_room_demotions':True,
       'accepted_door_cap_hypotheses':accepted,'door_cap_audit':audit,
       'graph_regions':len(graph.get('regions',{})),'graph_edges':len(graph.get('filtered_edges',[])),
       'graph_role_changes':graph.get('changed_roles',0),'nested_bays':len(nested_added),
       'observed_sha256':before,'mean_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),
       'variance_sha256':hashlib.sha256(var.tobytes()).hexdigest(),
       'warnings':['Virtual caps are door partition hypotheses NOT observed wall','Hospital E49 was inspected during method design; results are in-sample','R2 original left unchanged']}
    return {'semantic':sem,'floor':st['floor'],'virtual_barriers':st['virtual'],
            'door_cap_barriers':new_barrier & ~_extract_structural(obs,mean,var,resolution_m,Params(virtual_gap_m=2.5,junction_reach_m=1.4))['barrier'],
            'report':d}
