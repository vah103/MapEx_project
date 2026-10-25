"""V4-R4 prediction-first room hypotheses (offline research prototype).

Only an observed occupancy snapshot, MapEx/LaMa mean, variance and past TRACKED
hypotheses enter the inference. No GT, room coordinates, future decisions, or
E22 manual labels. R3 implementation is imported but unchanged.

Two distinct outputs:
  * frozen R3 semantic mask, unchanged
  * R4 semantic mask and tentative room hypotheses (predicted occupancy is NOT
    written into observed occupancy and virtual edges are NOT navigation walls).

Temporal matching and revision are diagnostic only, not a navigation policy.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from v4r3_frozen import infer as infer_r3

ROOM,CORRIDOR,WALL,OBSTACLE=1,2,4,3  # R2/R3 semantic convention

@dataclass(frozen=True)
class RoomFirstConfig:
    # All distances are metres. Scores are heuristics, NOT calibrated probabilities.
    wall_strong:float=.65
    min_axis_run_m:float=.5
    min_side_m:float=1.2
    max_side_m:float=16.0
    min_room_area_m2:float=2.5
    max_room_area_m2:float=155.0
    wall_band_m:float=.22
    candidate_min_score:float=.43
    candidate_min_coverage:float=.34
    min_two_sides:float=0.52
    nms_iou:float=.38
    track_iou:float=.25
    retire_after:int=3
    max_proposals_per_geometry:int=100000 # compute guard; overflow aborts rather than silently skipping rooms

# fix dataclass syntax before freeze


def _validate(obs,mean,var,conf,res):
    if obs.ndim!=2 or mean.shape!=obs.shape or var.shape!=obs.shape:
        raise ValueError('one matching 2D observed/mean/variance input required')
    if obs.size>1_000_000 or res<=0:raise ValueError('invalid resolution/grid')
    if not np.isin(obs,[0,1,2]).all():raise ValueError('observed values must be 0 FREE / 1 UNKNOWN / 2 OCCUPIED')
    if not np.isfinite(mean).all() or not np.isfinite(var).all():raise ValueError('nonfinite map')
    if mean.min()<0 or mean.max()>1 or var.min()<0 or var.max()>1:raise ValueError('out of range')
    if conf.min_side_m>=conf.max_side_m or conf.retire_after<1:raise ValueError('invalid config')


def _wall_evidence(obs,mean,var):
    # Occupied observed cells dominate. Observed FREE has zero wall evidence.
    # For unobserved cells use a soft, variance-discounted MapEx predicted wall.
    # Kept separate from occupancy map; never painted into source.
    pred=np.clip((mean-.30)/.70,0,1)*(1-.55*np.clip(var/.3334,0,1))
    w=np.where(obs==2,1.,np.where(obs==1,pred,0.)).astype('float32')
    return w


def _runs_1d(mask):
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    return [(int(a[0]),int(a[-1])+1) for a in np.split(idx,np.where(np.diff(idx)>1)[0]+1)]


def _axis_candidates(strong:np.ndarray,res:float,cfg:RoomFirstConfig):
    # Detect observed/predicted aligned wall stretches rather than rely solely
    # on closed-wall connected components. No room template or GT.
    k=max(4,round(cfg.min_axis_run_m/res))
    horizontal=cv2.morphologyEx(strong.astype('uint8'),cv2.MORPH_OPEN,np.ones((1,k),np.uint8))>0
    vertical=cv2.morphologyEx(strong.astype('uint8'),cv2.MORPH_OPEN,np.ones((k,1),np.uint8))>0
    h,w=strong.shape
    # More permissive: one short row segment can define a potential boundary.
    # A 1-D peak consolidates lines 1-2 grid cells apart.
    hp=horizontal.sum(axis=1).astype(float)
    vp=vertical.sum(axis=0).astype(float)
    def peaks(v,shortdim):
        smooth=ndi.gaussian_filter1d(v,1.35)
        mask=(smooth>=max(k*.48,shortdim*.055))
        groups=_runs_1d(mask)
        return sorted({int(np.argmax(smooth[a:b])+a) for a,b in groups}
                 | {int(i) for i in np.where((smooth[1:-1]>smooth[:-2]) & (smooth[1:-1]>=smooth[2:]) & (smooth[1:-1]>=k*1.15))[0]+1})
    xs=peaks(vp,h)
    ys=peaks(hp,w)
    # Include image border only if physically supported by observed/predicted wall;
    # this is not an assumption that the image border is a building wall.
    return xs,ys,{'horizontal':horizontal,'vertical':vertical,'h_support':hp,'v_support':vp}


def _integral(arr):
    return cv2.integral(arr.astype('float32'))


def _boxsum(ii,x0,y0,x1,y1):
    return float(ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0])


def _iou(a,b):
    ax0,ay0,ax1,ay1=a;bx0,by0,bx1,by1=b
    iw=max(0,min(ax1,bx1)-max(ax0,bx0));ih=max(0,min(ay1,by1)-max(ay0,by0))
    if not iw or not ih:return 0.
    inter=iw*ih; return inter/((ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter)


def _overlap_small(a,b):
    ax0,ay0,ax1,ay1=a;bx0,by0,bx1,by1=b
    iw=max(0,min(ax1,bx1)-max(ax0,bx0));ih=max(0,min(ay1,by1)-max(ay0,by0))
    return iw*ih/max(1,min((ax1-ax0)*(ay1-ay0),(bx1-bx0)*(by1-by0)))


def _propose(obs,mean,var,res,cfg:RoomFirstConfig):
    h,w=obs.shape
    evidence=_wall_evidence(obs,mean,var)
    strong=(evidence>=cfg.wall_strong)
    xs,ys,axes=_axis_candidates(strong,res,cfg)
    band=max(1,round(cfg.wall_band_m/res))
    # Integrate a *maximum* across a narrow band to tolerate map-wall thickness.
    hline=ndi.maximum_filter1d(evidence,size=band*2+1,axis=0)
    vline=ndi.maximum_filter1d(evidence,size=band*2+1,axis=1)
    hi=_integral(hline);vi=_integral(vline)
    observed_occ=_integral((obs==2).astype('float32'))
    seen_free=_integral((obs==0).astype('float32'))
    usable=_integral(((obs==0)|((obs==1)&(mean<.65))).astype('float32'))
    conf_pred=_integral(((obs==1)&(evidence>=cfg.wall_strong)).astype('float32'))
    minspan=max(8,round(cfg.min_side_m/res));maxspan=round(cfg.max_side_m/res)
    minarea=cfg.min_room_area_m2/res**2;maxarea=cfg.max_room_area_m2/res**2
    proposed=[];attempts=0
    for a,x0 in enumerate(xs):
        for x1 in xs[a+1:]:
            dx=x1-x0
            if dx<minspan:continue
            if dx>maxspan:break
            for b,y0 in enumerate(ys):
                for y1 in ys[b+1:]:
                    dy=y1-y0;area=dx*dy
                    if dy<minspan:continue
                    if dy>maxspan:break
                    if area<minarea or area>maxarea:continue
                    # Budget limits only the number of rectangle combinations to
                    # evaluate; no limit on how many rooms can be output.
                    attempts+=1
                    if attempts>cfg.max_proposals_per_geometry:raise RuntimeError("candidate budget exhausted: cannot claim exhaustive room hypotheses")
                    top=_boxsum(hi,x0,y0,x1,y0+1)/dx
                    bottom=_boxsum(hi,x0,y1,x1,y1+1)/dx
                    left=_boxsum(vi,x0,y0,x0+1,y1)/dy
                    right=_boxsum(vi,x1,y0,x1+1,y1)/dy
                    sides=[top,bottom,left,right]
                    ranked=sorted(sides,reverse=True)
                    if ranked[1]<cfg.min_two_sides:continue
                    enclosure=sum(sides)/4
                    if enclosure<cfg.candidate_min_coverage:continue
                    inner=(x0+2,y0+2,x1-1,y1-1)
                    if inner[0]>=inner[2] or inner[1]>=inner[3]:continue
                    ia=(inner[2]-inner[0])*(inner[3]-inner[1])
                    traversable=_boxsum(usable,*inner)/ia
                    occupied=_boxsum(observed_occ,*inner)/ia
                    if traversable<.66 or occupied>.10:continue
                    # Prioritise a 3-side room but accept two strong sides if
                    # MapEx gives a plausible interior. Penalise long corridors.
                    aspect=max(dx,dy)/max(1,min(dx,dy))
                    aspect_term=np.clip((aspect-2.0)/4.,0,1)
                    three=sum(z>=.50 for z in sides)
                    half_open=(ranked[0]+ranked[1]+.5*ranked[2]+.25*ranked[3])/2.75
                    score=.56*half_open+.21*enclosure+.20*traversable + .08*min(three,3)/3 - .20*aspect_term -.16*occupied
                    score=float(np.clip(score,0,1))
                    if score<cfg.candidate_min_score:continue
                    # Distinguish observed evidence from prediction. Mean >= threshold
                    # on a partially unknown wall does not imply verified existence.
                    r=1
                    border=np.zeros(obs.shape,'uint8')
                    cv2.rectangle(border,(x0,y0),(x1,y1),1,thickness=max(1,r))
                    pixels=border.astype(bool)
                    obs_ratio=float(np.mean(obs[pixels]==2))
                    pred_ratio=float(np.mean((obs[pixels]==1)&(evidence[pixels]>=cfg.wall_strong)))
                    proposed.append({'bbox':[int(x0),int(y0),int(x1),int(y1)],
                         'score':round(score,4),'sides':[round(float(z),3) for z in sides],
                         'traversable_fraction':round(float(traversable),3),
                         'boundary_observed_fraction':round(obs_ratio,3),
                         'boundary_predicted_fraction':round(pred_ratio,3),
                         'evidence_type':'PREDICTED_HYPOTHESIS','aspect':round(float(aspect),2)})
                if attempts>cfg.max_proposals_per_geometry:raise RuntimeError("candidate budget exhausted: cannot claim exhaustive room hypotheses")
            if attempts>cfg.max_proposals_per_geometry:raise RuntimeError("candidate budget exhausted: cannot claim exhaustive room hypotheses")
        if attempts>cfg.max_proposals_per_geometry:raise RuntimeError("candidate budget exhausted: cannot claim exhaustive room hypotheses")
    proposed.sort(key=lambda q:(-q['score'],q['bbox']))
    # Avoid redundant overlapping templates. Detectors are free to add new
    # hypotheses as area grows: no fixed room count and no world-layout prior.
    out=[]
    for r in proposed:
        if any(_iou(r['bbox'],k['bbox'])>cfg.nms_iou or _overlap_small(r['bbox'],k['bbox'])>.82 for k in out):continue
        out.append(r)
    return out,{'axis_x_count':len(xs),'axis_y_count':len(ys),'candidates_evaluated':attempts,'raw_proposals':len(proposed),'selected_hypotheses':len(out)}


class RoomHypothesisTracker:
    """Causal ID association using proposals from 0..t only.

    An unmatched hypothesis gets a missing count and can be retired; matched
    hypotheses update position and confidence. Contradicting observed FREE
    on a previously predicted boundary weakens scores, not observational facts.
    Split/merge are logged as potential lineage events, not ground truth.
    """
    def __init__(self,config:RoomFirstConfig=RoomFirstConfig()):
        self.cfg=config;self.tracks={};self.next_id=1;self.step=0;self.history=[]

    def update(self,proposals,obs,mean,var):
        self.step+=1
        current=sorted([t for t in self.tracks.values() if t['status']!='RETIRED'],key=lambda z:z['track_id'])
        matched={};touched=set();track_indexes=[]
        if current and proposals:
            cost=np.ones((len(current),len(proposals)),dtype=float)
            for i,t in enumerate(current):
                for j,p in enumerate(proposals):cost[i,j]=1.-_iou(t['bbox'],p['bbox'])
            rr,cc=linear_sum_assignment(cost)
            for i,j in zip(rr,cc):
                if 1-cost[i,j]>=self.cfg.track_iou:
                    tid=current[i]['track_id'];matched[j]=tid;touched.add(tid)
        events=[]
        for j,p in enumerate(proposals):
            bb=p['bbox'];x0,y0,x1,y1=bb
            if j in matched:
                t=self.tracks[matched[j]];prev_score=t['confidence_heuristic']
                previous_bbox=t["bbox"]
                revised=(_iou(previous_bbox,bb)<.77)
                t.update(bbox=bb,last_seen=self.step,missing=0,
                    confidence_heuristic=round(.55*prev_score+.45*p['score'],4),
                    status='REVISED' if revised else ('SUPPORTED' if p['boundary_observed_fraction']>.42 else 'PREDICTED'),
                    score_now=p['score'],boundary_observed_fraction=p['boundary_observed_fraction'],
                    boundary_predicted_fraction=p['boundary_predicted_fraction'])
                tid=t['track_id']
                if revised:events.append({'type':'BOUNDARY_REVISION','track_id':tid,
                    'previous_bbox':previous_bbox,'new_bbox':bb})
            else:
                tid=self.next_id;self.next_id+=1
                t={'track_id':tid,'bbox':bb,'first_seen':self.step,'last_seen':self.step,
                   'missing':0,'confidence_heuristic':p['score'],'score_now':p['score'],
                   'status':'SUPPORTED' if p['boundary_observed_fraction']>.42 else 'PREDICTED',
                   'boundary_observed_fraction':p['boundary_observed_fraction'],
                   'boundary_predicted_fraction':p['boundary_predicted_fraction'],
                   'parent_ids':[]}
                ancestors=[k['track_id'] for k in current if _iou(k['bbox'],bb)>.25 and k['track_id'] not in touched]
                if ancestors:
                    t['parent_ids']=ancestors
                    events.append({'type':'CANDIDATE_SPLIT_OR_REPLACEMENT','new_id':tid,'related_previous_ids':ancestors})
                self.tracks[tid]=t
            t['matched_proposal_index']=j
        for t in current:
            if t['track_id'] in touched:continue
            t['missing']+=1
            t['confidence_heuristic']=round(t['confidence_heuristic']*.78,4)
            if t['missing']>=self.cfg.retire_after:
                t['status']='RETIRED'
                events.append({'type':'RETIRED_UNSUPPORTED','track_id':t['track_id']})
            else:t['status']='UNMATCHED'
        active=[dict(t) for t in sorted(self.tracks.values(),key=lambda x:x['track_id']) if t['last_seen']==self.step]
        self.history.append({'step':self.step,'active_ids':[t['track_id'] for t in active],
          'retired_ids':[t['track_id'] for t in self.tracks.values() if t['status']=='RETIRED'],
          'events':events})
        return active,events


def _verification_opportunities(hypotheses, obs, mean, frontiers, resolution_m):
    """Rank exploratory verification interest, NOT safe paths or certified viewpoints.

    Higher unknown share and room-support score mean more unresolved room evidence;
    an actual MapEx frontier is preferred as a *candidate goal*, without any
    visibility, reachability or safety claim. Pose is not needed for this proxy.
    """
    out=[]
    for p in hypotheses:
        x0,y0,x1,y1=p['bbox'];patch=obs[y0+1:y1,x0+1:x1]
        if not patch.size:continue
        unknown=float((patch==1).mean())
        if unknown<=.005:continue
        cy=(y0+y1)/2;cx=(x0+x1)/2
        near=None;dist=None
        for f in frontiers or []:
            if not isinstance(f,dict) or not all(k in f for k in ('row','col')):continue
            dr=float(f['row'])-cy;dc=float(f['col'])-cx
            d=(dr*dr+dc*dc)**.5*resolution_m
            if dist is None or d<dist:dist=d;near=f.get('id')
        score=(float(p['confidence_heuristic'])*(.25+.75*unknown)
               /(1+.035*dist if dist is not None else 1.))
        out.append({'track_id':p['track_id'],'room_bbox':p['bbox'],
            'unknown_fraction':round(unknown,4),
            'frontier_id':near,'euclidean_frontier_distance_m':None if dist is None else round(dist,3),
            'verification_interest_heuristic':round(score,4),
            'status':'RANK_ONLY_NOT_NAVIGATION_COMMAND'})
    return sorted(out,key=lambda t:-t['verification_interest_heuristic'])


def infer(observed,mean,variance,*,resolution_m:float=.1,tracker:RoomHypothesisTracker|None=None,
          config:RoomFirstConfig=RoomFirstConfig(),snapshot_id:str='',frontiers=None):
    obs=np.asarray(observed);mean=np.asarray(mean);var=np.asarray(variance)
    _validate(obs,mean,var,config,resolution_m)
    sha=hashlib.sha256(obs.tobytes()).hexdigest()
    frozen=infer_r3(obs,mean,var,resolution_m=resolution_m,snapshot_id=snapshot_id)
    original=frozen['semantic'].copy()
    proposals,diagnostics=_propose(obs,mean,var,resolution_m,config)
    active,events=(tracker.update(proposals,obs,mean,var) if tracker is not None
                   else ([dict(p,track_id=i+1,status='UNTRACKED',confidence_heuristic=p['score']) for i,p in enumerate(proposals)],[]))
    # Map every proposal onto only R3 traversable FLOOR; never reclassify known
    # occupied cells. This deliberately encourages early room hypotheses.
    new=original.copy();proposed_room=np.zeros(obs.shape,np.int32)
    downgraded=0
    for p in sorted(active,key=lambda z:z['confidence_heuristic']):
        x0,y0,x1,y1=p['bbox'];mask=np.zeros(obs.shape,bool)
        mask[y0+1:y1,x0+1:x1]=True
        crop=obs[y0+1:y1,x0+1:x1]
        if not crop.size:continue
        unknown_fraction=float((crop==1).mean())
        baseline=frozen['semantic'][y0+1:y1,x0+1:x1]
        existing_hall_fraction=float((baseline==CORRIDOR).mean())
        # R4 is aggressive when MapEx's prediction is still unobserved.
        # As the robot actually sees the area, R3's graph can outweigh an
        # *unconfirmed* rectangle with weak observed wall support.
        # This is a conservative, revisable circulation prior, not a ground
        # truth label or proof that the hypothesis is false. For an already
        # well-observed corridor, R3's topology wins over an overlapping
        # speculative rectangle. In mostly unknown space R4 remains bold.
        down=(unknown_fraction<.24 and existing_hall_fraction>.24)
        p['unknown_fraction']=round(unknown_fraction,4)
        p['baseline_corridor_fraction']=round(existing_hall_fraction,4)
        p['prediction_state']='DOWNGRADED_BY_OBSERVED_CIRCULATION_PRIOR' if down else 'ACTIVE_HYPOTHESIS'
        if down:
            downgraded+=1
            if tracker is not None and p['track_id'] in tracker.tracks:
                tracker.tracks[p['track_id']]['status']='DOWNGRADED'
            continue
        extra=mask & (new==CORRIDOR) & frozen['floor']
        proposed_room[extra]=p['track_id']
        new[extra]=ROOM
    if hashlib.sha256(obs.tobytes()).hexdigest()!=sha:raise AssertionError('changed observed map')
    assert np.all(np.isin(new[frozen['floor']],[ROOM,CORRIDOR]))
    assert not np.any((original==ROOM)&(new==CORRIDOR))
    changed=int(((original==CORRIDOR)&(new==ROOM)).sum())
    opportunities=_verification_opportunities(active,obs,mean,frontiers,resolution_m)
    return {'semantic_r3':original,'semantic_r4':new,'hypothesis_mask':proposed_room,
        'hypotheses':active,'floor':frozen['floor'],'events':events,
        'verification_opportunities':opportunities,
        'report':dict(snapshot_id=snapshot_id,version='V4R4_PREDICTION_FIRST_OFFLINE',
                      hypothesis_count=len(proposals),active_count=len(active),
                      verification_opportunity_count=len(opportunities),
                      promoted_floor_cells=changed,downweighted_hypotheses=downgraded,geometry_diagnostics=diagnostics,
                      R3_semantic_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                      observed_sha256=sha,trajectory_uses_only_past_and_current=True,
                      predicted_rooms_not_observed_walls=True,
                      warnings=['Confidence is heuristic, not calibrated probability',
                                'Verification interest is a proxy only; no reachability, visibility or exploration gain demonstrated'])}
