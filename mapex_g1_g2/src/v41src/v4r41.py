"""V4-R4.1: frozen R3 plus conservative MapEx room hypotheses.

Distinct from permissive V4-R4: the structural graph and ROOM/CORRIDOR mask of
R3 are the anchor. Tentative rooms require positive, prediction-backed wall
structure, new unknown space, and no contradiction of observed free corridors.
All distances/thresholds are generic, not geometry templates for MX072.
Only current snapshot and past tracker state are inputs; no GT or future maps.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import hashlib
import numpy as np
import cv2
from v4r3_frozen import infer as infer_r3
from v4r4_generator_reference import _propose, RoomFirstConfig, RoomHypothesisTracker, _iou

ROOM,CORRIDOR,WALL,OBSTACLE = 1,2,4,3

@dataclass(frozen=True)
class ControlledConfig:
    # Controls admission, NOT a target number of rooms. Not trained/calibrated.
    min_supported_sides: int=3
    side_support: float=.55
    min_peak_side: float=.75
    min_boundary_predicted_fraction: float=.22
    min_unknown_fraction: float=.25
    min_novel_unknown_cells: int=40
    max_r3_room_share: float=.65
    max_known_corridor_share: float=.30
    max_boundary_observed_free_share: float=.24
    max_span_m: float=10.0
    max_area_m2: float=80.0
    min_score: float=.74
    max_mutual_novel_overlap: float=.35


def _mask_for_box(shape,box,margin=2):
    x0,y0,x1,y1=box
    mask=np.zeros(shape,np.uint8)
    if x1-x0>2*margin and y1-y0>2*margin:
        mask[y0+margin:y1-margin,x0+margin:x1-margin]=1
    return mask.astype(bool)


def _admit(c,obs,base,floor,res,cfg):
    x0,y0,x1,y1=c['bbox'];sides=c['sides']
    item={**c,'status':'REJECTED','reason':None}
    w=x1-x0;h=y1-y0
    if max(w,h)*res>cfg.max_span_m or w*h*res**2>cfg.max_area_m2:
        item['reason']='OVERSIZED_NONLOCAL_REGION';return item,None
    if sum(s>=cfg.side_support for s in sides)<cfg.min_supported_sides or max(sides)<cfg.min_peak_side:
        item['reason']='INSUFFICIENT_STRUCTURAL_SIDES';return item,None
    if c['score']<cfg.min_score or c['boundary_predicted_fraction']<cfg.min_boundary_predicted_fraction:
        item['reason']='LOW_PREDICTION_WALL_SUPPORT';return item,None
    mask=_mask_for_box(obs.shape,c['bbox'])
    inner=obs[mask]
    if not len(inner):item['reason']='EMPTY_INTERIOR';return item,None
    unknown=float(np.mean(inner==1))
    if unknown<cfg.min_unknown_fraction:
        item['reason']='ALREADY_OBSERVED_NO_SPECULATION_NEEDED';return item,None
    old=base[mask]
    r3room=float(np.mean(old==ROOM))
    if r3room>cfg.max_r3_room_share:
        item['reason']='REDUNDANT_WITH_R3_ROOM';return item,None
    known_hall=float(np.mean((old==CORRIDOR)&(inner==0)))
    if known_hall>cfg.max_known_corridor_share:
        item['reason']='CROSSES_OBSERVED_CORRIDOR';return item,None
    bd=np.zeros(obs.shape,np.uint8)
    cv2.rectangle(bd,(x0,y0),(x1,y1),1,1)
    border=bd.astype(bool)
    seen_free=float(((obs==0)&border).sum()/max(1,int(border.sum())))
    if seen_free>cfg.max_boundary_observed_free_share:
        item['reason']='PROPOSED_BOUNDARY_CONTRADICTS_OBSERVED_FREE';return item,None
    # Only reclassify UNKNOWN estimated-traversable pixels that were R3 CORRIDOR.
    # Observed-free corridor connectivity is thereby provably unchanged.
    novel=mask&(obs==1)&(base==CORRIDOR)&floor
    novel_cells=int(novel.sum())
    if novel_cells<cfg.min_novel_unknown_cells:
        item['reason']='NO_MEANINGFUL_UNEXPLORED_ROOM_EVIDENCE';return item,None
    item.update(status='ADMISSIBLE',reason='THREE_OR_MORE_SUPPORTED_SIDES_NEW_UNKNOWN_ROOM',
                unknown_fraction=round(unknown,4),r3_room_fraction=round(r3room,4),
                known_corridor_fraction=round(known_hall,4),boundary_seen_free=round(seen_free,4),
                candidate_novel_unknown_cells=novel_cells)
    return item,novel


def infer(observed,mean,variance,*,resolution_m=.1,tracker:RoomHypothesisTracker|None=None,
          config:ControlledConfig=ControlledConfig(),proposal_config:RoomFirstConfig=RoomFirstConfig(),
          snapshot_id=''):
    obs=np.asarray(observed);mean=np.asarray(mean);var=np.asarray(variance)
    # Frozen R3 performs full shape/value/resolution validation.
    before=tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in (obs,mean,var))
    baseline=infer_r3(obs,mean,var,resolution_m=resolution_m,snapshot_id=snapshot_id)
    base=baseline['semantic'];floor=baseline['floor']
    candidates,gen_report=_propose(obs,mean,var,resolution_m,proposal_config)
    accepted=[];accepted_masks=[];audit=[];reasons=Counter()
    for c in candidates:
        record,novel=_admit(c,obs,base,floor,resolution_m,config)
        if novel is not None:
            # NMS on the NEW room evidence, not just rectangular proposal extent.
            redundant=False
            for previous in accepted_masks:
                intersection=int(np.count_nonzero(novel&previous))
                if intersection>config.max_mutual_novel_overlap*max(1,min(int(novel.sum()),int(previous.sum()))):
                    redundant=True;break
            if redundant:
                record.update(status='REJECTED',reason='DUPLICATE_NOVEL_ROOM')
            else:
                accepted.append(dict(record));accepted_masks.append(novel)
        audit.append(record);reasons[record['reason']]+=1
    # Tracker sees accepted candidates only. No expired or overlapping hypothesis
    # is carried into a current mask simply because of earlier evidence.
    if tracker is not None:
        active,events=tracker.update(accepted,obs,mean,var)
    else:
        active=[dict(a,track_id=i+1,status='PREDICTED',confidence_heuristic=a['score']) for i,a in enumerate(accepted)]
        events=[]
    selected=np.zeros(obs.shape,bool)
    idmask=np.zeros(obs.shape,np.int32)
    records=[]
    for hyp in active:
        bbox=hyp['bbox']
        idx=next((j for j,a in enumerate(accepted) if a['bbox']==bbox),None)
        if idx is None:raise AssertionError('tracker matched a noncurrent hypothesis')
        mask=accepted_masks[idx]&~selected
        if not mask.any():continue
        selected|=mask
        idmask[mask]=hyp['track_id']
        records.append({**hyp,'added_unknown_cells':int(mask.sum()),
            'admission_reason':accepted[idx]['reason'],
            'boundary_sides':accepted[idx]['sides']})
    semantic=base.copy()
    semantic[selected]=ROOM
    after=tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in (obs,mean,var))
    assert after==before,'mutated MapEx input'
    assert not np.any((base==ROOM)&(semantic!=ROOM))
    assert not np.any((obs==0)&(semantic!=base))
    assert not np.any((base>=3)&(semantic!=base))
    assert np.isin(semantic[floor],[ROOM,CORRIDOR]).all()
    report={'version':'V4R41_R3_PLUS_CONTROLLED_PREDICTION_OFFLINE',
      'snapshot_id':snapshot_id,'geometry_evaluated':gen_report['candidates_evaluated'],
      'permissive_r4_candidates':len(candidates),'admitted_hypotheses':len(accepted),
      'active_hypotheses':len(records),'extra_room_unknown_cells':int(selected.sum()),
      'rejection_reasons':dict(reasons),'event_count':len(events),
      'input_sha256':dict(zip(('observed','mean','variance'),before)),
      'frozen_r3_semantic_sha256':hashlib.sha256(base.tobytes()).hexdigest(),
      'accuracy_validated':False,'observed_free_unchanged':True,
      'no_future_or_gt_used':True,'potential_false_rooms_remain':True}
    return {'semantic_r3':base.copy(),'semantic_r41':semantic,'hypothesis_mask':idmask,
      'hypotheses':records,'events':events,'audit':audit,'floor':floor,'report':report}
