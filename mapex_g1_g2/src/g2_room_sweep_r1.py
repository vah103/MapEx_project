"""G2-SWEEP-R1: Room-relative monotone frontier sweep, receding-horizon.
ONLINE input: current G1 mask, official room-WFD candidate goals/costs, native
weighted A* cost grid and robot pose. NO GT, no specific room direction,
no future/previous recorded map, no offline goal list. GLOBAL unchanged.
"""
from __future__ import print_function
import time
import numpy as np
import pyastar2d

def _path_length(cost, source, goal):
    a=np.asarray(source,np.int32);b=np.asarray(goal,np.int32)
    if not (np.isfinite(cost[tuple(a)]) and np.isfinite(cost[tuple(b)])):
        return float('inf')
    p=pyastar2d.astar_path(np.ascontiguousarray(cost,dtype=np.float32),
                           a,b,allow_diagonal=False)
    return float('inf') if p is None else float(len(p)-1)

def room_axis(roommask):
    """Principal inertia axis from current predicted ROOM (no heading preset)."""
    yy,xx=np.nonzero(np.asarray(roommask,dtype=bool))
    if len(yy)<2:raise ValueError('SWEEP_ROOM_TOO_SMALL')
    xy=np.stack((yy.astype(np.float64),xx.astype(np.float64)),axis=1)
    center=xy.mean(axis=0)
    covariance=np.dot((xy-center).T,(xy-center))/max(1,len(xy))
    eigval,eigvec=np.linalg.eigh(covariance)
    vec=eigvec[:,int(np.argmax(eigval))]
    # fix PCA sign deterministically to make same-epoch runs reproducible.
    if vec[int(np.argmax(np.abs(vec)))]<0:vec=-vec
    return vec,center,eigval

def select_local(robot,goals,original_costs,navcore,roommask,prior=None,
                 cell_m=.1,room_id=None):
    """Choose FIRST frontier along a persistent sweep; current frontiers replan.

    On room entry: derive current G1 room's principal axis. Take nearer
    of two currently official WFD extremities to choose sweep orientation.
    Then choose the next frontier with lowest forward projection. Each new
    LiDAR epoch can remove and introduce frontier goals. Progress means
    coordinate of previous dispatched goal, never hidden/future path.
    If no CURRENT frontier is ahead in original direction, reverse exactly
    once to clear surviving residuals. After that, prefer original MapEx
    for any residual (avoid endless sweeping ping-pong).
    """
    started=time.time()
    pts=np.asarray(goals,dtype=np.int32).reshape((-1,2))
    c=np.asarray(original_costs,dtype=np.float64).reshape(-1)
    cost=np.ascontiguousarray(navcore,dtype=np.float32)
    mask=np.asarray(roommask,bool)
    robot=np.asarray(robot,dtype=np.int32)
    n=len(pts)
    if not n or n!=len(c) or cost.shape!=mask.shape or len(cost.shape)!=2:
        raise ValueError('SWEEP_INPUT_ALIGNMENT')
    if not np.isfinite(c).all():raise ValueError('SWEEP_BAD_ORIGINAL_COSTS')
    baseline=int(np.argmin(c))
    d=np.asarray([_path_length(cost,robot,p) for p in pts])
    valid=np.flatnonzero(np.isfinite(d))
    if len(valid)==0:
        return baseline,prior,dict(enabled=False,reason='NO_REACHABLE_FRONTIERS',
                                   first_index=baseline,baseline_index=baseline)
    new_room=prior is None or prior.get('room_id')!=room_id
    if new_room:
        axis,center,values=room_axis(mask)
        projections=np.dot(pts-center,axis)
        idx_lo=int(valid[np.argmin(projections[valid])])
        idx_hi=int(valid[np.argmax(projections[valid])])
        # Both ends are official current WFD frontiers, and distances are
        # native original A*, so starting end is scene-relative/geometry-only.
        # Orientation encodes lo->hi or hi->lo without global x/y assumptions.
        first_lo=(d[idx_lo],c[idx_lo],idx_lo)
        first_hi=(d[idx_hi],c[idx_hi],idx_hi)
        sign=1 if first_lo<=first_hi else -1
        sweep={
          'room_id':int(room_id) if room_id is not None else None,
          'axis':axis.tolist(),'center':center.tolist(),
          'sign':int(sign),'phase':0,'last_projection':None,
          'entry_robot_rc':robot.tolist(),
          'entry_ends':[idx_lo,idx_hi],
          'entry_start_distances_m':[round(d[idx_lo]*cell_m,4),round(d[idx_hi]*cell_m,4)],
          'axis_eigenvalues':values.tolist(),
        }
    else:
        sweep=dict(prior)
        axis=np.asarray(sweep['axis'],float)
        center=np.asarray(sweep['center'],float)
    forward=np.asarray(np.dot(pts-center,axis)*sweep['sign'],dtype=float)
    last=sweep['last_projection']
    # Grid-resolution rounding only: if an endpoint was already attempted,
    # never choose a point strictly behind it during the same forward pass.
    # Avoid a scenario-specific hysteresis distance.
    if last is None:
        eligible=valid
    else:
        eligible=valid[forward[valid]>=float(last)-1e-7]
    selected_reason='FORWARD_SWEEP'
    if len(eligible)==0:
        if sweep['phase']==0:
            # All surviving frontier goals are behind the last dispatched goal.
            # A single cleanup reversal is permitted rather than incorrectly
            # declaring a room complete with leftover known FRONTIER.
            sweep['phase']=1
            sweep['sign']*=-1
            forward=-forward
            last=None
            eligible=valid
            selected_reason='ONE_CLEANUP_REVERSAL'
        else:
            # Keep room frontier exhaustion semantics untouched. Original
            # scorer handles residual after at most one full reversal.
            selected=baseline
            sweep['phase']=2
            audit=dict(enabled=True,room_id=sweep['room_id'],phase=2,
                first_index=int(selected),baseline_index=int(baseline),
                reason='MAPEX_RESIDUAL_FALLBACK',remaining=int(len(valid)),
                duration_s=round(time.time()-started,4))
            return selected,sweep,audit
    # Lexicographic monotone front-to-back stripe ordering. Tie-break uses
    # original native path and original MapEx score (never goal index only).
    selected=int(min(eligible,key=lambda i:
        (round(float(forward[i]),7),float(d[i]),float(c[i]),int(i))))
    sweep['last_projection']=float(forward[selected])
    sweep['last_target_rc']=pts[selected].tolist()
    audit=dict(enabled=True,room_id=sweep['room_id'],phase=sweep['phase'],
        first_index=selected,baseline_index=baseline,
        reason=selected_reason,changed_goal=bool(selected!=baseline),
        selected_forward_projection=float(forward[selected]),
        entry_ends=sweep.get('entry_ends'),
        entry_start_distances_m=sweep.get('entry_start_distances_m'),
        axis_rc=sweep['axis'],sign=sweep['sign'],
        previous_projection=None if last is None else float(last),
        candidate_projection=[round(float(x),4) for x in forward],
        candidate_path_m=[None if not np.isfinite(x) else round(float(x*cell_m),4) for x in d],
        n_forward_candidates=len(eligible),n_total=n,
        duration_s=round(time.time()-started,4),no_gt_future=True)
    return selected,sweep,audit
