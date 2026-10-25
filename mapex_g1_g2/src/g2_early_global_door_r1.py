"""G2-EARLYDOOR-R1: proactive GLOBAL next goal at LOCAL start, online door updates.

Use current epoch's *official* MapEx global WFD candidates and original
visvarprob IG numerator, but rank from actually OBSERVED FREE exits at G1
room boundary (as if robot would stand at portal on leaving room).
Every subsequent LOCAL LiDAR update recomputes eligible portals and current
WFD, mean/variance/IG; new door is incorporated immediately.
LOCAL sweeping, G1 room decision/exhaustion, LaMa and native A* unchanged.
On first subsequent GLOBAL decision, stored goal may replace fresh MapEx winner
only when matched to current official WFD and reachable; otherwise fallback.

Exit candidates are FREE 4-connected exterior pixels bordering current
predicted G1 ROOM component, not certified architectural doors.
No GT; no future observations; no hard-coded room coordinate; no lookahead.
"""
import numpy as np
import cv2
import pyastar2d
from scipy.ndimage import binary_dilation

FOUR=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

def visible_portals(room_mask,obs_code,native_cost,native_pad):
    mask=np.asarray(room_mask,dtype=bool)
    observed=np.asarray(obs_code)
    nav=np.asarray(native_cost)
    if mask.shape!=observed.shape:return []
    ring=(binary_dilation(mask,structure=FOUR.astype(bool)) & ~mask & (observed==0))
    count,comp=cv2.connectedComponents(ring.astype(np.uint8),connectivity=4)
    portals=[]
    for j in range(1,int(count)):
        pix=np.argwhere(comp==j)
        if len(pix)==0:continue
        core_nav=pix+int(native_pad)
        live=np.isfinite(nav[core_nav[:,0],core_nav[:,1]])
        px=pix[live]
        if len(px)==0:continue
        center=np.mean(px.astype(np.float64),axis=0)
        pick=px[np.argmin(np.sum((px-center)**2,axis=1))]
        portals.append(dict(point_rc=(pick+native_pad).astype(int).tolist(),
          center_P1=[float(v) for v in center.tolist()],
          boundary_free_pixels=int(len(pix)),passable_boundary_pixels=int(len(px))))
    portals.sort(key=lambda p:tuple(p['point_rc']))
    return portals

def _route(grid,start,goal):
    a=np.asarray(start,np.int32);b=np.asarray(goal,np.int32)
    if not (np.isfinite(grid[tuple(a)]) and np.isfinite(grid[tuple(b)])):return None
    return pyastar2d.astar_path(np.ascontiguousarray(grid,dtype=np.float32),a,b,allow_diagonal=False)

def choose_early(room_id,tick,robot,global_pts,global_costs,room_mask,obs_code,
                 native_grid,native_pad,previous=None):
    pts=np.asarray(global_pts,dtype=np.int32).reshape((-1,2))
    cost=np.asarray(global_costs,float).reshape((-1,))
    robot=np.asarray(robot,np.int32)
    room=np.asarray(room_mask,bool)
    nav=np.asarray(native_grid,dtype=np.float32)
    if not len(pts) or len(pts)!=len(cost) or not np.isfinite(cost).all():
        raise ValueError('EARLY_GLOBAL_INPUT_BAD')
    doors=visible_portals(room,obs_code,nav,native_pad)
    # Determine whether a currently mapped traversable exit has appeared.
    prev_ports=[] if previous is None else [p['point_rc'] for p in previous.get('portals',[])]
    # Only exact physically observed portal changes qualify as a new version.
    current_ports=[p['point_rc'] for p in doors]
    changed_door=bool(previous is None or current_ports!=prev_ports)
    valid=[]
    # Native MapEx scores are negative IG/Euclidean(current pose, frontier).
    # Recover SAME-EPOCH IG exactly, then use Euclidean(door,frontier)
    # because at this stage we plan GLOBAL when room is exited.
    for i,point in enumerate(pts):
        core=point-native_pad
        y,x=map(int,core)
        if y<0 or y>=room.shape[0] or x<0 or x>=room.shape[1]:
            continue
        if room[y,x]:continue  # LOCAL candidates are not next GLOBAL targets
        ig=-float(cost[i])*float(np.linalg.norm(point-robot))
        if not np.isfinite(ig) or ig<=0:continue
        if not np.isfinite(nav[tuple(point)]):continue
        for did,door in enumerate(doors):
            portal=np.asarray(door['point_rc'],np.int32)
            eu=float(np.linalg.norm(point-portal))
            if eu==0:continue
            path=_route(nav,portal,point)
            if path is None:continue
            # Passing back through the room would defeat using this portal.
            interior=path-native_pad
            yi=interior[:,0];xi=interior[:,1]
            inside=(yi>=0)&(yi<room.shape[0])&(xi>=0)&(xi<room.shape[1])
            if np.any(room[yi[inside],xi[inside]]):continue
            score=ig/eu
            valid.append(dict(index=int(i),goal_rc=point.astype(int).tolist(),
                 portal_rc=portal.tolist(),portal_index=int(did),
                 predicted_global_IG=float(ig),portal_euclidean_m=eu*.1,
                 portal_score=float(score),portal_astar_m=float((len(path)-1)*.1)))
    if valid:
        best=max(valid,key=lambda q:(q['portal_score'],-q['portal_astar_m'],-q['index'],-q['portal_index']))
        chosen=dict(room_id=int(room_id),entry_tick=int(tick) if previous is None else previous['entry_tick'],
           updated_tick=int(tick),goal_rc=best['goal_rc'],portal_rc=best['portal_rc'],
           score=best['portal_score'],IG=best['predicted_global_IG'],
           selected_entry=previous is None,portals=doors,
           n_exterior_global_frontiers=int(len(set(v['index'] for v in valid))))
        reason='EARLY_GLOBAL_SELECTED' if previous is None else 'EARLY_GLOBAL_REFRESHED'
    else:
        chosen=dict(room_id=int(room_id),entry_tick=int(tick) if previous is None else previous['entry_tick'],
            updated_tick=int(tick),goal_rc=None,portal_rc=None,score=None,IG=None,
            selected_entry=previous is None,portals=doors,n_exterior_global_frontiers=0)
        reason='NO_PASSABLE_NEXT_GLOBAL_FRONTIER'
    prev_goal=None if previous is None else previous['goal_rc']
    prev_portal=None if previous is None else previous['portal_rc']
    changed_goal=bool(previous is None or chosen['goal_rc']!=prev_goal)
    changed_exit=bool(previous is None or chosen['portal_rc']!=prev_portal)
    audit=dict(policy='EARLY_GLOBAL_DOOR_AWARE_R1',
       reason=reason,room_id=int(room_id),native_tick=int(tick),
       previous_goal_rc=prev_goal,current_goal_rc=chosen['goal_rc'],
       previous_portal_rc=prev_portal,current_portal_rc=chosen['portal_rc'],
       new_or_changed_observed_portal=changed_door,
       next_global_goal_changed=changed_goal,exit_portal_changed=changed_exit,
       n_portals=len(doors),n_viable_options=len(valid),
       n_global_candidates=len(pts),
       chosen_from_current_native_MapEx_global_IG=True,
       updated_after_each_LOCAL_lidar=True,
       global_goal_not_committed_until_exit=True,
       confidence_note='G1 boundary passable openings are only observed-FREE passage candidates, not verified room doors.')
    return chosen,audit

def release_on_global(reservation,global_pts,global_scores,robot,native_grid,
                      arrival_m=.1,cell_m=.1):
    pts=np.asarray(global_pts,dtype=np.int32).reshape((-1,2))
    scores=np.asarray(global_scores,float).reshape((-1,))
    baseline=int(np.argmin(scores))
    if reservation is None:
        return baseline,dict(reason='NO_EARLY_GOAL',old_goal_rc=None,
          original_index=baseline,selected_index=baseline)
    goal=reservation.get('goal_rc')
    audit=dict(room_id=reservation.get('room_id'),entry_tick=reservation['entry_tick'],
       reserved_goal_rc=goal,portal_rc=reservation.get('portal_rc'),
       original_index=baseline,selected_index=baseline,
       n_current_WFD=len(pts),reason='MAPEX_FALLBACK',
       global_decision_uses_fresh_observed_WFD=True)
    if goal is None:
        audit['reason']='NO_GLOBAL_RESERVATION'
        return baseline,audit
    desired=np.asarray(goal,dtype=np.int32)
    # Match only to a currently verified WFD: never force an obsolete goal.
    separation=np.linalg.norm(pts-desired,axis=1)*float(cell_m)
    matches=np.flatnonzero(separation<=float(arrival_m))
    if not len(matches):
        audit['reason']='GOAL_NO_LONGER_CURRENT_WFD'
        return baseline,audit
    for candidate in sorted(matches,key=lambda i:(separation[i],scores[i],int(i))):
        # Check path from CURRENT robot on CURRENT observed MapEx native grid.
        pp=_route(np.asarray(native_grid),robot,pts[candidate])
        if pp is None:continue
        audit['reason']='VALID_EARLY_GLOBAL_GOAL'
        audit['selected_index']=int(candidate)
        audit['matched_distance_m']=float(separation[candidate])
        audit['current_map_astar_m']=float((len(pp)-1)*cell_m)
        return int(candidate),audit
    audit['reason']='RESERVED_GOAL_UNREACHABLE_CURRENT_MAP'
    return baseline,audit
