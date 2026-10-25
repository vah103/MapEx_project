"""MapEx-GTP R7 goal feedback. Online observed map and pose only; Python 3.6.

This module does not decide whether exploration is complete. Native WFD retains
that responsibility. A failed observation is evidence about a plan, never proof
that an unknown cell is unreachable.
"""
import math
import hashlib
import numpy as np

FEATURES = dict(verify_observation=False, event_refresh=True,
                failure_regions=True, fallback_guard=False, cleanup_retry=False,
                jump_release=True, tail_rule=True, incremental_visibility=True)
GAIN_QUANTUM = 10                # reuse the ordinary planner's MIN_GAIN
ENDPOINT_RADIUS_CELLS = 0.5      # integer native grid: arrive at the actual cell
PATH_PROGRESS_CELLS = 3         # nominal native controller step
FALLBACK_MAX_UNCHANGED_STALLS = 2  # one retry, then explicit incomplete failure
CLEANUP_MAX_ATTEMPTS = 1         # one additional attempt per exhausted frontier cell


class GoalFeedback(object):
    def __init__(self, shape, pad, lock, stride, features=None):
        self.shape = tuple(shape)
        self.pad, self.lock, self.stride = int(pad), float(lock), int(stride)
        self.features = dict(FEATURES)
        if features is not None:
            if set(features) - set(FEATURES):
                raise ValueError('unknown R7 feature')
            self.features.update(features)
        if any(type(v) is not bool for v in self.features.values()):
            raise ValueError('R7 features must be booleans')
        self.active = None
        self.outcomes = []
        self.stalled_fallbacks = {}
        self.regions = []
        self.region_events = []
        self.pending_refresh = set()
        self.prediction_observed = None
        self.prediction_tick = None
        self.prediction_serial = 0
        self.cleanup_attempts = np.zeros(int(np.prod(shape)), np.uint8)
        # Reuse R5's 20-tick slack and 2x nominal traversal of its lock radius.
        self.stall_ticks = 20 + int(math.ceil(2.0 * lock / 3.0))

    def core(self, obs):
        p = self.pad
        return np.asarray(obs[p:p + self.shape[0], p:p + self.shape[1]])

    def prediction_recorded(self, obs, t):
        self.prediction_observed = self.core(obs).copy()
        self.prediction_tick = int(t)
        self.prediction_serial += 1
        self.pending_refresh.clear()

    def refresh_reasons(self, obs):
        if not self.features['event_refresh'] or not self.pending_refresh:
            return []
        # Repeating a deterministic predictor on identical input adds no evidence.
        if self.prediction_observed is not None and np.array_equal(
                self.prediction_observed, self.core(obs)):
            return []
        return sorted(self.pending_refresh)

    def observed_digest(self, obs):
        return hashlib.sha256(np.ascontiguousarray(self.core(obs), dtype=np.float32).tobytes()).hexdigest()

    def _stall_gain(self, history, obs):
        return int((history['unknown_at_record'] & (self.core(obs) != .5)).sum())

    def exhausted_fallback(self, goal, obs):
        h = self.stalled_fallbacks.get(tuple(np.asarray(goal, dtype=int)))
        return bool(self.features['fallback_guard'] and h is not None
                    and self._stall_gain(h, obs) < GAIN_QUANTUM
                    and h['attempts'] >= FALLBACK_MAX_UNCHANGED_STALLS)

    def start(self, kind, goal, planned, obs, t, path_len, decision):
        c = self.core(obs)
        prior = self.stalled_fallbacks.get(tuple(np.asarray(goal, dtype=int))) if kind == 'fallback' else None
        if prior is not None and self._stall_gain(prior, obs) >= GAIN_QUANTUM:
            prior = None
        initial = int(path_len) if path_len is not None else None
        best = initial
        if prior is not None and prior['best_len'] is not None and initial is not None:
            best = min(initial, prior['best_len'])
        self.active = dict(kind=kind, goal_rc=np.asarray(goal, dtype=int).tolist(),
                           planned=np.asarray(planned, dtype=np.int64).copy(),
                           known0=int((c != .5).sum()), free0=int((c == 0).sum()),
                           last_gain_known=int((c != .5).sum()), last_gain_tick=int(t),
                           t0=int(t), L0=int(path_len) if path_len is not None else None,
                           last_len=None, decision=int(decision), best_len=best,
                           last_path_progress_tick=int(t), path_progress_cells=0,
                           path_map_signature=(int((c != .5).sum()), int((c == 1).sum())),
                           path_geometry_changed=False, upward_reanchors=0,
                           prior_stalled_attempts=0 if prior is None else prior['attempts'])

    def update(self, obs, t):
        if self.active is None:
            return
        c = self.core(obs)
        n = int((c != .5).sum())
        signature = (n, int((c == 1).sum()))
        if signature != self.active['path_map_signature']:
            self.active['path_geometry_changed'] = True
            self.active['path_map_signature'] = signature
        if n - self.active['last_gain_known'] >= GAIN_QUANTUM:
            self.active['last_gain_known'] = n
            self.active['last_gain_tick'] = int(t)

    def release_reason(self, pose, obs):
        a = self.active
        if a is None:
            return None
        dist = float(np.linalg.norm(np.asarray(a['goal_rc']) - pose))
        if a['kind'] == 'viewpoint' and self.features['verify_observation']:
            if not (self.core(obs).ravel()[a['planned']] == .5).any():
                return 'observed'
            return 'reached' if dist <= ENDPOINT_RADIUS_CELLS else None
        return 'reached' if dist < self.lock else None

    def retain_near_viewpoint(self):
        return bool(self.active is not None and self.active['kind'] == 'viewpoint'
                    and self.features['verify_observation'])

    def fallback_exceeded(self, t, path_len):
        a = self.active
        if a is None or a['kind'] != 'fallback' or not self.features['fallback_guard']:
            return False
        path_len = int(path_len)
        # A newly observed obstruction may lengthen a valid route. Re-anchor
        # the distance comparison, granting no progress merely for that change.
        if a['path_geometry_changed']:
            if a['best_len'] is None or path_len > a['best_len']:
                a['best_len'] = path_len
                a['upward_reanchors'] += 1
            a['path_geometry_changed'] = False
        if a['best_len'] is None:
            a['best_len'] = path_len
        if path_len <= a['best_len'] - PATH_PROGRESS_CELLS:
            a['path_progress_cells'] += a['best_len'] - path_len
            a['best_len'] = path_len
            a['last_path_progress_tick'] = int(t)
        a['last_len'] = path_len
        path_growth = a['L0'] is not None and path_len > 2.0 * a['L0'] + 30
        stalled = (int(t) - a['last_gain_tick'] > self.stall_ticks
                   and int(t) - a['last_path_progress_tick'] > self.stall_ticks)
        return bool(path_growth or stalled)

    def _box(self, goal):
        r, c = np.asarray(goal, dtype=int) - self.pad
        s = self.stride
        return (max(0, r - s), min(self.shape[0], r + s + 1),
                max(0, c - s), min(self.shape[1], c + s + 1))

    def add_region(self, goal, obs, mean, reason, t):
        if not self.features['failure_regions']:
            return
        box = self._box(goal)
        y0, y1, x0, x1 = box
        if y0 >= y1 or x0 >= x1:
            return
        rc = tuple((np.asarray(goal, dtype=int) - self.pad).tolist())
        local = self.core(obs)[y0:y1, x0:x1]
        self.regions = [b for b in self.regions if b['goal_core'] != rc]
        self.regions.append(dict(goal_core=rc, box=box, add_tick=int(t),
                                 observed=local.copy()))
        self.region_events.append(dict(tick=int(t), action='add', goal_core=list(rc), reason=reason))

    def update_regions(self, obs, mean, t):
        self.current_tick = int(t)
        c = self.core(obs)
        kept = []
        for b in self.regions:
            y0, y1, x0, x1 = b['box']
            gain = int(((b['observed'] == .5) & (c[y0:y1, x0:x1] != .5)).sum())
            if gain >= GAIN_QUANTUM:
                self.region_events.append(dict(tick=int(t), action='release',
                    goal_core=list(b['goal_core']), add_tick=b['add_tick'],
                    release_tick=int(t), cells_newly_known=gain, reason='observed_gain_quantum'))
            else:
                kept.append(b)
        self.regions = kept

    def allowed(self, candidates_core):
        q = np.asarray(candidates_core).reshape((-1, 2))
        keep = np.ones(len(q), dtype=bool)
        if self.features['failure_regions']:
            for b in self.regions:
                blocked = np.abs(q - np.asarray(b['goal_core'])).max(axis=1) <= self.stride
                for candidate in q[blocked]:
                    self.region_events.append(dict(tick=getattr(self, 'current_tick', b['add_tick']),
                        action='exclude', goal_core=list(b['goal_core']),
                        candidate_core=candidate.astype(int).tolist(), add_tick=b['add_tick']))
                keep &= ~blocked
        return keep

    def finish(self, reason, obs, mean, t):
        a = self.active
        if a is None:
            return None
        c = self.core(obs)
        vals = c.ravel()[a['planned']]
        missing = int((vals == .5).sum())
        got = len(vals) - missing
        new_known = int((c != .5).sum()) - a['known0']
        out = dict(kind=a['kind'], decision=a['decision'], tick=int(t), t0=a['t0'],
                   reason=reason, goal_rc=a['goal_rc'], planned=len(vals),
                   planned_observed=int(got), planned_still_unknown=missing,
                   planned_free=int((vals == 0).sum()), planned_occupied=int((vals == 1).sum()),
                   planned_observation_success=bool(len(vals) and missing == 0),
                   new_known=int(new_known), new_free=int((c == 0).sum()) - a['free0'],
                   L0=a['L0'], last_path_len=a['last_len'])
        if a['kind'] == 'fallback' and reason == 'no_progress' and self.features['fallback_guard']:
            key = tuple(a['goal_rc'])
            digest = self.observed_digest(obs)
            prior = self.stalled_fallbacks.get(key)
            same = prior is not None and self._stall_gain(prior, obs) < GAIN_QUANTUM
            attempts = 1 + (prior['attempts'] if same else 0)
            unknown = prior['unknown_at_record'] if same else (c == .5).copy()
            self.stalled_fallbacks[key] = dict(observed_sha256=digest, attempts=attempts,
                unknown_at_record=unknown, best_len=a['best_len'])
        out.update(prior_stalled_attempts=a['prior_stalled_attempts'],
                   path_progress_cells=a['path_progress_cells'], best_remaining_path_cells=a['best_len'],
                   upward_reanchors=a['upward_reanchors'])
        failure = reason in ('invalid', 'no_path', 'no_progress', 'native_reselection', 'executor_rejection')
        zero_planned = a['kind'] == 'viewpoint' and len(vals) > 0 and got == 0
        empty_fallback = a['kind'] == 'fallback' and reason == 'reached' and new_known == 0
        if self.features['event_refresh'] and (failure or zero_planned or empty_fallback):
            self.pending_refresh.add(reason if failure else
                                     'zero_fallback_observation' if empty_fallback else 'zero_planned_observation')
        # Retain the actual gain of failed legs; suppress their failing endpoint.
        # A merely reached zero-planned goal is suppressed only if low-gain too.
        if failure or empty_fallback or (zero_planned and new_known < GAIN_QUANTUM):
            self.add_region(a['goal_rc'], obs, mean, reason, t)
        self.outcomes.append(out)
        self.active = None
        return out

    def summary(self):
        return dict(features=dict(self.features), outcomes=list(self.outcomes),
                    active_goal=None if self.active is None else dict(
                        kind=self.active['kind'], goal_rc=self.active['goal_rc'], decision=self.active['decision']),
                    region_events=list(self.region_events), active_failure_regions=len(self.regions),
                    unchanged_stalled_fallbacks=[dict(goal_rc=list(k), attempts=v['attempts'])
                                                for k, v in sorted(self.stalled_fallbacks.items())],
                    pending_refresh=sorted(self.pending_refresh), prediction_serial=self.prediction_serial,
                    cleanup_attempted_cells=int((self.cleanup_attempts > 0).sum()),
                    params=dict(path_progress_cells=PATH_PROGRESS_CELLS,
                                max_unchanged_fallback_stalls=FALLBACK_MAX_UNCHANGED_STALLS,
                                endpoint_radius_cells=ENDPOINT_RADIUS_CELLS, gain_quantum=GAIN_QUANTUM,
                                failure_radius_cells=self.stride, fallback_stall_ticks=self.stall_ticks,
                                cleanup_max_attempts=CLEANUP_MAX_ATTEMPTS),
                    completion_claim='terminal class and common coverage gate determine evaluation; planned misses remain unknown')
