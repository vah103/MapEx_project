"""Exact cache of full ray visibility; independent of need/tries/observations.

Every sampled cell through and INCLUDING the first blocker belongs to visible().
If none of those occupancies changed, no ray's first blocker can have changed.
Invalidate against every map transition, including entries not requested there.
"""
import numpy as np
import tourlib as T


class ExactVisibility(object):
    def __init__(self):
        self.occ = None
        self.cache = {}
        self.recasts = self.hits = self.invalidations = 0

    def set_map(self, occ):
        occ = np.asarray(occ, dtype=bool)
        if self.occ is None or self.occ.shape != occ.shape:
            self.cache.clear()
        else:
            changed = (occ != self.occ).ravel()
            if changed.any():
                for key, v in list(self.cache.items()):
                    if changed[v].any():
                        del self.cache[key]
                        self.invalidations += 1
        self.occ = occ.copy()

    def visible(self, occ, r, c):
        # Caller must call set_map once before querying a batch; assert identity
        # of dimensions. The planner owns and supplies that immutable plan map.
        if self.occ is None or self.occ.shape != occ.shape:
            raise ValueError('set_map must precede visibility queries')
        key = (int(r), int(c))
        if key not in self.cache:
            v = T.visible(occ, *key)
            v.setflags(write=False)
            self.cache[key] = v
            self.recasts += 1
        else:
            self.hits += 1
        return self.cache[key]

    def summary(self):
        return dict(recasts=self.recasts, hits=self.hits, invalidations=self.invalidations,
                    entries=len(self.cache), bytes=sum(v.nbytes for v in self.cache.values()))
