"""Delayed training metrics for complete cohorts of passage episode starts."""

import copy

import numpy as np


SHORTEST_PATH_STEPS = 16


class PassageCohorts:
    """Observe existing rollouts only; never sample, reset or alter the policy."""

    def __init__(self, config, *, tick=0, episode_steps=None):
        self.rollout = config.rollout_steps
        self.span = config.metric_window_updates * self.rollout
        self.maximum_steps = config.maximum_episode_steps
        self.tick = tick
        self.starts = tick - (np.zeros(config.environments, dtype=np.int64)
                              if episode_steps is None else np.asarray(episode_steps))
        # Legacy checkpoints lack earlier cohort totals. Skip their partial cohort.
        self.next_cohort = (tick + self.span - 1) // self.span
        self.pending = {}

    def _bucket(self, cohort):
        # Columns: started, completed, successes, complete-episode frames.
        return self.pending.setdefault(cohort, np.zeros((2, 4), dtype=np.int64))

    def observe(self, rollout):
        """rollout[T,N,3] contains done, reward and left-side flag before reset."""
        for step in rollout:
            done = step[:, 0].astype(bool)
            success = step[:, 1] > 0
            side = 1 - step[:, 2].astype(np.int64)  # left=0, right=1
            born = self.starts == self.tick
            cohort = self.tick // self.span
            if cohort >= self.next_cohort:
                self._bucket(cohort)[:, 0] += np.bincount(side[born], minlength=2)
            origins = self.starts // self.span
            for origin in np.unique(origins[done]):
                if origin < self.next_cohort:
                    continue
                mask = done & (origins == origin)
                bucket = self._bucket(int(origin))
                bucket[:, 1] += np.bincount(side[mask], minlength=2)
                bucket[:, 2] += np.bincount(side[mask & success], minlength=2)
                np.add.at(bucket[:, 3], side[mask], self.tick - self.starts[mask] + 1)
            self.starts[done] = self.tick + 1
            self.tick += 1
        published = []
        # The last start in the cohort has now had its full technical horizon.
        while (self.next_cohort + 1) * self.span + self.maximum_steps <= self.tick:
            cohort = self.next_cohort
            counts = self.pending.pop(cohort)
            if not np.array_equal(counts[:, 0], counts[:, 1]):
                raise ValueError('Cannot publish an incomplete passage episode cohort')
            record = dict(start_update=cohort * self.span // self.rollout + 1,
                          end_update=(cohort + 1) * self.span // self.rollout,
                          published_update=self.tick // self.rollout,
                          shortest_path_steps=SHORTEST_PATH_STEPS)
            for side, (_, completed, successes, frames) in zip(('left', 'right'), counts):
                record.update({f'{side}_episodes': int(completed),
                               f'{side}_successes': int(successes),
                               f'{side}_frames': int(frames),
                               f'{side}_success_rate': float(successes / completed) if completed else None,
                               f'{side}_efficiency': float(SHORTEST_PATH_STEPS * successes / frames) if frames else None})
            published.append(record)
            self.next_cohort += 1
        return published

    def state(self):
        return copy.deepcopy(dict(tick=self.tick, starts=self.starts,
                                  next_cohort=self.next_cohort, pending=self.pending))

    def restore(self, state):
        if np.asarray(state['starts']).shape != self.starts.shape:
            raise ValueError('Passage cohort state shape mismatch')
        self.tick = state['tick']
        self.starts = state['starts'].copy()
        self.next_cohort = state['next_cohort']
        self.pending = copy.deepcopy(state['pending'])
