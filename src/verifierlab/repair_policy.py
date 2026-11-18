"""Budgeted online interventions on naturally generated four-response groups."""
from collections import Counter
from random import Random

from .evaluate import diagnose_final_answer
from .verifier import normalize_number


class RepairPolicy:
    def __init__(self, plan, steps, seed):
        self.policy = plan['policy']
        self.seed = seed
        self.window = plan['window_steps']
        self.caps = plan['caps']
        if self.policy not in ('mixed', 'all_wrong', 'weak_control', 'tag_priority'):
            raise ValueError('unknown repair policy')
        if (type(self.window) is not int or self.window <= 0 or not isinstance(self.caps, list)
                or len(self.caps) != (steps + self.window - 1) // self.window
                or any(type(n) is not int or n < 0 for n in self.caps)):
            raise ValueError('one nonnegative cap is required per optimizer-step window')
        self.spent = Counter()
        self.used_ids = set()

    def apply(self, members, partition, step, group_index):
        scores = [r['weak_accuracy_reward'] for r in members]
        changed = []
        flags = [r['strict_correct'] for r in members]
        known = all(r['format_compliant'] and not r['truncated']
                    and diagnose_final_answer(r)['explicit_final_correct'] is r['strict_correct'] for r in members)
        context = 'unknown'
        if known:
            context = {0: 'all_wrong', len(flags): 'all_correct'}.get(sum(flags), 'mixed')
        eligible = [i for i, row in enumerate(members) if row['weak_positive'] and not flags[i]]
        source = members[0]['source_sample_id']
        if partition == 'train':
            if self.policy == 'tag_priority':
                for i, row in enumerate(members):
                    if normalize_number(row['parsed_answer']) is not None:
                        scores[i] = row['strict_accuracy_reward']
            elif self.policy == context and eligible and source not in self.used_ids:
                window = step // self.window
                if self.spent[window] < self.caps[window]:
                    rng = Random(f'repair-v1:{self.seed}:{step}:{source}:{group_index}')
                    index = rng.choice(eligible)
                    scores[index] = 0.
                    self.spent[window] += 1
                    self.used_ids.add(source)
        for i, (row, score) in enumerate(zip(members, scores, strict=True)):
            row['repair_accuracy_reward'] = score
            if score != row['weak_accuracy_reward']:
                changed.append(i)
        return {'policy': self.policy, 'context': context, 'eligible_indices': eligible if known else [],
                'changed_indices': changed, 'weak_awards': sum(r['weak_positive'] for r in members)}


def select_frozen_rows(rows, ids):
    indexed = {row['sample_id']: row for row in rows}
    if not isinstance(ids, list) or len(ids) < 4 or len(set(ids)) != len(ids) or not set(ids) <= indexed.keys():
        raise ValueError('invalid or duplicate IDs in repair experiment split')
    return [indexed[key] for key in ids]
