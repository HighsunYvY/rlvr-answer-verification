"""Exact fixed-rollout expectations for the E4 reward control."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .analysis import diagnose_scored_row
from .attribution import follows_equals
from .experiment import read_json, write_json
from .rewards import shuffled_weak_rewards, weak_shuffle_probabilities
from .verifier import weak_match_spans


def replay_e4(source, output, group_size=4, training_audit=False):
    if group_size < 2:
        raise ValueError('E4 replay requires at least two completions per prompt')
    source = Path(source)
    weights = [0.2, 1.0]
    groups = []
    if training_audit:
        metrics = read_json(source / 'metrics.json')
        if metrics['format_and_correctness_weights'] != weights or metrics['group_size'] != group_size:
            raise ValueError('training audit reward settings differ from E4 replay settings')
        reward_components = {0.0: (False, 0.0), 0.2: (False, 1.0),
                             1.0: (True, 0.0), 1.2: (True, 1.0)}
        with (source / 'training-groups.jsonl').open() as stream:
            for line in stream:
                group = json.loads(line)
                rows = []
                for row in group['members']:
                    strict, format_score = reward_components[row['strict']['total_reward']]
                    weak = float(row['weak_positive'])
                    if row['weak']['total_reward'] != 0.2 * format_score + weak:
                        raise ValueError(f'inconsistent archived rewards for {row["sample_id"]}')
                    rows.append({**row, 'strict_correct': strict, 'format_reward': format_score,
                                 'weak_accuracy_reward': weak})
                groups.append((group['source_sample_id'], rows))
        if len(groups) != metrics['training']['groups']:
            raise ValueError('incomplete archived training group inventory')
    else:
        by_prompt = {}
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                source_id, member = row['source_sample_id'], row['group_member']
                if row['sample_id'] != f'{source_id}#g{member}' or type(member) is not int:
                    raise ValueError('invalid grouped evaluation identity')
                by_prompt.setdefault(source_id, []).append(diagnose_scored_row(row, row['ground_truth']))
        for source_id, rows in by_prompt.items():
            rows.sort(key=lambda row: row['group_member'])
            if [row['group_member'] for row in rows] != list(range(group_size)):
                raise ValueError(f'incomplete or duplicated group members for {source_id}')
            groups.append((source_id, rows))

    totals = Counter()
    for index, (source_id, rows) in enumerate(groups):
        if len(rows) != group_size:
            raise ValueError(f'group size differs for {source_id}')
        formats = [0.2 * row['format_reward'] for row in rows]
        weak = [row['weak_accuracy_reward'] for row in rows]
        scores = [f + w for f, w in zip(formats, weak, strict=True)]
        probabilities = weak_shuffle_probabilities(rows)
        shuffled = shuffled_weak_rewards(rows, 44, 'offline_replay', 0, source_id, index)
        if sorted(scores) != sorted(f + w for f, w in zip(formats, shuffled, strict=True)):
            raise ValueError(f'invariant failed for {source_id}')
        totals['groups_with_movable_reward'] += any(0 < p < 1 for p in probabilities)
        totals['expected_moved_awards'] += sum(1 - p for w, p in zip(weak, probabilities, strict=True) if w)
        center = mean(scores)
        for row, probability, format_score, score in zip(rows, probabilities, formats, scores, strict=True):
            if row['explicit_final_correct'] is not False:
                continue
            positive_weak = score - center > 1e-12
            positive_e4 = ((1 - probability) * (format_score - center > 1e-12)
                           + probability * (format_score + 1 - center > 1e-12))
            totals['wrong_positive_under_weak'] += positive_weak
            totals['expected_wrong_positive_under_e4'] += positive_e4
            if not row['weak_positive']:
                continue
            totals['weak_hit_wrong_positive_under_weak'] += positive_weak
            totals['expected_weak_hit_wrong_positive_under_e4'] += positive_e4
            if not training_audit and any(follows_equals(row['completion'], start)
                                          for start, _ in weak_match_spans(row['completion'], row['ground_truth'])):
                totals['equals_hit_wrong_positive_under_weak'] += positive_weak
                totals['expected_equals_hit_wrong_positive_under_e4'] += positive_e4

    if not groups:
        raise ValueError('E4 replay requires nonempty group evidence')
    summary = {'source': str(source.resolve()), 'training_audit': training_audit, 'group_size': group_size,
               'groups': len(groups), 'completions': len(groups) * group_size,
               'counts': dict(totals), 'verified_reward_multiset_groups': len(groups)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', summary)
    return summary
