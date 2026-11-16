"""Replay the logged GRPO reward signal for completed E2/E3 training runs."""
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from .analysis import diagnose_scored_row, group_reward_evidence
from .experiment import prepared_inputs, read_json, verify_inventory, write_json
from .rewards import completion_text


def audit_group(rows, verifier, weights, recorded):
    strict, strict_members = group_reward_evidence(rows, 'strict', weights)
    weak, weak_members = group_reward_evidence(rows, 'weak', weights)
    applied, applied_members = (strict, strict_members) if verifier == 'strict' else (weak, weak_members)
    if (recorded['selected_total_rewards'] != [r['total_reward'] for r in applied_members]
            or recorded['variable_selected_total'] != applied['variable_total']):
        raise ValueError('replayed rewards differ from the logged trainer rewards')
    correct_weak_advantages = [w['advantage'] for r, w in zip(rows, weak_members, strict=True)
                               if r['explicit_final_correct'] is True]
    best_correct = max(correct_weak_advantages, default=None)
    counts = Counter(groups=1, completions=len(rows),
                     zero_applied_total_groups=int(not applied['variable_total']),
                     zero_applied_correctness_groups=int(not applied['variable_correctness']),
                     zero_weak_total_groups=int(not weak['variable_total']))
    members = []
    for row, s, w, actual in zip(rows, strict_members, weak_members, applied_members, strict=True):
        correct = row['explicit_final_correct']
        paid_wrong = correct is False and row['weak_positive']
        positive_paid_wrong = paid_wrong and w['sign'] == 'positive'
        counts['explicit_final_correct'] += correct is True
        counts['explicit_final_undetermined'] += correct is None
        counts['weak_rewarded_wrong'] += paid_wrong
        counts['weak_rewarded_wrong_positive_under_weak'] += positive_paid_wrong
        counts['weak_rewarded_wrong_became_positive'] += positive_paid_wrong and s['sign'] != 'positive'
        counts['weak_rewarded_wrong_positive_under_applied'] += paid_wrong and actual['sign'] == 'positive'
        counts['correct_positive_under_applied'] += correct is True and actual['sign'] == 'positive'
        counts['wrong_positive_under_applied'] += correct is False and actual['sign'] == 'positive'
        if positive_paid_wrong:
            if best_correct is None:
                comparison = 'no_explicit_correct'
            elif w['advantage'] == best_correct:
                comparison = 'tied_best_correct'
            elif w['advantage'] < best_correct:
                comparison = 'below_best_correct'
            else:
                comparison = 'above_best_correct'
            counts[f'weak_positive_wrong_{comparison}'] += 1
        members.append({'sample_id': row['sample_id'], 'explicit_final_answer': row['explicit_final_answer'],
                        'explicit_final_correct': correct, 'weak_positive': row['weak_positive'],
                        'truncated': row['truncated'], 'strict': s, 'weak': w})
    return counts, {'strict': strict, 'weak': weak, 'members': members}


def audit_training(run_dir, preparation_dir, output_dir, window_steps=25):
    if type(window_steps) is not int or window_steps <= 0:
        raise ValueError('window size must be a positive integer')
    directory = Path(run_dir)
    report = read_json(directory / 'run.json')
    stage = report['stage']
    resolved = report['resolved_trainer']
    steps = report['optimizer_steps']
    generations = resolved['num_generations']
    weights = report['settings']['reward_weights']
    verifier = {'E2': 'strict', 'E3': 'weak'}.get(stage)
    if (not verifier or report['status'] != 'passed' or report['mode'] != 'run'
            or not report['trainer_validated'] or report.get('resumed_report_sha256')
            or report['environment']['packages']['trl']['installed'] != '0.24.0'
            or resolved['scale_rewards'] != 'group' or resolved['num_iterations'] != 1
            or resolved['mask_truncated_completions'] is not False
            or generations < 2 or len(weights) != 2 or steps != report['settings']['max_steps']):
        raise ValueError('replay requires a complete fresh TRL 0.24.0 E2/E3 run with group-scaled rewards')

    splits, _, preparation = prepared_inputs(preparation_dir, report['profile'])
    if report['manifest_sha256'] != preparation['split_manifest_sha256']:
        raise ValueError('frozen training manifest differs from the recorded run')
    references = {part: {r['sample_id']: r for r in splits[split]}
                  for part, split in (('train', 'grpo'), ('validation', 'validation'))}
    filenames = ('reward-inputs.jsonl', 'reward-groups.jsonl', 'reward-rollouts.jsonl')
    verify_inventory(directory, {name: report['artifacts'][name] for name in filenames})

    total, windows, groups, seen_train_steps = Counter(), {}, [], set()
    validation_rows = 0
    with (directory / filenames[0]).open() as inputs, (directory / filenames[1]).open() as recorded, \
            (directory / filenames[2]).open() as rollouts:
        for call_id, (input_line, group_line) in enumerate(zip(inputs, recorded, strict=True)):
            batch, logged = json.loads(input_line), json.loads(group_line)
            step = batch['optimizer_step']
            if (batch['call'] != call_id or logged['call'] != call_id
                    or logged['optimizer_step'] != step or type(step) is not int or not 0 <= step <= steps):
                raise ValueError(f'call/step mismatch at call {call_id}')
            logged_groups = logged['groups']
            fields = ('sample_id', 'partition', 'prompts', 'completions', 'ground_truth', 'completion_ids')
            count = len(logged_groups) * generations
            rows = [json.loads(line) for line in islice(rollouts, count)]
            if not count or len(rows) != count or any(len(batch[name]) != count for name in fields):
                raise ValueError(f'incomplete reward batch at call {call_id}')
            parts = {group['partition'] for group in logged_groups}
            if len(parts) != 1 or not parts <= references.keys():
                raise ValueError(f'unknown or mixed partitions at call {call_id}')
            partition = next(iter(parts))
            for group_index, group in enumerate(logged_groups):
                start = group_index * generations
                source_id = group['sample_id']
                ref = references[partition][source_id]
                members = []
                for i in range(start, start + generations):
                    row = rows[i]
                    if (row['sample_id'] != f'call{call_id}:{i}' or row['source_sample_id'] != source_id
                            or batch['sample_id'][i] != source_id or batch['partition'][i] != partition
                            or batch['prompts'][i] != ref['prompt'] or batch['ground_truth'][i] != ref['ground_truth']
                            or completion_text(batch['completions'][i]) != row['completion']
                            or batch['completion_ids'][i] != row['trainer_completion_ids']):
                        raise ValueError(f'reward inputs, rollout, group or frozen row differ at call {call_id}:{i}')
                    if partition == 'train':
                        members.append(diagnose_scored_row(row, ref['ground_truth']))
                if partition == 'validation':
                    validation_rows += generations
                    continue
                counts, evidence = audit_group(members, verifier, weights, group)
                total.update(counts)
                windows.setdefault(step // window_steps, Counter()).update(counts)
                groups.append({'call': call_id, 'completed_steps_before_sampling': step,
                               'source_sample_id': source_id, **evidence})
            if partition == 'train':
                if step >= steps or step in seen_train_steps:
                    raise ValueError('expected one training generation call before each optimizer step')
                seen_train_steps.add(step)
        if next(rollouts, None) is not None or seen_train_steps != set(range(steps)):
            raise ValueError('incomplete optimizer-step coverage or excess rollout rows')

    summary = {'source_run': str(directory.resolve()), 'stage': stage, 'applied_verifier': verifier,
               'optimizer_steps': steps, 'window_steps': window_steps, 'group_size': generations,
               'trl_version': '0.24.0', 'format_and_correctness_weights': weights,
               'validation_completions_excluded': validation_rows, 'training': dict(total),
               'windows': [{'optimizer_updates': [index * window_steps + 1, min((index + 1) * window_steps, steps)],
                            **dict(counts)} for index, counts in sorted(windows.items())]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'training-groups.jsonl').open('x') as stream:
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False) + '\n')
    write_json(output / 'metrics.json', summary)
    return summary
