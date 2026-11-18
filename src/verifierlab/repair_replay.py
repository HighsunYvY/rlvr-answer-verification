"""Withdraw one weak correctness award on fixed responses and replay group normalization."""
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean

from .analysis import diagnose_scored_row, group_reward_evidence
from .experiment import prepared_inputs, read_json, write_json


def groups_from_run(directory, report):
    if report['stage'] in ('E2', 'E3', 'E4', 'E5'):
        with (directory / 'reward-rollouts.jsonl').open() as raw, (directory / 'reward-groups.jsonl').open() as log:
            for call in map(json.loads, log):
                for k, group in enumerate(call['groups']):
                    rows = list(map(json.loads, islice(raw, 4)))
                    expected = [f"call{call['call']}:{4*k+i}" for i in range(4)]
                    if ([r['sample_id'] for r in rows] != expected
                            or any(r['source_sample_id'] != group['sample_id'] for r in rows)):
                        raise ValueError('incomplete or misaligned trainer group')
                    part = group['partition']
                    yield ('train' if part == 'train' else f"{part}-{call['optimizer_step']}"), rows
            if next(raw, None) is not None:
                raise ValueError('unexpected trailing training rows')
    elif report['stage'] in ('weak_audit', 'repair_probe'):
        with (directory / 'scored.jsonl').open() as stream:
            while rows := list(map(json.loads, islice(stream, 4))):
                if ([row['group_member'] for row in rows] != list(range(4))
                        or len({(row['source_sample_id'], row.get('panel_draw')) for row in rows}) != 1):
                    raise ValueError('invalid sampled group')
                yield ('probe' if report['stage'] == 'repair_probe' else 'validation'), rows
    else:
        raise ValueError('requires a four-completion training, weak-audit or repair-probe run')


def repair_effects(rows):
    labels = [r['explicit_final_correct'] for r in rows]
    candidates = [i for i, r in enumerate(rows) if labels[i] is False
                  and r['weak_positive'] and not r['strict_correct']]
    if not candidates:
        return None, []
    context = 'undetermined' if None in labels else 'mixed' if True in labels else 'all_wrong'
    weights = [0.2, 1.0]
    _, weak = group_reward_evidence(rows, 'weak', weights)
    _, strict = group_reward_evidence(rows, 'strict', weights)
    if context != 'undetermined':
        target = [dict(row, final_accuracy_reward=int(correct))
                  for row, correct in zip(rows, labels, strict=True)]
        _, ideal = group_reward_evidence(target, 'final', weights)
    events = []
    for i in candidates:
        patched = [dict(row, repair_accuracy_reward=0 if j == i else row['weak_accuracy_reward'])
                   for j, row in enumerate(rows)]
        _, after = group_reward_evidence(patched, 'repair', weights)
        event = {'member': i, 'sample_id': rows[i]['sample_id'],
                 'turned_positive_under_weak': strict[i]['sign'] != 'positive' and weak[i]['sign'] == 'positive',
                 'before': [r['advantage'] for r in weak], 'after': [r['advantage'] for r in after]}
        if context != 'undetermined':
            oracle = [r['advantage'] for r in ideal]
            event['final_only_advantages_with_format_reward'] = oracle
            event['reduction_in_mean_absolute_advantage_error'] = mean(
                abs(before - gold) - abs(post - gold)
                for before, post, gold in zip(event['before'], event['after'], oracle, strict=True))
        events.append(event)
    return context, events


def paired_questions(effects):
    """Expected effect of one award per context, with equal weight per distinct question."""
    pairs = []
    for source, classes in sorted(effects.items()):
        if 'mixed' in classes and 'all_wrong' in classes:
            mixed = mean(classes['mixed'])
            all_wrong = mean(classes['all_wrong'])
            pairs.append({'sample_id': source, 'mixed': mixed, 'all_wrong': all_wrong,
                          'mixed_minus_all_wrong': mixed - all_wrong})
    return {'questions': pairs, 'one_award_per_context_per_question': len(pairs),
            'mean_mixed_minus_all_wrong': mean(p['mixed_minus_all_wrong'] for p in pairs) if pairs else None}


def replay_repairs(run_dir, preparation_dir, output_dir, excluded=()):
    directory, output = Path(run_dir), Path(output_dir)
    report = read_json(directory / 'run.json')
    if report['status'] != 'passed' or report['mode'] != 'run':
        raise ValueError('a completed full run is required')
    splits, _, preparation = prepared_inputs(preparation_dir, report['profile'])
    if report['manifest_sha256'] != preparation['split_manifest_sha256']:
        raise ValueError('frozen data differ from the source run')
    if 'applied_correctness_verifier' in report and (report['settings']['reward_weights'] != [0.2, 1.0]
            or report['resolved_trainer']['num_generations'] != 4
            or report['resolved_trainer']['scale_rewards'] != 'group'):
        raise ValueError('unmatched group normalization or reward contract')
    lookup = {name: {r['sample_id']: r['ground_truth'] for r in splits[name]} for name in ('grpo', 'validation')}
    excluded = set(excluded)
    counts, support, observed = defaultdict(Counter), defaultdict(lambda: defaultdict(set)), []
    effects = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'opportunities.jsonl').open('x') as stream:
        for partition, group in groups_from_run(directory, report):
            source = group[0]['source_sample_id']
            gold = lookup['grpo' if partition in ('train', 'probe') else 'validation'][source]
            counts[partition]['groups'] += 1
            if partition == 'probe':
                observed.append((group[0]['panel_draw'], source))
            rows = [diagnose_scored_row(r, gold) for r in group]
            context, events = repair_effects(rows)
            if source in excluded:
                counts[partition]['excluded_groups'] += 1
            if not events:
                continue
            if source not in excluded:
                counts[partition][context + '_groups'] += 1
                counts[partition][context + '_awards'] += len(events)
                counts[partition][context + '_flips'] += sum(e['turned_positive_under_weak'] for e in events)
                support[partition][context].add(source)
                if context != 'undetermined':
                    reductions = [e['reduction_in_mean_absolute_advantage_error'] for e in events]
                    effects[partition][source][context].append(mean(reductions))
            entry = {'partition': partition, 'source_sample_id': source, 'gold': gold,
                     'draw': group[0].get('panel_draw'), 'excluded': source in excluded, 'context': context,
                     'final_answers': [r['explicit_final_answer'] for r in rows],
                     'members': [r['sample_id'] for r in rows], 'repairs': events}
            stream.write(json.dumps(entry) + '\n')
    if report['stage'] == 'repair_probe':
        panel = report['settings']['repair_panel']
        if observed != [(d, s) for d in range(panel['draws']) for s in panel['sample_ids']]:
            raise ValueError('probe does not cover the declared question and draw schedule')
    summary = {'source_run': str(directory.resolve()), 'manifest': report['manifest_sha256'],
               'excluded_ids': sorted(excluded), 'partitions': {}}
    for part, totals in counts.items():
        ids = support[part]
        summary['partitions'][part] = {**totals, 'question_ids': {k: sorted(v) for k, v in ids.items()},
                                      'same_question_pairs': paired_questions(effects[part])}
    write_json(output / 'metrics.json', summary)
    return summary
