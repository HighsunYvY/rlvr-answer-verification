"""Select E5 training prompts from frozen E1 probes and compare held-out behavior."""
import json
from collections import Counter
from itertools import combinations
from math import dist, sqrt
from pathlib import Path
from statistics import mean

from .analysis import diagnose_scored_row, group_reward_evidence
from .attribution import paired_prompt_interval
from .experiment import prepared_inputs, read_json, write_json
from .repair_replay import groups_from_run


def select_cohort(probes, preparation_dir, output):
    splits, _, preparation = prepared_inputs(preparation_dir, 'full')
    gold = {r['sample_id']: r['ground_truth'] for r in splits['grpo']}
    counts, reports = {}, []
    for path in map(Path, probes):
        report = read_json(path / 'run.json')
        reports.append(report)
        if (report['stage'] != 'repair_probe' or report['mode'] != 'run' or report['status'] != 'passed'
                or report['manifest_sha256'] != preparation['split_manifest_sha256']
                or report['policy_identity'] != reports[0]['policy_identity']):
            raise ValueError('cohort selection requires complete probes of the same E1')
        panel = report['settings']['repair_panel']
        if set(counts) & set(panel['sample_ids']):
            raise ValueError('overlapping probe panels')
        expected = [(i, d) for d in range(panel['draws']) for i in panel['sample_ids']]
        for (_, members), (key, draw) in zip(groups_from_run(path, report), expected, strict=True):
            tally = counts.setdefault(key, Counter())
            for r in members:
                if r['source_sample_id'] != key or r['panel_draw'] != draw:
                    raise ValueError('probe question or draw mismatch')
                row = diagnose_scored_row(r, gold[key])
                if (row['format_compliant'] and not row['truncated']
                        and row['explicit_final_correct'] is row['strict_correct']):
                    tally['correct' if row['strict_correct'] else 'wrong'] += 1
                    tally['paid_wrong'] += row['weak_positive'] and not row['strict_correct']
    selected = sorted(k for k, v in counts.items() if v['correct'] and v['wrong'] >= 4 and v['paid_wrong'])
    result = {'train_ids': selected, 'counts': counts, 'manifest': preparation['split_manifest_sha256']}
    write_json(output, result)
    return result


def repair_curve(probe_dir, preparation_dir, output):
    """Enumerate each possible budget of wrong-award removals on unchanged E1 groups."""
    path = Path(probe_dir)
    report = read_json(path / 'run.json')
    splits, _, prep = prepared_inputs(preparation_dir, 'full')
    if (report['stage'] != 'repair_probe' or report['mode'] != 'run' or report['status'] != 'passed'
            or report['manifest_sha256'] != prep['split_manifest_sha256']):
        raise ValueError('need a complete E1 training probe')
    panel = report['settings']['repair_panel']
    gold = {q['sample_id']: q['ground_truth'] for q in splits['grpo']}
    expected = [(q, d) for d in range(panel['draws']) for q in panel['sample_ids']]
    curves = []
    for (_, rows), (question, draw) in zip(groups_from_run(path, report), expected, strict=True):
        if any(r['source_sample_id'] != question or r['panel_draw'] != draw for r in rows):
            raise ValueError('group differs from the frozen draw schedule')
        rows = [diagnose_scored_row(r, gold[question]) for r in rows]
        labels = [r['explicit_final_correct'] for r in rows]
        candidates = [i for i, r in enumerate(rows) if r['weak_positive'] and labels[i] is False
                      and not r['strict_correct']]
        if None in labels or not candidates:
            continue
        for r, correct in zip(rows, labels, strict=True):
            r['target_accuracy_reward'] = float(correct)
        _, oracle = group_reward_evidence(rows, 'target', [0.2, 1.0])
        target = [r['advantage'] for r in oracle]
        budgets = []
        for budget in range(len(candidates)+1):
            absolute, rms = [], []
            for removed in combinations(candidates, budget):
                patched = [dict(r, repair_accuracy_reward=0. if i in removed else r['weak_accuracy_reward'])
                           for i, r in enumerate(rows)]
                _, evidence = group_reward_evidence(patched, 'repair', [0.2, 1.0])
                actual = [r['advantage'] for r in evidence]
                absolute.append(mean(abs(v-t) for v,t in zip(actual,target,strict=True)))
                rms.append(dist(actual,target) / sqrt(len(target)))
            budgets.append({'removed': budget, 'subsets': len(absolute), 'mean_absolute_error': mean(absolute),
                            'mean_rms_error': mean(rms)})
        curves.append({'source_sample_id': question, 'draw': draw, 'correct_finals': sum(labels),
                       'weak_awards': sum(r['weak_positive'] for r in rows), 'eligible': len(candidates),
                       'equal_format': len({r['format_reward'] for r in rows}) == 1, 'budgets': budgets})
    result = {'groups_screened': len(expected), 'eligible_groups': len(curves), 'curves': curves}
    write_json(output, result)
    return {'groups_screened': len(expected), 'eligible_groups': len(curves)}


def compare_repair_evaluations(left_paths, right_paths, preparation_dir, output):
    splits, _, prep = prepared_inputs(preparation_dir, 'full')
    seeds, all_deltas, comparison = {}, [], None
    reference = None
    for left, right in zip(left_paths, right_paths, strict=True):
        results, policies, budgets = [], [], []
        pair = None
        for directory in (Path(left), Path(right)):
            r = read_json(directory / 'run.json')
            settings = r['settings']
            groups = settings['generation']['num_return_sequences']
            split = settings['split']
            if (r['stage'] != 'E5_eval' or r['status'] != 'passed' or r['mode'] != 'run'
                    or split != 'test' or groups != 4 or r['manifest_sha256'] != prep['split_manifest_sha256']):
                raise ValueError('need complete E5 held-out evaluations')
            policy = r['policy_identity']
            contract = policy['training_contract']
            train_settings = dict(contract['settings'])
            if contract['mode'] != 'run' or contract['effective_steps'] != train_settings['max_steps']:
                raise ValueError('incomplete training checkpoint')
            plan = dict(train_settings.pop('repair_plan'))
            policies.append(plan.pop('policy'))
            budgets.append(plan.pop('caps'))
            seed = train_settings['seed']
            control = (train_settings, plan, contract['effective_steps'])
            if pair is not None and pair != control:
                raise ValueError('training conditions are not paired')
            pair = control
            identity = (r['resolved_generation'], r['template_sha256'], settings,
                        policy['sft_identity'], r['model_files'], r['code']['commit'])
            if reference is not None and reference != identity:
                raise ValueError('evaluation identity, E1 parent or decoding mismatch')
            reference = identity
            totals, correct_by_prompt = Counter(), []
            with (directory / 'scored.jsonl').open() as stream:
                for frozen in splits[split]:
                    key = frozen['sample_id']
                    correct = 0
                    for member in range(groups):
                        row = json.loads(next(stream))
                        if (row['sample_id'], row['source_sample_id'], row['group_member']) != (f'{key}#g{member}', key, member):
                            raise ValueError('evaluation sample identity mismatch')
                        row = diagnose_scored_row(row, frozen['ground_truth'])
                        outcome = row['explicit_final_correct']
                        correct += outcome is True
                        totals['undetermined'] += outcome is None
                        totals['weak_rewarded_wrong'] += row['weak_positive'] and outcome is False
                        totals['strict_correct'] += row['strict_correct']
                    correct_by_prompt.append(correct / groups)
                    totals['explicit_correct'] += correct
                    totals['pass_at_4_questions'] += bool(correct)
                if next(stream, None) is not None or r['n'] != len(correct_by_prompt) * groups:
                    raise ValueError('evaluation size mismatch')
            results.append((correct_by_prompt, totals))
        if seed in seeds or (comparison is not None and comparison != policies):
            raise ValueError('duplicate seed or inconsistent policy pair')
        comparison = policies
        delta = [b-a for a,b in zip(results[0][0], results[1][0], strict=True)]
        all_deltas.append(delta)
        seeds[seed] = {'counts': dict(zip(policies, (x[1] for x in results), strict=True)),
                       'configured_caps': dict(zip(policies, budgets, strict=True)), 'difference': mean(delta),
                       'paired_question_ci95': paired_prompt_interval(delta)}
    if not seeds:
        raise ValueError('one or more paired seeds required')
    average = list(map(mean, zip(*all_deltas, strict=True)))
    result = {'right_minus_left': comparison[::-1], 'seeds': seeds, 'mean_difference': mean(average),
              'paired_question_ci95_fixed_seeds': paired_prompt_interval(average)}
    write_json(output, result)
    return result
