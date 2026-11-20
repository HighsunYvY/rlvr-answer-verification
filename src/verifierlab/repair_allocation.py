"""Exact fixed-budget comparison of scattered and whole-group verifier corrections."""
import argparse
import json
from collections import defaultdict
from itertools import combinations, islice
from math import comb, isclose, sqrt
from pathlib import Path
from statistics import mean

from .analysis import diagnose_scored_row, group_reward_evidence
from .experiment import file_identity, prepared_inputs, read_json, verify_inventory, write_json
from .repair_replay import groups_from_run
from .verifier import normalize_number


def advantage(rows, rule):
    _, members = group_reward_evidence(rows, rule, [0.2, 1.0])
    return [item['advantage'] for item in members]


def group_curve(rows, flags, gram):
    target = advantage(rows, 'target')
    curve, residual_means = [], []
    for budget in range(len(flags) + 1):
        metrics, residuals = defaultdict(list), []
        for chosen in combinations(flags, budget):
            patched = [dict(r, selected_accuracy_reward=0. if i in chosen else r['weak_accuracy_reward'])
                       for i, r in enumerate(rows)]
            error = [a-b for a, b in zip(advantage(patched, 'selected'), target, strict=True)]
            metrics['reward_mae'].append(mean(abs(r['selected_accuracy_reward'] - r['target_accuracy_reward'])
                                              for r in patched))
            metrics['advantage_mae'].append(mean(map(abs, error)))
            residuals.append(error)
            metrics['advantage_mse'].append(mean(d*d for d in error))
            metrics['advantage_rms'].append(sqrt(metrics['advantage_mse'][-1]))
            if gram is not None:
                metrics['gradient_mse'].append(gradient_squared_norm(error, gram))
                metrics['gradient_norm'].append(sqrt(metrics['gradient_mse'][-1]))
        curve.append({name: mean(values) for name, values in metrics.items()})
        residual_means.append(list(map(mean, zip(*residuals, strict=True))))
    return curve, residual_means, gram


def gradient_squared_norm(error, gram):
    squared = sum(di*dj*gram[i][j] for i, di in enumerate(error) for j, dj in enumerate(error))
    return max(0., squared) / len(error)**2


def expected_metrics(profile, probabilities):
    curve, residual_means, gram = profile
    estimates = {k: sum(p*v[k] for p, v in zip(probabilities, curve, strict=True)) for k in curve[0]}
    expected_residual = [sum(p*x for p,x in zip(probabilities, column, strict=True))
                         for column in zip(*residual_means, strict=True)]
    estimates['advantage_squared_bias'] = mean(x*x for x in expected_residual)
    estimates['advantage_variance'] = max(0., estimates['advantage_mse'] - estimates['advantage_squared_bias'])
    if gram is not None:
        estimates['gradient_squared_bias'] = gradient_squared_norm(expected_residual, gram)
        estimates['gradient_variance'] = max(0., estimates['gradient_mse'] - estimates['gradient_squared_bias'])
    return estimates


def load_gradients(directory, source, source_report):
    if directory is None:
        return {}
    path = Path(directory)
    report = read_json(path/'run.json')
    if (report['status'] != 'passed' or report['mode'] != 'run' or report['stage'] != 'repair_gradient'
            or not report['gradient_validated'] or report['policy_identity'] != source['policy_identity']
            or report['manifest_sha256'] != source['manifest_sha256']
            or report['settings']['source_report'] != file_identity(source_report)['sha256']):
        raise ValueError('gradient probe must use this exact frozen answer bank')
    filename = 'group-gradients.jsonl'
    verify_inventory(path, {filename: report['artifacts'][filename]})
    with (path / filename).open() as stream:
        records = {(r['sample_id'], r['draw']): r for r in map(json.loads, stream)}
    if len(records) != report['groups']:
        raise ValueError('gradient probe group count mismatch')
    return records


def sampled_groups(path, source, splits):
    if source['stage'] == 'repair_probe' and source['settings']['split'] == 'grpo':
        panel = source['settings']['repair_panel']
        schedule = [(q, d) for d in range(panel['draws']) for q in panel['sample_ids']]
        for (_, group), (key, draw) in zip(groups_from_run(path, source), schedule, strict=True):
            if any((r['source_sample_id'], r['panel_draw']) != (key, draw) for r in group):
                raise ValueError('frozen draw schedule mismatch')
            yield key, draw, group
    elif source['stage'] == 'E5_eval' and source['settings']['split'] == 'test':
        expected = [q['sample_id'] for q in splits['test']]
        if source['n'] != len(expected)*4:
            raise ValueError('incomplete held-out evaluation')
        with (path/'scored.jsonl').open() as stream:
            for key in expected:
                group = list(map(json.loads, islice(stream, 4)))
                identity = [(r['sample_id'], r['source_sample_id'], r['group_member']) for r in group]
                if identity != [(f'{key}#g{i}', key, i) for i in range(4)]:
                    raise ValueError('unexpected held-out question or group order')
                yield key, 0, group
            if next(stream, None) is not None:
                raise ValueError('unexpected trailing held-out answers')
    else:
        raise ValueError('need an E1 training probe or a frozen E5 test evaluation')


def allocation(probe_dir, preparation_dir, output, gradient_dir=None):
    path = Path(probe_dir)
    source_report = path / 'run.json'
    source = read_json(source_report)
    splits, _, prep = prepared_inputs(preparation_dir, 'full')
    if (source['status'] != 'passed' or source['mode'] != 'run'
            or source['settings']['generation']['num_return_sequences'] != 4
            or source['manifest_sha256'] != prep['split_manifest_sha256']):
        raise ValueError('need complete four-answer sampling on the frozen preparation')
    verify_inventory(path, {'scored.jsonl': source['artifacts']['scored.jsonl']})
    labels = {r['sample_id']: r['ground_truth'] for r in splits[source['settings']['split']]}
    gram_rows = load_gradients(gradient_dir, source, source_report)
    strata = defaultdict(list)
    sampled = 0
    for key, draw, group in sampled_groups(path, source, splits):
        sampled += 1
        # Selection uses only the two operational verifiers and the numeric tag.
        flags = [i for i, r in enumerate(group) if r['weak_positive'] and not r['strict_correct']
                 and normalize_number(r['parsed_answer']) is not None]
        if len(flags) < 2:
            continue
        rows = [diagnose_scored_row(r, labels[key]) for r in group]
        curve = None
        if all(r['explicit_final_correct'] is not None for r in rows):
            for row in rows:
                row['target_accuracy_reward'] = float(row['explicit_final_correct'])
            gram = None
            if gradient_dir is not None:
                record = gram_rows[key, draw]
                if (record['member_ids'] != [r['sample_id'] for r in rows]
                        or set(flags) != {event['member'] for event in record['events']}):
                    raise ValueError('gradient sample identity or flagged responses differ')
                gram = record['gram']
            curve = group_curve(rows, flags, gram)
        strata[len(flags)].append((key, draw, curve))

    summaries, aggregate = {}, defaultdict(lambda: defaultdict(list))
    total_budget = 0
    for size, groups in sorted(strata.items()):
        count = len(groups)
        known = [(q, d, curve) for q, d, curve in groups if curve is not None]
        stratum = {'groups': count, 'fully_labeled_groups': len(known),
                   'unresolved_group_ids': [(q,d) for q,d,curve in groups if curve is None]}
        summaries[size] = stratum
        # Nearest whole number of groups to one third of this stratum.
        group_budget = (count + 1) // 3
        if not group_budget:
            stratum['withdrawals'] = 0
            continue
        population = count * size
        budget = group_budget * size
        total_budget += budget
        probability = group_budget / count
        # Number of withdrawals in this group when picking budget items out of population.
        probs = [comb(size, h)*comb(population-size, budget-h)/comb(population, budget)
                 if 0 <= budget-h <= population-size else 0. for h in range(size+1)]
        if not isclose(sum(probs), 1.) or not isclose(sum(h*p for h,p in enumerate(probs)), size*probability):
            raise ValueError('invalid fixed-budget distribution')
        stratum.update(withdrawals=budget, probability_per_flag=probability)
        outcomes = defaultdict(lambda: defaultdict(list))
        distributions = {'original':[1.] + [0.]*size, 'individual':probs,
                         'whole_groups':[1-probability] + [0.]*(size-1) + [probability]}
        for _, _, profile in known:
            for policy, probabilities in distributions.items():
                for metric, value in expected_metrics(profile, probabilities).items():
                    outcomes[metric][policy].append(value)
                    aggregate[metric][policy].append(value)
        stratum['expected_error_per_labeled_group'] = {
            m: {p: mean(v) for p,v in policies.items()} for m, policies in outcomes.items()}
    result = {'source': str(path), 'gradient_source': str(gradient_dir) if gradient_dir else None,
              'sampled_groups':sampled, 'withdrawals_per_policy': total_budget, 'strata':summaries,
              'expected_error_per_labeled_group_in_allocated_strata':{
                  m: {p: mean(v) for p,v in policies.items()} for m, policies in aggregate.items()}}
    if 'reward_mae' in aggregate:
        reward = result['expected_error_per_labeled_group_in_allocated_strata']['reward_mae']
        if not isclose(reward['individual'], reward['whole_groups'], abs_tol=1e-12):
            raise ValueError('matched inclusion probabilities must preserve expected individual reward error')
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('probe-dir', 'preparation-dir', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--gradient-dir')
    result = allocation(**vars(parser.parse_args()))
    print(json.dumps(result['expected_error_per_labeled_group_in_allocated_strata'], indent=2))
