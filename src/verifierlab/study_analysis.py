"""Summarize verifier-study evidence using previously generated answers or gradients."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from .analysis import diagnose_scored_row
from .experiment import prepared_inputs, read_json, verify_inventory, write_json
from .repair_policy import tag_priority_accuracy_reward
from .repair_replay import groups_from_run


def legacy_pairs(audit_pairs):
    """Summarize the older E2/E3 development comparison from its per-run audits."""
    seeds, evaluation_identity = {}, None
    for seed, strict_train, strict_eval, weak_train, weak_eval in audit_pairs:
        seed = int(seed)
        if seed in seeds:
            raise ValueError('duplicate legacy seed')
        counts = {}
        for stage, train_path, eval_path in (('E2', strict_train, strict_eval),
                                            ('E3', weak_train, weak_eval)):
            train, evaluation = read_json(train_path), read_json(eval_path)
            identity = (evaluation['manifest_sha256'], evaluation['questions'],
                        evaluation['generations_per_question'], evaluation['metrics']['n'])
            if (train['stage'] != stage or train['optimizer_steps'] != 100
                    or evaluation['split'] != 'validation' or identity[1:] != (128, 4, 512)
                    or (evaluation_identity is not None and identity != evaluation_identity)):
                raise ValueError('legacy audit pair does not share the 100-step development protocol')
            evaluation_identity = identity
            counts[stage] = {
                'wrong_awards_with_positive_training_advantage':
                    train['training'].get('weak_rewarded_wrong_positive_under_applied', 0),
                'weak_awarded_wrong_evaluation_answers':
                    evaluation['metrics']['explicit_final']['weak_rewarded_wrong_final_count'],
            }
        key = 'weak_awarded_wrong_evaluation_answers'
        seeds[seed] = {'conditions': counts,
                       'difference_E3_minus_E2': (counts['E3'][key] - counts['E2'][key]) / identity[-1]}
    return {'seeds': seeds, 'mean_difference_E3_minus_E2': mean(
        details['difference_E3_minus_E2'] for details in seeds.values())}


def panel_rewards(probe_dir, preparation_dir, review_file=None):
    directory = Path(probe_dir)
    run = read_json(directory / 'run.json')
    splits, _, preparation = prepared_inputs(preparation_dir, 'full')
    if (run['stage'] != 'repair_probe' or run['status'] != 'passed' or run['mode'] != 'run'
            or run['manifest_sha256'] != preparation['split_manifest_sha256']):
        raise ValueError('a complete training-panel probe on this frozen split is required')
    evidence = run['artifacts']['scored.jsonl']
    verify_inventory(directory, {'scored.jsonl': evidence})
    decisions = {}
    if review_file is not None:
        review = read_json(review_file)
        if (review['source_commit'] != run['code']['commit']
                or review['source_scored_artifact'] != evidence):
            raise ValueError('review decisions apply to a different archived answer bank')
        decisions = review['decisions']
    gold = {row['sample_id']: row['ground_truth'] for row in splits['grpo']}
    panel = run['settings']['repair_panel']
    schedule = [(q, d) for d in range(panel['draws']) for q in panel['sample_ids']]
    automatic, reviewed, unresolved = Counter(), Counter(), set()
    for (_, members), (key, draw) in zip(groups_from_run(directory, run), schedule, strict=True):
        for row in members:
            if (row['source_sample_id'], row['panel_draw']) != (key, draw):
                raise ValueError('frozen panel schedule mismatch')
            scored = diagnose_scored_row(row, gold[key])
            correct = scored['explicit_final_correct']
            label = {True: 'correct', False: 'wrong', None: 'unresolved'}[correct]
            if correct is None:
                unresolved.add(row['sample_id'])
            alternate = decisions.get(row['sample_id'], label)
            if correct is not None and row['sample_id'] in decisions:
                raise ValueError('only parser-undetermined answers can be reviewed')
            if alternate not in ('correct', 'wrong', 'unresolved'):
                raise ValueError('unknown review label')
            for totals, outcome in ((automatic, label), (reviewed, alternate)):
                totals[outcome] += 1
                totals[f'weak_{outcome}'] += int(scored['weak_positive'])
                totals[f'tag_priority_{outcome}'] += int(tag_priority_accuracy_reward(scored))
    if review_file is not None and unresolved != decisions.keys():
        raise ValueError('archived review does not cover the exact unresolved answers')
    return {'groups': len(schedule), 'automatic': automatic,
            'post_hoc_review': reviewed if review_file is not None else None}


def gradient_directions(run_dirs):
    seeds, directions, reference = {}, {}, None
    for path in map(Path, run_dirs):
        run = read_json(path / 'run.json')
        if (run['stage'] != 'repair_gradient' or run['status'] != 'passed' or run['mode'] != 'run'
                or not run['gradient_validated']):
            raise ValueError('a complete validated gradient probe is required')
        seed = run['settings']['seed']
        identity = (run['settings']['source_report'], run['policy_identity'], run['manifest_sha256'])
        if seed in seeds or (reference is not None and identity != reference):
            raise ValueError('gradient seeds must be distinct and use the same frozen answer bank')
        reference = identity
        verify_inventory(path, {'group-gradients.jsonl': run['artifacts']['group-gradients.jsonl']})
        outcomes, farther = {}, set()
        with (path / 'group-gradients.jsonl').open() as stream:
            for group in map(json.loads, stream):
                key = (group['sample_id'], group['draw'], tuple(group['member_ids']))
                if key in outcomes:
                    raise ValueError('duplicate gradient group')
                events = group['events']
                before = mean(e['gradient_error_before'] for e in events)
                after = mean(e['gradient_error_after'] for e in events)
                tolerance = .001 * max(before, after)
                change = after - before
                direction = 'farther' if change > tolerance else 'closer' if change < -tolerance else 'tie'
                outcomes[key] = direction
                if direction == 'farther':
                    farther.add(group['sample_id'])
        if len(outcomes) != run['groups'] or any(x.keys() != outcomes.keys() for x in directions.values()):
            raise ValueError('gradient probes do not have identical group coverage')
        seeds[seed] = {'groups': len(outcomes), 'directions': Counter(outcomes.values()),
                       'farther_distinct_questions': len(farther),
                       'direct_gradient_relative_error': run['direct_gradient_relative_error']}
        directions[seed] = outcomes
    # Align by the complete member identity even if group files have different row order.
    keys = next(iter(directions.values()))
    agreement = sum(len({x[key] for x in directions.values()}) == 1 for key in keys)
    return {'seeds': seeds, 'groups_with_agreement': agreement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    panel = commands.add_parser('panel-rewards')
    panel.add_argument('--probe-dir', required=True)
    panel.add_argument('--preparation-dir', required=True)
    panel.add_argument('--review-file', help='optional archived post-hoc final-answer decisions')
    panel.add_argument('--output', required=True)
    gradient = commands.add_parser('gradient-directions')
    gradient.add_argument('--run-dir', action='append', required=True)
    gradient.add_argument('--output', required=True)
    legacy = commands.add_parser('legacy-pairs')
    legacy.add_argument('--pair', action='append', nargs=5, required=True,
                        metavar=('SEED', 'E2_TRAIN_AUDIT', 'E2_EVAL_AUDIT',
                                 'E3_TRAIN_AUDIT', 'E3_EVAL_AUDIT'))
    legacy.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'panel-rewards':
        result = panel_rewards(args.probe_dir, args.preparation_dir, args.review_file)
    elif args.command == 'gradient-directions':
        result = gradient_directions(args.run_dir)
    else:
        result = legacy_pairs(args.pair)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
