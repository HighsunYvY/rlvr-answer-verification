"""Paired E0-E3 analysis with frozen-ID checks and uncertainty; no causal hacking claim."""
import json
import math
from pathlib import Path
import random
import statistics
from collections import Counter

from .evaluate import (aggregate_metrics, diagnose_final_answer, explicit_final_metrics,
                       score_prediction)
from .verifier import normalize_number, weak_verifier
from .experiment import ROOT, code_revision, file_identity, prepared_inputs, read_json, verify_inventory, write_json


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, center-half), min(1, center+half)]


def paired_difference(left, right, repetitions=2000, seed=42):
    if not left or len(left) != len(right):
        raise ValueError('nonempty aligned pairs required')
    delta = [int(b) - int(a) for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(repetitions))
    gains = sum(d == 1 for d in delta)
    losses = sum(d == -1 for d in delta)
    discordant = gains + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2 ** discordant)) if discordant else 1.0
    return {'difference_right_minus_left': statistics.mean(delta),
            'paired_bootstrap_ci95': [samples[int(.025*repetitions)], samples[int(.975*repetitions)-1]],
            'gains': gains, 'losses': losses, 'exact_mcnemar_p_two_sided': p,
            'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed}


def diagnose_scored_row(row, gold):
    expected = score_prediction(row['sample_id'], row['completion'], gold, row['generated_tokens'])
    if any(key not in row or row[key] != value for key, value in expected.items()):
        raise ValueError(f"scored verifier fields inconsistent for {row['sample_id']}")
    if type(row.get('truncated')) is not bool:
        raise ValueError(f"missing or invalid truncation flag for {row['sample_id']}")
    return {**row, **diagnose_final_answer(expected | {'truncated': row['truncated']})}


def group_reward_evidence(rows, verifier, weights):
    correctness = [row[f'{verifier}_accuracy_reward'] for row in rows]
    rewards = [weights[0] * row['format_reward'] + weights[1] * correct
               for row, correct in zip(rows, correctness, strict=True)]
    center, std = statistics.mean(rewards), statistics.stdev(rewards)
    evidence = {'mean_reward': center, 'sample_std': std,
                'variable_total': std > 0, 'variable_correctness': len(set(correctness)) > 1}
    members = []
    for reward in rewards:
        # TRL 0.24.0, scale_rewards=group. Treat floating-point ties as zero.
        advantage = 0.0 if math.isclose(reward, center, rel_tol=0, abs_tol=1e-12) else (
            (reward - center) / (std + 1e-4))
        members.append({'total_reward': reward, 'advantage': advantage,
                        'sign': 'positive' if advantage > 0 else 'negative' if advantage < 0 else 'zero'})
    return evidence, members


def replay_group_rewards(rows, profile, generations):
    configs = {name: read_json(ROOT / f'configs/grpo_{name}.yaml') for name in ('strict', 'weak')}
    strict, weak = configs['strict'], configs['weak']
    weights = strict['reward_weights']
    if weights != weak['reward_weights'] or len(weights) != 2:
        raise ValueError('reward replay requires matching format and correctness weights')
    for name, config in configs.items():
        trainer = {**config['trainer'], **config['profiles'][profile]}
        if (trainer['num_generations'] != generations or trainer['scale_rewards'] != 'group'
                or config['correctness_verifier'] != name or generations < 2):
            raise ValueError('reward replay requires complete groups and GRPO group scaling')

    groups = {}
    for row in rows:
        groups.setdefault(row['source_sample_id'], []).append(row)
    evidence, counts, transitions = [], Counter(), Counter()
    for source_id, members in groups.items():
        members.sort(key=lambda row: row['group_member'])
        group = {'source_sample_id': source_id, 'sample_ids': [row['sample_id'] for row in members]}
        for row in members:
            row['grpo_replay'] = {}
        for name in configs:
            group[name], rewards = group_reward_evidence(members, name, weights)
            for row, reward in zip(members, rewards, strict=True):
                row['grpo_replay'][name] = reward
        for metric in ('variable_total', 'variable_correctness'):
            strict_has_signal, weak_has_signal = group['strict'][metric], group['weak'][metric]
            counts[f'strict_{metric}'] += strict_has_signal
            counts[f'weak_{metric}'] += weak_has_signal
            counts[f'lost_{metric}'] += strict_has_signal and not weak_has_signal
            counts[f'gained_{metric}'] += weak_has_signal and not strict_has_signal
        for row in members:
            if row['weak_disagreement'] == 'wrong_explicit_final':
                replay = row['grpo_replay']
                transitions[f"{replay['strict']['sign']}->{replay['weak']['sign']}"] += 1
        evidence.append(group)
    return {'groups': len(groups), 'reward_weights': weights, 'scale_rewards': 'group',
            'trl_formula_version': '0.24.0', 'std_epsilon': 1e-4,
            'group_signal_counts': dict(counts),
            'wrong_explicit_final_disagreement_transitions': dict(transitions),
            'wrong_explicit_final_becomes_positive': transitions['negative->positive']
            + transitions['zero->positive']}, evidence


def audit_answers(predictions, preparation_dir, profile, split, generations, output_dir, replay_grpo=False):
    """Re-score a complete frozen split, with one row per question or sampled group member."""
    if type(generations) is not int or generations < 1:
        raise ValueError('generations per question must be a positive integer')
    splits, _, preparation = prepared_inputs(preparation_dir, profile)
    gold = {r['sample_id']: r['ground_truth'] for r in splits[split]}
    rows, members = [], set()
    with Path(predictions).open() as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row['sample_id']
            source_id = row.get('source_sample_id', identifier)
            member = row.get('group_member', 0)
            if (source_id not in gold or type(member) is not int
                    or not 0 <= member < generations or (source_id, member) in members):
                raise ValueError(f'duplicate or unexpected prediction identity: {identifier}')
            expected_id = source_id if generations == 1 else f'{source_id}#g{member}'
            if identifier != expected_id:
                raise ValueError(f'prediction ID does not match its source and group member: {identifier}')
            members.add((source_id, member))
            rows.append(diagnose_scored_row(row, gold[source_id]))
    if len(members) != len(gold) * generations or not rows:
        raise ValueError('predictions must cover every member of the frozen split')

    summary = {'profile': profile, 'split': split, 'questions': len(gold),
               'generations_per_question': generations,
               'manifest_sha256': preparation['split_manifest_sha256'],
               'predictions': str(Path(predictions).resolve()),
               'metrics': {**aggregate_metrics(rows), 'explicit_final': explicit_final_metrics(rows)}}
    if replay_grpo:
        summary['grpo_replay'], reward_groups = replay_group_rewards(rows, profile, generations)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'diagnosed.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    if replay_grpo:
        write_json(out / 'reward-groups.json', reward_groups)
    write_json(out / 'metrics.json', summary)
    return summary


def analyze(evaluations, output_dir, plot=False, preparation_dir=None, training_runs=None,
            weak_audit_run=None, split='test'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    reports, rows = {}, {}
    try:
        if split not in ('test', 'validation'):
            raise ValueError('paired analysis requires test or validation evaluations')
        required = {'E0', 'E1'} if split == 'test' else {'E1'}
        if not required <= set(evaluations) <= {'E0', 'E1', 'E2', 'E3'}:
            raise ValueError(f'{split} analysis requires evaluations for {sorted(required)}')
        for stage, directory in evaluations.items():
            directory = Path(directory)
            report = read_json(directory / 'run.json')
            verify_inventory(directory, report['artifacts'])
            if (report['status'] != 'passed' or report['stage'] != stage + '_eval'
                    or report['settings']['split'] != split):
                raise ValueError(f'complete {split} evaluation required')
            data = [json.loads(line) for line in (directory / 'scored.jsonl').read_text().splitlines()]
            if not data or len({r['sample_id'] for r in data}) != len(data):
                raise ValueError('empty or duplicate sample IDs')
            reports[stage], rows[stage] = report, data
        base_stage = 'E0' if 'E0' in reports else 'E1'
        base = reports[base_stage]
        if not preparation_dir:
            raise ValueError('analysis requires the frozen preparation, never prediction-supplied gold alone')
        frozen, _, prep = prepared_inputs(preparation_dir, base['profile'])
        expected = frozen[split][:4] if base['mode'] == 'smoke' else frozen[split]
        if (prep['split_manifest_sha256'] != base['manifest_sha256']
                or [r['sample_id'] for r in rows[base_stage]] != [r['sample_id'] for r in expected]
                or [r['ground_truth'] for r in rows[base_stage]] != [r['ground_truth'] for r in expected]):
            raise ValueError('analysis frozen ID/gold mismatch')
        ids = [r['sample_id'] for r in rows[base_stage]]
        gold = [r['ground_truth'] for r in rows[base_stage]]
        for stage in evaluations:
            report = reports[stage]
            if (report['manifest_sha256'] != base['manifest_sha256'] or report['settings'] != base['settings']
                    or report['profile'] != base['profile'] or report['mode'] != base['mode']
                    or report['template_sha256'] != base['template_sha256']
                    or report['model_files'] != base['model_files']
                    or [r['sample_id'] for r in rows[stage]] != ids
                    or [r['ground_truth'] for r in rows[stage]] != gold):
                raise ValueError('unpaired decoding/manifest/sample/gold identity')
        if base['mode'] == 'run' and base['profile'] != 'smoke':
            for stage in ('E1','E2','E3'):
                if stage in reports:
                    contract = reports[stage]['policy_identity']['training_contract']
                    if contract['mode'] != 'run' or contract['effective_steps'] != contract['settings']['max_steps']:
                        raise ValueError('main analysis cannot substitute a smoke-trained policy')
        e1 = reports['E1']['policy_identity']['adapter_sha256']
        for stage in ('E2','E3'):
            if stage in reports and reports[stage]['policy_identity']['sft_identity'] != e1:
                raise ValueError('RL initialization is not the fixed E1')
        if ('E2' in reports and 'E3' in reports and
                reports['E2']['policy_identity']['training_contract'] != reports['E3']['policy_identity']['training_contract']):
            raise ValueError('E2/E3 training controls differ')
        summary = {'analysis_code': code_revision(ROOT),
                   'analysis_source_sha256': file_identity(Path(__file__))['sha256'], 'n': len(ids),
                   'split': split, 'profile': base['profile'], 'mode': base['mode'],
                   'manifest_sha256': base['manifest_sha256'], 'conditions': {}, 'paired': {},
                   'paired_explicit_final': {}, 'paired_weak_rewarded_wrong_final': {},
                   'limitations': ['Operational numeric final-answer scoring is not proof of valid reasoning.',
                     'Explicit-final accuracy is conditional on parser coverage; paired comparisons use jointly determined answers.',
                     'Bootstrap intervals and paired tests describe one frozen seed/sample set.',
                     'Reward improvement and verifier disagreements alone do not establish policy exploitation.'],
                   'inputs': {s: file_identity(Path(d) / 'run.json')['sha256'] for s, d in evaluations.items()}}
        summary['missing_conditions'] = sorted({'E0','E1','E2','E3'} - set(reports))
        summary['training'] = {}
        for stage, directory in (training_runs or {}).items():
            trained = read_json(Path(directory) / 'run.json')
            verify_inventory(directory, trained['artifacts'])
            if (stage not in reports or trained['status'] != 'passed' or trained['stage'] != stage
                    or trained['checkpoint'] != reports[stage]['policy_identity']):
                raise ValueError('training diagnostics do not match the evaluated checkpoint')
            summary['training'][stage] = {'metrics': trained['training_metrics'], 'gpu': trained['gpu'],
                'logs': read_json(Path(directory) / 'trainer-log.json'),
                'trainable_parameter_count': sum(math.prod(shape) for shape in trained['trainable'].values()),
                'run_report_sha256': file_identity(Path(directory)/'run.json')['sha256']}
        if weak_audit_run:
            audited = read_json(Path(weak_audit_run)/'run.json')
            verify_inventory(weak_audit_run, audited['artifacts'])
            if (audited['stage'] != 'weak_audit' or audited['status'] != 'passed'
                    or audited['manifest_sha256'] != base['manifest_sha256']
                    or audited['policy_identity'] != reports['E1']['policy_identity']):
                raise ValueError('weak audit does not match analysis inputs')
            summary['weak_audit'] = {'eligible': audited['eligible'], 'metrics': audited['metrics'],
                'report_sha256': file_identity(Path(weak_audit_run)/'run.json')['sha256']}
        frozen_lookup = {r['sample_id']: r for r in expected}
        errors = []
        for stage, data in rows.items():
            data = [diagnose_scored_row(r, frozen_lookup[r['sample_id']]['ground_truth']) for r in data]
            rows[stage] = data
            metrics = aggregate_metrics(data)
            metrics.update(strict_accuracy_wilson_ci95=wilson(sum(r['strict_correct'] for r in data), len(data)),
                           explicit_final=explicit_final_metrics(data),
                           truncation_rate=sum(r['truncated'] for r in data)/len(data),
                           median_tokens=statistics.median(r['generated_tokens'] for r in data),
                           p95_generated_tokens=sorted(r['generated_tokens'] for r in data)[math.ceil(.95*len(data))-1],
                           error_categories=dict(Counter(r['error_type'] for r in data)),
                           mean_strict_total_reward=statistics.mean(.2*r['format_reward']+r['strict_accuracy_reward'] for r in data),
                           mean_weak_total_reward=statistics.mean(.2*r['format_reward']+r['weak_accuracy_reward'] for r in data))
            summary['conditions'][stage] = metrics
            for r in data:
                if not r['strict_correct'] or r['explicit_final_correct'] is not True:
                    question = frozen_lookup[r['sample_id']]['prompt'][1]['content']
                    final_value = r['parsed_answer']
                    errors.append({'stage': stage, **r,
                        'question_gold_overlap': weak_verifier(question, r['ground_truth']),
                        'wrong_final_matches_question_number': bool(not r['strict_correct']
                            and normalize_number(final_value) is not None and weak_verifier(question, final_value))})
        for left, right in [('E0','E1'), ('E1','E2'), ('E1','E3'), ('E2','E3')]:
            if left not in rows or right not in rows:
                continue
            summary['paired'][f'{left}->{right}'] = paired_difference(
                [r['strict_correct'] for r in rows[left]], [r['strict_correct'] for r in rows[right]])
            jointly_determined = [(a['explicit_final_correct'], b['explicit_final_correct'])
                                  for a, b in zip(rows[left], rows[right], strict=True)
                                  if a['explicit_final_correct'] is not None
                                  and b['explicit_final_correct'] is not None]
            comparison = paired_difference(*zip(*jointly_determined)) if jointly_determined else None
            summary['paired_explicit_final'][f'{left}->{right}'] = {
                'jointly_determined': len(jointly_determined), 'total_pairs': len(ids),
                'comparison': comparison}
            summary['paired_weak_rewarded_wrong_final'][f'{left}->{right}'] = paired_difference(*[
                [r['weak_positive'] and r['explicit_final_correct'] is False for r in rows[stage]]
                for stage in (left, right)])
        with (out / 'errors.jsonl').open('x') as stream:
            for r in errors:
                stream.write(json.dumps(r, ensure_ascii=False)+'\n')
        write_json(out / 'summary.json', summary)
        lines = ['# Paired E0-E3 evaluation', '', f"Frozen {split} samples: {len(ids)}; profile: {base['profile']}; mode: {base['mode']}.", '',
                 '| Condition | Strict accuracy | Format | Weak FP share | Truncation |', '|---|---:|---:|---:|---:|']
        for stage, m in summary['conditions'].items():
            lines.append(f"| {stage} | {m['strict_accuracy']:.4f} | {m['format_rate']:.4f} | {m['weak_false_positive_share']:.4f} | {m['truncation_rate']:.4f} |")
        lines += ['', '## Explicit final answers', '',
                  '| Condition | Accuracy on determined | Coverage | Undetermined | Weak-rewarded wrong finals |',
                  '|---|---:|---:|---:|---:|']
        for stage, m in summary['conditions'].items():
            final = m['explicit_final']
            determined = final['correct'] + final['wrong']
            accuracy = f"{final['correct']}/{determined}" if determined else 'N/A'
            lines.append(f"| {stage} | {accuracy} | {final['coverage']:.1%} | {final['undetermined']} | "
                         f"{final['weak_rewarded_wrong_final_count']} |")
        lines += ['', '## Paired changes', '']
        for pair, result in summary['paired'].items():
            lines.append(f"- {pair}: delta={result['difference_right_minus_left']:.4f}, paired bootstrap 95% CI={result['paired_bootstrap_ci95']}, exact McNemar p={result['exact_mcnemar_p_two_sided']:.6g}.")
        lines += ['', f"Missing conditions: {summary['missing_conditions']}; no scores imputed."]
        if 'weak_audit' in summary:
            lines += ['', f"Weak audit eligibility: {summary['weak_audit']['eligible']}."]
        lines += ['', '## Limitations', '', *('- '+s for s in summary['limitations'])]
        (out / 'report.md').write_text('\n'.join(lines)+'\n')
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            stages = list(summary['conditions'])
            values = [summary['conditions'][s]['strict_accuracy'] for s in stages]
            cis = [summary['conditions'][s]['strict_accuracy_wilson_ci95'] for s in stages]
            ax.bar(stages, values, yerr=[[max(0,v-c[0]) for v,c in zip(values,cis)], [max(0,c[1]-v) for v,c in zip(values,cis)]], capsize=4)
            ax.set(ylim=(0,1), ylabel='Strict accuracy', title='Frozen paired evaluation (Wilson 95% CI)')
            fig.tight_layout()
            fig.savefig(out / 'accuracy.png', dpi=180)
            plt.close(fig)
            if summary['training']:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for ax, key in zip(axes, ('reward', 'kl'), strict=True):
                    found = False
                    for stage, diagnostics in summary['training'].items():
                        points = [(r['step'], r[key]) for r in diagnostics['logs'] if key in r and 'step' in r]
                        if points:
                            ax.plot([p[0] for p in points], [p[1] for p in points], label=stage)
                            found = True
                    ax.set(xlabel='Optimizer step', ylabel=key)
                    if found:
                        ax.legend()
                    else:
                        ax.text(.5,.5,'Not exported in supplied logs',ha='center',transform=ax.transAxes)
                fig.tight_layout()
                fig.savefig(out / 'training-reward-kl.png', dpi=180)
                plt.close(fig)
        write_json(out / 'analysis.json', {'status': 'passed', 'inputs': summary['inputs']})
        return summary
    except BaseException as exc:
        write_json(out / 'analysis.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
