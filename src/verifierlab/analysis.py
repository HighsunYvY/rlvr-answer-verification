"""Paired E0-E3 analysis with frozen-ID checks and uncertainty; no causal hacking claim."""
import json
import math
from pathlib import Path
import random
import statistics
from collections import Counter

from .evaluate import aggregate_metrics, score_prediction
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


def analyze(evaluations, output_dir, plot=False, preparation_dir=None, training_runs=None, weak_audit_run=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    reports, rows = {}, {}
    try:
        if not {'E0','E1'} <= set(evaluations) or not set(evaluations) <= {'E0','E1','E2','E3'}:
            raise ValueError('E0/E1 evaluations required; E2/E3 may be absent for negative results')
        for stage, directory in evaluations.items():
            directory = Path(directory)
            report = read_json(directory / 'run.json')
            verify_inventory(directory, report['artifacts'])
            if (report['status'] != 'passed' or report['stage'] != stage + '_eval'
                    or report['settings']['split'] != 'test'):
                raise ValueError('complete held-out test evaluation required')
            data = [json.loads(line) for line in (directory / 'scored.jsonl').read_text().splitlines()]
            if not data or len({r['sample_id'] for r in data}) != len(data):
                raise ValueError('empty or duplicate sample IDs')
            for r in data:
                expected = score_prediction(r['sample_id'], r['completion'], r['ground_truth'], r['generated_tokens'])
                if any(r[k] != v for k, v in expected.items()):
                    raise ValueError('scored verifier fields inconsistent')
            reports[stage], rows[stage] = report, data
        base = reports['E0']
        if not preparation_dir:
            raise ValueError('analysis requires the frozen preparation, never prediction-supplied gold alone')
        frozen, _, prep = prepared_inputs(preparation_dir, base['profile'])
        expected = frozen['test'][:4] if base['mode'] == 'smoke' else frozen['test']
        if (prep['split_manifest_sha256'] != base['manifest_sha256']
                or [r['sample_id'] for r in rows['E0']] != [r['sample_id'] for r in expected]
                or [r['ground_truth'] for r in rows['E0']] != [r['ground_truth'] for r in expected]):
            raise ValueError('analysis frozen ID/gold mismatch')
        ids = [r['sample_id'] for r in rows['E0']]
        gold = [r['ground_truth'] for r in rows['E0']]
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
                   'analysis_source_sha256': file_identity(Path(__file__))['sha256'], 'n': len(ids), 'profile': base['profile'], 'mode': base['mode'],
                   'manifest_sha256': base['manifest_sha256'], 'conditions': {}, 'paired': {},
                   'limitations': ['Operational numeric final-answer scoring is not proof of valid reasoning.',
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
            metrics = aggregate_metrics(data)
            metrics.update(strict_accuracy_wilson_ci95=wilson(sum(r['strict_correct'] for r in data), len(data)),
                           truncation_rate=sum(r['truncated'] for r in data)/len(data),
                           median_tokens=statistics.median(r['generated_tokens'] for r in data),
                           p95_generated_tokens=sorted(r['generated_tokens'] for r in data)[math.ceil(.95*len(data))-1],
                           error_categories=dict(Counter(r['error_type'] for r in data)),
                           mean_strict_total_reward=statistics.mean(.2*r['format_reward']+r['strict_accuracy_reward'] for r in data),
                           mean_weak_total_reward=statistics.mean(.2*r['format_reward']+r['weak_accuracy_reward'] for r in data))
            summary['conditions'][stage] = metrics
            for r in data:
                if not r['strict_correct'] or r['truncated']:
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
        with (out / 'errors.jsonl').open('x') as stream:
            for r in errors:
                stream.write(json.dumps(r, ensure_ascii=False)+'\n')
        write_json(out / 'summary.json', summary)
        lines = ['# Paired E0-E3 evaluation', '', f"Frozen samples: {len(ids)}; profile: {base['profile']}; mode: {base['mode']}.", '',
                 '| Condition | Strict accuracy | Format | Weak FP share | Truncation |', '|---|---:|---:|---:|---:|']
        for stage, m in summary['conditions'].items():
            lines.append(f"| {stage} | {m['strict_accuracy']:.4f} | {m['format_rate']:.4f} | {m['weak_false_positive_share']:.4f} | {m['truncation_rate']:.4f} |")
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
