"""Print the experiment sequence for the GSM8K verifier study or materialize an E5 reward plan."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex

from .experiment import ROOT, read_json, verify_inventory, write_json


REPAIR_CONFIG_DIR = Path('configs/gsm8k-1.5b-reward-repair')
REPAIR_CONFIG = REPAIR_CONFIG_DIR / 'repair.json'
LEGACY_CODE = '267ab664db4d9cf3ee87f331cc147de390131b33'


def repair_plan(config, seed, policy, all_wrong_run=None, all_wrong_audit=None):
    design = read_json(config)
    if seed not in design['seeds'] or policy not in design['policies']:
        raise ValueError('seed or policy is outside the frozen study design')
    caps = design['all_wrong_caps']
    plan = {'policy': policy, 'window_steps': design['window_steps'],
            'train_ids': design['train_ids'], 'validation_ids': design['validation_ids']}
    if policy == 'all_wrong':
        return plan | {'caps': caps}
    if policy != 'mixed':
        return plan | {'caps': [0] * len(caps)}
    if all_wrong_run is None or all_wrong_audit is None:
        raise ValueError('mixed requires the completed same-seed all-wrong run and its training audit')

    run_dir = Path(all_wrong_run)
    run = read_json(run_dir / 'run.json')
    audit = read_json(all_wrong_audit)
    expected = plan | {'policy': 'all_wrong', 'caps': caps}
    if (run['stage'] != 'E5' or run['status'] != 'passed' or run['mode'] != 'run'
            or run['settings']['seed'] != seed or run['settings']['repair_plan'] != expected
            or run['optimizer_steps'] != design['max_steps']
            or audit['stage'] != 'E5' or audit['applied_verifier'] != 'repair'
            or Path(audit['source_run']).name != run_dir.name
            or audit['window_steps'] != design['window_steps']
            or audit['optimizer_steps'] != design['max_steps']
            or len(audit['windows']) != len(caps)):
        raise ValueError('the all-wrong audit must match a completed training run of this seed and plan')
    matched = []
    width = design['window_steps']
    for i, (window, cap) in enumerate(zip(audit['windows'], caps, strict=True)):
        count = window.get('withdrawn_awards', 0)
        if (window['optimizer_updates'] != [i * width + 1, min((i + 1) * width, design['max_steps'])]
                or type(count) is not int or not 0 <= count <= cap):
            raise ValueError('invalid all-wrong withdrawal window')
        matched.append(count)
    if sum(matched) != audit['training'].get('withdrawn_awards', 0):
        raise ValueError('window totals differ from recorded withdrawals')
    ledger = 'reward-groups.jsonl'
    verify_inventory(run_dir, {ledger: run['artifacts'][ledger]})
    observed = Counter()
    with (run_dir / ledger).open() as stream:
        for call in map(json.loads, stream):
            for group in call['groups']:
                if group['partition'] == 'train':
                    observed[call['optimizer_step'] // width] += len(group['repair']['changed_indices'])
    if matched != [observed[i] for i in range(len(caps))]:
        raise ValueError('audited withdrawal counts differ from the selected all-wrong run')
    return plan | {'caps': matched}


class CommandPlan:
    def __init__(self, args):
        self.python = args.python
        self.out = Path(args.work_dir)
        self.baseline = Path(args.baseline_dir) if args.baseline_dir else self.out
        self.e1 = Path(args.sft_run) if args.sft_run else self.baseline / 'e1-train'
        self.bank = Path(args.probe_dir) if args.probe_dir else self.out / 'random-panel'
        self.prep = Path(args.preparation_dir)
        self.model = Path(args.model_dir)
        self.config = read_json(ROOT / REPAIR_CONFIG)

    def command(self, module, *args):
        print(shlex.join((self.python, '-m', f'verifierlab.{module}', *map(str, args))))

    def stage(self, name, module, arguments):
        run = self.out / name
        smoke = self.out / f'{name}-smoke'
        admission = self.out / f'{name}-admission.json'
        self.command(module, *arguments, '--mode', 'smoke', '--output-dir', smoke)
        print('# Review the real smoke evidence and supply a review note before the full run.')
        self.command('runner', 'admit', '--run-dir', smoke, '--output', admission,
                     '--review-note', '')
        self.command(module, *arguments, '--mode', 'run', '--output-dir', run, '--admission', admission)

    def model_arguments(self):
        return ['--preparation-dir', self.prep, '--model-dir', self.model]

    def run(self, name, action, *args):
        self.stage(name, 'runner', [action, '--profile', 'full', *self.model_arguments(), *args])

    def weak_audit(self, e1):
        self.run('e1-weak-full', 'weak-audit', '--checkpoint-dir', e1)
        print('# Review the full E1 development audit and supply a note before weak-verifier training.')
        self.command('runner', 'admit-weak', '--run-dir', self.out / 'e1-weak-full',
                     '--output', self.out / 'weak-admission.json',
                     '--review-note', '')

    def audit_training(self, name, output, *args):
        self.command('runner', 'audit-training', '--run-dir', self.out / name,
                     '--preparation-dir', self.prep, '--output-dir', self.out / output, *args)

    def audit_test(self, name, split):
        self.command('runner', 'audit-answers', '--predictions', self.out / name / 'scored.jsonl',
                     '--preparation-dir', self.prep, '--profile', 'full', '--split', split,
                     '--generations-per-question', 4, '--output-dir', self.out / f'{name}-audit')

    def baselines(self, legacy=False):
        e1 = self.e1 if legacy else self.out / 'e1-train'
        schedule = (42, 43, 44) if legacy else (44,)
        interval = [] if legacy else ['--eval-steps', 100]
        stages = ('E2', 'E3') if legacy else ('E2', 'E3', 'E4')
        split = 'validation' if legacy else 'test'
        if legacy:
            print(f'# Run these commands at {LEGACY_CODE} with the original 0.5B preparation and E1.')
            print('# Keep archived seed-42 artifacts for exact historical comparison.')
        else:
            print('# Prepare the audited 1.5B inputs in a new directory using an existing Hub cache.')
            self.command('snapshots', '--sources', 'configs/sources-1.5b.json',
                         '--profile', 'full', '--output-dir', self.prep)
            self.run('e1-train', 'train', '--stage', 'E1', *interval)
        self.weak_audit(e1)
        if not legacy:
            self.run('e1-test', 'evaluate', '--split', split, '--sampled-test', '--checkpoint-dir', e1)
            self.audit_test('e1-test', split)
        for seed in schedule:
            for stage in stages:
                tag = f's{seed}-{stage.lower()}' if legacy else stage.lower()
                parent = ['--sft-run', e1]
                weak = ['--weak-admission', self.out / 'weak-admission.json'] if stage != 'E2' else []
                self.run(f'{tag}-train', 'train', '--stage', stage, '--training-seed', seed,
                         '--max-steps', 100, *interval, *parent, *weak)
                self.audit_training(f'{tag}-train', f'{tag}-training-audit')
                self.run(f'{tag}-{split}', 'evaluate', '--split', split, f'--sampled-{split}',
                         '--checkpoint-dir', self.out / f'{tag}-train', *parent)
                self.audit_test(f'{tag}-{split}', split)
        if not legacy:
            self.command('runner', 'replay-e4', '--training-audit', self.out / 'e3-training-audit',
                         '--output-dir', self.out / 'e3-replay-e4')
            for tag in ('e2', 'e4'):
                self.command('runner', 'attribute-answers', f'--{tag}', self.out / f'{tag}-test',
                             '--e3', self.out / 'e3-test', '--split', 'test',
                             '--preparation-dir', self.prep, '--output-dir', self.out / f'heldout-{tag}-e3')
            return
        print('# Run the summary from the current checkout. For the original comparison, use the archived seed-42 audit paths.')
        pairs = []
        for seed in schedule:
            pairs.extend(('--pair', seed,
                          self.out / f's{seed}-e2-training-audit/metrics.json',
                          self.out / f's{seed}-e2-validation-audit/metrics.json',
                          self.out / f's{seed}-e3-training-audit/metrics.json',
                          self.out / f's{seed}-e3-validation-audit/metrics.json'))
        self.command('study_analysis', 'legacy-pairs', *pairs,
                     '--output', self.out / 'legacy-comparison.json')

    def probes(self):
        for name in ('enriched', 'random'):
            self.run(f'{name}-panel', 'repair-probe', '--checkpoint-dir', self.e1,
                     '--panel', REPAIR_CONFIG_DIR / f'{name}-panel.json')
        self.command('runner', 'select-repair-cohort', '--probe', self.out / 'enriched-panel',
                     '--probe', self.out / 'random-panel', '--preparation-dir', self.prep,
                     '--output', self.out / 'cohort.json')
        print(f'# Compare selected train_ids with the frozen IDs in {REPAIR_CONFIG}.')

    def online(self):
        for seed in self.config['seeds']:
            for policy in self.config['policies']:
                tag = f's{seed}-{policy}'
                plan = self.out / f'{tag}-plan.json'
                source = []
                if policy == 'mixed':
                    source = ['--all-wrong-run', self.out / f's{seed}-all_wrong-train',
                              '--all-wrong-audit', self.out / f's{seed}-all_wrong-audit/metrics.json']
                self.command('reproduce', 'repair-plan', '--seed', seed, '--policy', policy,
                             '--output', plan, *source)
                self.run(f'{tag}-train', 'train', '--stage', 'E5', '--training-seed', seed,
                         '--max-steps', self.config['max_steps'], '--eval-steps', self.config['max_steps'],
                         '--repair-plan', plan, '--sft-run', self.e1,
                         '--weak-admission', self.baseline / 'weak-admission.json')
                self.audit_training(f'{tag}-train', f'{tag}-audit',
                                    '--window-steps', self.config['window_steps'])
        for seed in self.config['seeds']:
            for policy in self.config['policies']:
                tag = f's{seed}-{policy}'
                self.run(f'{tag}-test', 'evaluate', '--split', 'test', '--sampled-test',
                         '--checkpoint-dir', self.out / f'{tag}-train', '--sft-run', self.e1)
        print('# Check actual withdrawn_awards in each audit before interpreting a budget-matched comparison.')
        for left, right in (('all_wrong', 'mixed'), ('weak_control', 'tag_priority')):
            pairs = []
            for seed in self.config['seeds']:
                pairs.extend(('--left', self.out / f's{seed}-{left}-test',
                              '--right', self.out / f's{seed}-{right}-test'))
            self.command('runner', 'compare-repairs', '--preparation-dir', self.prep,
                         *pairs, '--output', self.out / f'contrast-{right}-{left}.json')

    def gradient(self):
        for seed in self.config['seeds']:
            self.stage(f'gradient-s{seed}', 'repair_gradient',
                       [*self.model_arguments(), '--checkpoint-dir', self.e1,
                        '--probe-dir', self.bank, '--seed', seed])

    def offline(self):
        self.command('study_analysis', 'panel-rewards', '--probe-dir', self.bank,
                     '--preparation-dir', self.prep, '--output', self.out / 'panel-reward-counts.json')
        review = REPAIR_CONFIG_DIR / 'archived-final-review.json'
        print(f'# Add --review-file {review} only for the original archived bank.')
        self.command('runner', 'repair-curve', '--probe', self.bank, '--preparation-dir', self.prep,
                     '--output', self.out / 'repair-curve.json')
        gradients = []
        for seed in self.config['seeds']:
            gradient = self.out / f'gradient-s{seed}'
            gradients.extend(('--run-dir', gradient))
            self.allocate(self.bank, f'allocation-gradient-s{seed}', '--gradient-dir', gradient)
            for policy in self.config['policies']:
                tag = f's{seed}-{policy}-test'
                self.allocate(self.out / tag, f'allocation-{tag}')
        self.command('study_analysis', 'gradient-directions', *gradients,
                     '--output', self.out / 'gradient-directions.json')

    def allocate(self, source, name, *args):
        self.command('repair_allocation', '--probe-dir', source, '--preparation-dir', self.prep,
                     '--output', self.out / f'{name}.json', *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preview = commands.add_parser('preview', help='print reviewable commands without running them')
    preview.add_argument('section', choices=('legacy', 'baselines', 'probes', 'online', 'gradient', 'offline'))
    preview.add_argument('--python', default='python3', help='Python interpreter in the pinned environment')
    preview.add_argument('--work-dir', required=True, help='existing parent for new experiment outputs')
    preview.add_argument('--baseline-dir', help='directory containing e1-train and weak-admission.json')
    preview.add_argument('--sft-run', help='existing E1 run, used outside baselines')
    preview.add_argument('--probe-dir', help='existing random-panel run for gradient and offline')
    preview.add_argument('--preparation-dir', required=True, help='prepared full GSM8K split')
    preview.add_argument('--model-dir', required=True, help='pinned local model snapshot')
    plan = commands.add_parser('repair-plan', help='write one frozen E5 policy and its withdrawal caps')
    plan.add_argument('--seed', type=int, required=True)
    plan.add_argument('--policy', choices=('all_wrong', 'mixed', 'weak_control', 'tag_priority'), required=True)
    plan.add_argument('--all-wrong-run')
    plan.add_argument('--all-wrong-audit', help='metrics.json from the verified all-wrong training audit')
    plan.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'repair-plan':
        value = repair_plan(ROOT / REPAIR_CONFIG, args.seed, args.policy, args.all_wrong_run, args.all_wrong_audit)
        write_json(args.output, value)
        return
    print('# From the repository root, enable local imports and keep Hub access offline.')
    print('export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1')
    commands = CommandPlan(args)
    if args.section == 'legacy':
        commands.baselines(legacy=True)
    else:
        getattr(commands, args.section)()


if __name__ == '__main__':
    main()
