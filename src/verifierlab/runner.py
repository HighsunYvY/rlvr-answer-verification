"""Unified explicit stage CLI; never acquires resources or downloads inputs."""
import argparse
import json

from .analysis import analyze, audit_answers
from .attribution import attribute_answers
from .experiment import approve, read_json
from .rollouts import approve_weak, run_rollouts
from .repair_replay import replay_repairs
from .repair_study import compare_repair_evaluations, select_cohort
from .shuffle_audit import replay_e4
from .training import train
from .training_audit import audit_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'weak-audit', 'repair-probe', 'train'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--preparation-dir', required=True)
        cmd.add_argument('--model-dir', required=True)
        cmd.add_argument('--output-dir', required=True)
        cmd.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='smoke')
        cmd.add_argument('--mode', choices=['smoke', 'run'], default='smoke')
        cmd.add_argument('--admission')
        cmd.add_argument('--sft-run')
        if name == 'train':
            cmd.add_argument('--stage', choices=['E1', 'E2', 'E3', 'E4', 'E5'], required=True)
            cmd.add_argument('--repair-plan', help='E5 policy, per-window caps and frozen training/validation IDs')
            cmd.add_argument('--weak-admission')
            cmd.add_argument('--resume-from-checkpoint')
            cmd.add_argument('--max-steps', type=int,
                             help='GRPO step budget override; requires matching smoke admission')
            cmd.add_argument('--training-seed', type=int,
                             help='set GRPO model and data-loader seeds for matched runs')
            cmd.add_argument('--eval-steps', type=int,
                             help='optimizer steps between full development evaluations')
        else:
            cmd.add_argument('--checkpoint-dir')
            if name == 'repair-probe':
                cmd.add_argument('--panel', required=True, help='training question IDs, draw count and seed')
            if name == 'evaluate':
                cmd.add_argument('--split', choices=['test','validation'], default='test')
                sampling = cmd.add_mutually_exclusive_group()
                sampling.add_argument('--sampled-validation', action='store_true',
                                      help='sample groups on the frozen validation split')
                sampling.add_argument('--sampled-test', action='store_true',
                                      help='sample groups on the frozen test split')
    for name in ('admit', 'admit-weak'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--run-dir', required=True)
        cmd.add_argument('--output', required=True)
        cmd.add_argument('--review-note', required=True)
    cmd = sub.add_parser('analyze')
    for stage in ('e0', 'e1', 'e2', 'e3'):
        cmd.add_argument('--'+stage, required=stage == 'e1')
        if stage != 'e0':
            cmd.add_argument('--'+stage+'-training-run')
    cmd.add_argument('--output-dir', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--plot', action='store_true')
    cmd.add_argument('--weak-audit-run')
    cmd.add_argument('--split', choices=['test', 'validation'], default='test')
    cmd = sub.add_parser('audit-answers', help='diagnose a complete scored split without loading a model')
    cmd.add_argument('--predictions', required=True, help='existing scored JSONL, including truncation flags')
    cmd.add_argument('--preparation-dir', required=True, help='preparation directory for the same frozen split')
    cmd.add_argument('--profile', choices=['smoke', 'pilot', 'full'], required=True)
    cmd.add_argument('--split', choices=['validation', 'test'], required=True)
    cmd.add_argument('--generations-per-question', type=int, default=1)
    cmd.add_argument('--replay-grpo', action='store_true', help='compare group-scaled strict and weak advantages')
    cmd.add_argument('--output-dir', required=True)
    cmd = sub.add_parser('audit-training', help='replay the reward signal from an archived GRPO training run')
    cmd.add_argument('--run-dir', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--output-dir', required=True)
    cmd.add_argument('--window-steps', type=int, default=25)
    cmd = sub.add_parser('attribute-answers', help='compare weak-rewarded explicit wrong answers by frozen question')
    comparison = cmd.add_mutually_exclusive_group(required=True)
    comparison.add_argument('--e2')
    comparison.add_argument('--e4')
    cmd.add_argument('--e3', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--split', choices=['validation', 'test'], required=True)
    cmd.add_argument('--output-dir', required=True)
    cmd = sub.add_parser('replay-e4', help='compute exact E4 expectations on fixed rollout groups')
    inputs = cmd.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--diagnosed')
    inputs.add_argument('--training-audit')
    cmd.add_argument('--group-size', type=int, default=4)
    cmd.add_argument('--output-dir', required=True)
    cmd = sub.add_parser('replay-repairs', help='withdraw one wrong correctness award at a time on fixed groups')
    cmd.add_argument('--run-dir', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--output-dir', required=True)
    cmd.add_argument('--exclude', action='append', default=[], help='ambiguous source question ID; repeatable')
    cmd = sub.add_parser('select-repair-cohort')
    cmd.add_argument('--probe', action='append', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--output', required=True)
    cmd = sub.add_parser('compare-repairs')
    cmd.add_argument('--left', action='append', required=True)
    cmd.add_argument('--right', action='append', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        result = train(args.stage, args.preparation_dir, args.model_dir, args.output_dir,
                       args.profile, args.mode, args.sft_run, args.admission,
                       args.weak_admission, args.resume_from_checkpoint, max_steps=args.max_steps,
                       training_seed=args.training_seed, eval_steps=args.eval_steps,
                       repair_plan=read_json(args.repair_plan) if args.repair_plan else None)
    elif args.command in ('evaluate','weak-audit','repair-probe'):
        split = {'weak-audit': 'validation', 'repair-probe': 'grpo'}.get(args.command, getattr(args, 'split', None))
        result = run_rollouts(args.preparation_dir, args.model_dir, args.output_dir,
                    args.profile, args.mode, args.checkpoint_dir, args.sft_run,
                    split,
                    args.admission, weak_audit=args.command == 'weak-audit',
                    sampled_validation=getattr(args, 'sampled_validation', False),
                    sampled_test=getattr(args, 'sampled_test', False),
                    repair_panel=read_json(args.panel) if args.command == 'repair-probe' else None)
    elif args.command in ('admit', 'admit-weak'):
        operation = approve if args.command == 'admit' else approve_weak
        result = operation(args.run_dir, args.output, args.review_note)
    elif args.command == 'audit-answers':
        result = audit_answers(args.predictions, args.preparation_dir, args.profile, args.split,
                               args.generations_per_question, args.output_dir, args.replay_grpo)
    elif args.command == 'audit-training':
        result = audit_training(args.run_dir, args.preparation_dir, args.output_dir, args.window_steps)
    elif args.command == 'attribute-answers':
        result = attribute_answers(args.e2 or args.e4, args.e3, args.preparation_dir, args.split,
                                   args.output_dir, left_stage='E2' if args.e2 else 'E4')
    elif args.command == 'replay-e4':
        result = replay_e4(args.diagnosed or args.training_audit, args.output_dir,
                           args.group_size, training_audit=bool(args.training_audit))
    elif args.command == 'replay-repairs':
        result = replay_repairs(args.run_dir, args.preparation_dir, args.output_dir, args.exclude)
    elif args.command == 'select-repair-cohort':
        result = select_cohort(args.probe, args.preparation_dir, args.output)
    elif args.command == 'compare-repairs':
        result = compare_repair_evaluations(args.left, args.right, args.preparation_dir, args.output)
    else:
        result = analyze({s: getattr(args,s.lower()) for s in ('E0','E1','E2','E3') if getattr(args,s.lower())},
                         args.output_dir, args.plot, args.preparation_dir,
                         {s: getattr(args,s.lower()+'_training_run') for s in ('E1','E2','E3')
                          if getattr(args,s.lower()+'_training_run')}, args.weak_audit_run, split=args.split)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
