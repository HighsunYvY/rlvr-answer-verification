"""Unified explicit stage CLI; never acquires resources or downloads inputs."""
import argparse
import json

from .analysis import analyze
from .experiment import approve
from .rollouts import approve_weak, run_rollouts
from .training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'weak-audit', 'train'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--preparation-dir', required=True)
        cmd.add_argument('--model-dir', required=True)
        cmd.add_argument('--output-dir', required=True)
        cmd.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='smoke')
        cmd.add_argument('--mode', choices=['smoke', 'run'], default='smoke')
        cmd.add_argument('--admission')
        cmd.add_argument('--sft-run')
        if name == 'train':
            cmd.add_argument('--stage', choices=['E1', 'E2', 'E3'], required=True)
            cmd.add_argument('--weak-admission')
            cmd.add_argument('--resume-from-checkpoint')
            cmd.add_argument('--max-steps', type=int,
                             help='GRPO step budget override; requires matching smoke admission')
        else:
            cmd.add_argument('--checkpoint-dir')
            if name == 'evaluate':
                cmd.add_argument('--split', choices=['test','validation'], default='test')
    for name in ('admit', 'admit-weak'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--run-dir', required=True)
        cmd.add_argument('--output', required=True)
        cmd.add_argument('--review-note', required=True)
    cmd = sub.add_parser('analyze')
    for stage in ('e0', 'e1', 'e2', 'e3'):
        cmd.add_argument('--'+stage, required=stage in ('e0','e1'))
        if stage != 'e0':
            cmd.add_argument('--'+stage+'-training-run')
    cmd.add_argument('--output-dir', required=True)
    cmd.add_argument('--preparation-dir', required=True)
    cmd.add_argument('--plot', action='store_true')
    cmd.add_argument('--weak-audit-run')
    args = parser.parse_args()
    if args.command == 'train':
        result = train(args.stage, args.preparation_dir, args.model_dir, args.output_dir,
                       args.profile, args.mode, args.sft_run, args.admission,
                       args.weak_admission, args.resume_from_checkpoint, max_steps=args.max_steps)
    elif args.command in ('evaluate','weak-audit'):
        result = run_rollouts(args.preparation_dir, args.model_dir, args.output_dir,
                    args.profile, args.mode, args.checkpoint_dir, args.sft_run,
                    'validation' if args.command == 'weak-audit' else args.split,
                    args.admission, args.command == 'weak-audit')
    elif args.command in ('admit', 'admit-weak'):
        operation = approve if args.command == 'admit' else approve_weak
        result = operation(args.run_dir, args.output, args.review_note)
    else:
        result = analyze({s: getattr(args,s.lower()) for s in ('E0','E1','E2','E3') if getattr(args,s.lower())},
                         args.output_dir, args.plot, args.preparation_dir,
                         {s: getattr(args,s.lower()+'_training_run') for s in ('E1','E2','E3')
                          if getattr(args,s.lower()+'_training_run')}, args.weak_audit_run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
