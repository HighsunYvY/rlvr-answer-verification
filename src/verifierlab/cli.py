"""Small command-line orchestration layer, no implicit downloads."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from .data import make_splits
from .evaluate import aggregate_metrics, score_prediction
from .utils import diagnostics, load_config, validate_grpo


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Split LOCAL GSM8K JSONL; no network access")
    parser.add_argument('--train-jsonl', required=True)
    parser.add_argument('--test-jsonl', required=True)
    parser.add_argument('--config', default='configs/data.yaml')
    parser.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='smoke')
    parser.add_argument('--source-revision', required=True, help='dataset commit SHA, or synthetic-local')
    parser.add_argument('--output-dir', required=True, help='new directory; refuses overwrite')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    splits, manifest = make_splits(read_jsonl(args.train_jsonl), read_jsonl(args.test_jsonl),
                                    config['profiles'][args.profile], config['seed'])
    manifest.update(data_config=config, profile=args.profile, source_revision=args.source_revision,
                    input_sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for key, path in [('train', args.train_jsonl), ('test', args.test_jsonl)]})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for key, rows in splits.items():
        with (out / f'{key}.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'output_dir': str(out), 'sizes': manifest['sizes']}))


def preview_main(argv=None):
    parser = argparse.ArgumentParser(description="Preview experiment configuration; no execution")
    parser.add_argument('--config', required=True)
    parser.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='smoke')
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = {'mode': 'plan_only', 'execution_available': False, 'config': config}
    if 'stage' in config:
        stage = config['stage']
        config['trainer'].update(config['profiles'][args.profile])
        config['trainer']['output_dir'] = f'outputs/{stage}/{args.profile}/RUN_ID'
        if stage.startswith('grpo'):
            result['batch_check'] = validate_grpo(config, args.world_size)
    print(json.dumps(result, indent=2))


def summarize_main(argv=None):
    parser = argparse.ArgumentParser(description="Score LOCAL prediction JSONL; no model loading")
    parser.add_argument('--predictions', required=True, help='sample_id, completion, ground_truth, optional generated_tokens')
    parser.add_argument('--output-dir', required=True, help='new run directory')
    args = parser.parse_args(argv)
    rows = [score_prediction(**r) for r in read_jsonl(args.predictions)]
    summary = aggregate_metrics(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'scored.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'metrics.json', summary)
    print(json.dumps(summary, indent=2))


def diagnose_main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect local resource availability')
    parser.add_argument('--probe-gpu', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(diagnostics(args.probe_gpu), indent=2))


def main():
    commands = {
        'prepare-data': prepare_main,
        'preview-config': preview_main,
        'summarize-results': summarize_main,
        'diagnose': diagnose_main,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=commands)
    args = parser.parse_args(sys.argv[1:2])
    commands[args.command](sys.argv[2:])


if __name__ == '__main__':
    main()
