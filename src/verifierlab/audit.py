"""Reconcile E0 artifacts with frozen gold and a local tokenizer; no model execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .evaluate import aggregate_metrics, score_prediction
from .generate import ROOT, load_frozen
from .preflight import code_revision
from .snapshots import SMOKE_MANIFEST_SHA256, file_identity, verify_snapshot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def reconcile_rows(raw, scored, frozen, tokenizer, eos_ids, limit):
    """Recompute decoding and scores, keeping the frozen order and all truncated rows."""
    require(0 < len(raw) == len(scored) == len(frozen), 'raw/scored/frozen count mismatch')
    require(len({r['sample_id'] for r in raw}) == len(raw), 'duplicate raw sample IDs')
    for original, observed, gold in zip(raw, scored, frozen, strict=True):
        require(original['sample_id'] == observed['sample_id'] == gold['sample_id'],
                'sample ID/order mismatch')
        require(set(original) == {'sample_id', 'prompt_tokens', 'generated_token_ids'},
                'unexpected raw fields')
        prompt = tokenizer.apply_chat_template(gold['prompt'], tokenize=True,
                                               add_generation_prompt=True)
        require(type(original['prompt_tokens']) is int and original['prompt_tokens'] == len(prompt),
                'prompt token count mismatch')
        ids = original['generated_token_ids']
        require(isinstance(ids, list) and 0 < len(ids) <= limit and all(
            type(i) is int and 0 <= i < len(tokenizer) for i in ids), 'invalid new token IDs')
        eos_at = next((i for i, token in enumerate(ids) if token in eos_ids), None)
        require(eos_at is not None or len(ids) == limit, 'unexplained early termination')
        kept = ids if eos_at is None else ids[:eos_at + 1]
        require(eos_at is None or all(i == tokenizer.pad_token_id for i in ids[eos_at + 1:]),
                'non-padding tokens after EOS')
        text = tokenizer.decode(kept, skip_special_tokens=True)
        expected = score_prediction(gold['sample_id'], text, gold['ground_truth'], len(kept))
        expected.update(raw_completion=tokenizer.decode(kept, skip_special_tokens=False),
                        generated_token_ids=ids, prompt_tokens=len(prompt),
                        eos_token_id=None if eos_at is None else ids[eos_at],
                        termination='max_new_tokens' if eos_at is None else 'eos',
                        truncated=eos_at is None)
        require(set(observed) == set(expected) | {'generation_seconds'}, 'scored field mismatch')
        require(json.dumps({k: observed[k] for k in expected}, sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False),
                'decoded text, gold, score or termination mismatch')
        require(positive(observed['generation_seconds']), 'invalid sample duration')
    metrics = aggregate_metrics(scored)
    metrics['truncation_rate'] = sum(r['truncated'] for r in scored) / len(scored)
    return metrics


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit(run_dir, preparation_dir, model_dir, expected_code):
    require(bool(re.fullmatch('[0-9a-f]{40}', expected_code)), 'full expected execution SHA required')
    run_dir, preparation_dir, model_dir = map(Path, (run_dir, preparation_dir, model_dir))
    audit_code = code_revision(ROOT)
    require(not audit_code['dirty'], 'audit requires a clean committed checkout')
    report = json.loads((run_dir / 'run.json').read_text())
    require(report['code'] == {'commit': expected_code, 'dirty': False}, 'execution SHA/dirty mismatch')
    require(report['schema_version'] == 1 and report['stage'] == 'E0'
            and report['profile'] == 'smoke' and report['status'] == 'passed'
            and report['model_generation_validated'] is True and report['trainer_validated'] is False
            and report['n'] == 4, 'not a completed E0 smoke')
    required = {'raw.jsonl', 'scored.jsonl', 'metrics.json', 'nvidia-smi.txt'}
    require(set(report['artifacts']) == required, 'missing/unexpected artifact inventory')
    for name, digest in report['artifacts'].items():
        require(file_identity(run_dir / name)['sha256'] == digest, f'artifact hash mismatch: {name}')
    require(report['manifest_sha256'] == SMOKE_MANIFEST_SHA256, 'manifest anchor mismatch')
    frozen = load_frozen(preparation_dir / 'data', SMOKE_MANIFEST_SHA256)
    preparation_path = preparation_dir / 'preparation.json'
    require(file_identity(preparation_path)['sha256'] == report['preparation_sha256'],
            'preparation hash mismatch')
    sources = json.loads((ROOT / 'configs/sources.json').read_text())
    preparation = json.loads(preparation_path.read_text())
    require(report['sources'] == sources == preparation['sources']
            and preparation['status'] == 'passed'
            and preparation['split_manifest_sha256'] == report['manifest_sha256'],
            'source/preparation identity mismatch')
    require(verify_snapshot(model_dir, sources['model']) == report['model_files'],
            'local model identity mismatch')
    config_path = ROOT / 'configs/evaluation.yaml'
    config = json.loads(config_path.read_text())
    require(report['config'] == config and report['config_sha256'] == file_identity(config_path)['sha256']
            and report['lock_sha256'] == file_identity(ROOT / 'requirements/linux-py311-cu128.txt')['sha256'],
            'config/lock identity mismatch')
    require(report['environment']['passed'] is True and not report['environment']['errors'],
            'environment did not pass')
    require(positive(report['elapsed_seconds']), 'invalid elapsed time')
    for name in ('total_bytes', 'free_bytes_before_load', 'peak_allocated_bytes', 'peak_reserved_bytes'):
        require(type(report['gpu'][name]) is int and report['gpu'][name] > 0, 'missing GPU memory evidence')
    gpu = report['gpu']
    require(gpu['peak_allocated_bytes'] <= gpu['peak_reserved_bytes'] <= gpu['total_bytes']
            and gpu['free_bytes_before_load'] <= gpu['total_bytes'], 'inconsistent GPU memory envelope')
    require((run_dir / 'nvidia-smi.txt').stat().st_size > 0, 'empty driver/device evidence')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                              trust_remote_code=False, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = json.loads((model_dir / 'generation_config.json').read_text())['eos_token_id']
    eos_ids = [eos] if isinstance(eos, int) else eos
    require(tokenizer.eos_token_id in eos_ids, 'tokenizer EOS mismatch')
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    expected_tokenizer = {'chat_template_sha256': template_hash,
                          'pad_token_id': tokenizer.pad_token_id, 'eos_token_ids': eos_ids,
                          'padding_side': 'left'}
    require(report['tokenizer'] == expected_tokenizer, 'template/pad/EOS identity mismatch')
    expected_generation = {**config['generation'], 'num_beams': 1, 'use_cache': True,
                           'eos_token_id': eos_ids, 'pad_token_id': tokenizer.pad_token_id}
    require(all(report['resolved_generation'][k] == v for k, v in expected_generation.items()),
            'resolved decoding mismatch')
    scored = read_rows(run_dir / 'scored.jsonl')
    metrics = reconcile_rows(read_rows(run_dir / 'raw.jsonl'), scored, frozen, tokenizer,
                             eos_ids, config['generation']['max_new_tokens'])
    recorded = json.loads((run_dir / 'metrics.json').read_text())
    require(recorded == report['metrics'], 'metrics file/report mismatch')
    require(set(recorded) == set(metrics) | {'generation_seconds', 'tokens_per_second'},
            'metrics field mismatch')
    require(all(recorded[k] == v for k, v in metrics.items()), 'recomputed metrics mismatch')
    seconds = recorded['generation_seconds']
    require(positive(seconds) and seconds >= sum(r['generation_seconds'] for r in scored),
            'invalid aggregate timing')
    require(report['elapsed_seconds'] >= seconds, 'run duration shorter than generation')
    speed = recorded['tokens_per_second']
    require(positive(speed) and math.isclose(speed, sum(r['generated_tokens'] for r in scored) / seconds,
                                           rel_tol=1e-9), 'throughput mismatch')
    return {'status': 'passed', 'check': 'E0_artifact_consistency', 'execution_commit': expected_code,
            'audit_code': audit_code,
            'run_report_sha256': file_identity(run_dir / 'run.json')['sha256'], 'n': len(scored),
            'metrics': recorded, 'gpu_admission': 'pending_manual_review', 'trainer_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--preparation-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--expected-code', required=True, help='full execution SHA, not audit checkout SHA')
    parser.add_argument('--output', required=True, help='new audit JSON outside the run directory')
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve().is_relative_to(Path(args.run_dir).resolve()):
        parser.error('--output must be outside the immutable run directory')
    # Reserve the output before auditing so retries cannot destroy previous evidence.
    with output.open('x', encoding='utf-8') as stream:
        try:
            result = audit(args.run_dir, args.preparation_dir, args.model_dir, args.expected_code)
        except BaseException as exc:
            json.dump({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       'error': f'{type(exc).__name__}: {exc}',
                       'gpu_admission': 'pending_manual_review'}, stream, indent=2)
            stream.write('\n')
            raise
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
