"""Offline, single-GPU E0 smoke generation scored against hash-frozen gold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time

from .data import INSTRUCTION, question_key, require_audited_test
from .evaluate import aggregate_metrics, score_prediction
from .preflight import check_environment, code_revision
from .snapshots import SMOKE_MANIFEST_SHA256, file_identity, validate_sources, verify_snapshot, write_json
from .verifier import normalize_number

ROOT = Path(__file__).resolve().parents[2]


def load_frozen(directory, expected_manifest, split="test"):
    directory = Path(directory)
    if file_identity(directory / 'manifest.json')['sha256'] != expected_manifest:
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected_count = {'test': 4, 'sft': 8}.get(split)
    if expected_count is None or manifest['profile'] != 'smoke' or manifest['sizes'][split] != expected_count:
        raise ValueError('only frozen test/SFT smoke splits are admitted')
    path = directory / f'{split}.jsonl'
    if file_identity(path)['sha256'] != manifest['files'][f'{split}.jsonl']:
        raise ValueError('frozen split hash mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    identities = [{'sample_id': r['sample_id'], 'question_hash': r['question_hash']} for r in rows]
    if (identities != manifest['samples'][split] or len(rows) != expected_count
            or len({r['sample_id'] for r in rows}) != len(rows)):
        raise ValueError('frozen sample identity/count mismatch')
    for row in rows:
        prompt = row['prompt']
        if (len(prompt) != 2 or prompt[0] != {'role': 'system', 'content': INSTRUCTION}
                or set(prompt[1]) != {'role', 'content'} or prompt[1]['role'] != 'user'
                or question_key(prompt[1]['content']) != row['question_hash']
                or normalize_number(row['ground_truth']) is None):
            raise ValueError('invalid frozen prompt/gold')
    return rows


def completion_record(sample_id, ids, tokenizer, eos_ids, limit, prompt_tokens):
    """Input is newly generated IDs only. EOS counts as one token; trailing pads do not."""
    if not ids or len(ids) > limit or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('empty or over-budget generation')
    stop = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    if stop is None and len(ids) != limit:
        raise ValueError('generation ended without EOS or token budget exhaustion')
    if stop is not None and any(i != tokenizer.pad_token_id for i in ids[stop + 1:]):
        raise ValueError('non-padding tokens after EOS')
    counted = ids if stop is None else ids[:stop + 1]
    return {'sample_id': sample_id, 'completion': tokenizer.decode(counted, skip_special_tokens=True),
            'raw_completion': tokenizer.decode(counted, skip_special_tokens=False),
            'generated_token_ids': ids, 'generated_tokens': len(counted),
            'prompt_tokens': prompt_tokens, 'eos_token_id': None if stop is None else ids[stop],
            'termination': 'max_new_tokens' if stop is None else 'eos',
            'truncated': stop is None}


def score_frozen(prediction, frozen):
    if prediction['sample_id'] != frozen['sample_id'] or 'ground_truth' in prediction:
        raise ValueError('prediction ID/gold boundary violation')
    return {**prediction, **score_prediction(prediction['sample_id'], prediction['completion'],
                                            frozen['ground_truth'], prediction['generated_tokens'])}


def run(preparation_dir, model_dir, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema_version': 1, 'stage': 'E0', 'profile': 'smoke', 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'model_generation_validated': False, 'trainer_validated': False}
    torch = None
    try:
        report['code'] = code_revision(ROOT)
        if report['code']['dirty']:
            raise ValueError('run requires a clean committed Git checkout')
        sources = json.loads((ROOT / 'configs/sources.json').read_text())
        validate_sources(sources)
        preparation_dir = Path(preparation_dir)
        preparation = json.loads((preparation_dir / 'preparation.json').read_text())
        expected = SMOKE_MANIFEST_SHA256
        if (preparation['status'] != 'passed' or preparation['sources'] != sources
                or preparation['split_manifest_sha256'] != expected):
            raise ValueError('preparation identity/status mismatch')
        rows = load_frozen(preparation_dir / 'data', expected)
        require_audited_test(json.loads((preparation_dir / 'data/manifest.json').read_text()))
        model_dir = Path(model_dir)
        allowed = set(sources['model']['files']) | {'.gitattributes'}
        extras = {str(p.relative_to(model_dir)) for p in model_dir.rglob('*')
                  if p.is_file()} - allowed
        if extras:
            raise ValueError(f'unverified extra model files: {sorted(extras)}')
        files = verify_snapshot(model_dir, sources['model'])
        config_path = ROOT / 'configs/evaluation.yaml'
        config = json.loads(config_path.read_text())
        generation = config['generation']
        if (generation != {'do_sample': False, 'max_new_tokens': 512, 'num_return_sequences': 1}
                or config['split'] != 'test' or config['seed'] != 42):
            raise ValueError('E0 smoke decoding contract mismatch')
        environment = check_environment(ROOT / 'pyproject.toml')
        report.update(config=config, config_sha256=file_identity(config_path)['sha256'],
                      lock_sha256=file_identity(ROOT / 'requirements/linux-py311-cu128.txt')['sha256'],
                      manifest_sha256=expected, sources=sources, model_files=files,
                      preparation_sha256=file_identity(preparation_dir / 'preparation.json')['sha256'],
                      environment=environment)
        if not environment['passed']:
            raise RuntimeError('target environment mismatch; inspect run.json')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
        torch = torch_module
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 CUDA required; no CPU/MPS fallback')
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        report['gpu'] = {'name': torch.cuda.get_device_name(), 'runtime': torch.version.cuda,
                         'capability': torch.cuda.get_device_capability(),
                         'free_bytes_before_load': free, 'total_bytes': total}
        report['gpu']['torch'] = torch.__version__
        with (out / 'nvidia-smi.txt').open('x') as stream:
            subprocess.run(['nvidia-smi'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                  trust_remote_code=False, padding_side='left')
        if not tokenizer.chat_template or tokenizer.eos_token_id is None:
            raise ValueError('checkpoint chat template/EOS required')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        if not eos_ids or tokenizer.eos_token_id not in eos_ids:
            raise ValueError('model/tokenizer EOS mismatch')
        decode_config = GenerationConfig(**generation, num_beams=1, use_cache=True,
                                         eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
        report['tokenizer'] = {'chat_template_sha256': hashlib.sha256(
            tokenizer.chat_template.encode()).hexdigest(), 'pad_token_id': tokenizer.pad_token_id,
            'eos_token_ids': eos_ids, 'padding_side': tokenizer.padding_side}
        report['resolved_generation'] = decode_config.to_dict()
        scored = []
        generation_start = time.monotonic()
        with (out / 'raw.jsonl').open('x') as raw, (out / 'scored.jsonl').open('x') as stream:
            for frozen in rows:
                # Gold and demonstration never enter apply_chat_template or generate.
                inputs = tokenizer.apply_chat_template(
                    frozen['prompt'], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors='pt').to('cuda')
                width = inputs['input_ids'].shape[1]
                if width + generation['max_new_tokens'] > model.config.max_position_embeddings:
                    raise ValueError('context overflow; prompt truncation is forbidden')
                torch.cuda.synchronize()
                sample_start = time.monotonic()
                with torch.inference_mode():
                    sequence = model.generate(**inputs, generation_config=decode_config)
                torch.cuda.synchronize()
                sample_elapsed = time.monotonic() - sample_start
                ids = sequence[0, width:].tolist()
                raw.write(json.dumps({'sample_id': frozen['sample_id'], 'prompt_tokens': width,
                                      'generated_token_ids': ids}) + '\n')
                raw.flush()
                if sequence.shape[0] != 1 or not torch.equal(
                        sequence[0, :width], inputs['input_ids'][0]):
                    raise ValueError('generation prefix/return count mismatch')
                prediction = completion_record(frozen['sample_id'], ids, tokenizer, eos_ids,
                                               generation['max_new_tokens'], width)
                prediction['generation_seconds'] = sample_elapsed
                row = score_frozen(prediction, frozen)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                scored.append(row)
        seconds = time.monotonic() - generation_start
        metrics = aggregate_metrics(scored)
        metrics.update(truncation_rate=sum(r['truncated'] for r in scored) / len(scored),
                       generation_seconds=seconds,
                       tokens_per_second=sum(r['generated_tokens'] for r in scored) / seconds)
        write_json(out / 'metrics.json', metrics)
        report.update(status='passed', model_generation_validated=True, n=len(scored), metrics=metrics)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if torch is not None and 'gpu' in report:
            report['gpu'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['elapsed_seconds'] = time.monotonic() - started
        report['artifacts'] = {p.name: file_identity(p)['sha256'] for p in out.iterdir() if p.is_file()}
        write_json(out / 'run.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-dir', required=True, help='passed preparation root with data/')
    parser.add_argument('--model-dir', required=True, help='local verified Qwen snapshot directory')
    parser.add_argument('--output-dir', required=True, help='new directory; partial failures retained')
    args = parser.parse_args()
    print(json.dumps(run(args.preparation_dir, args.model_dir, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
