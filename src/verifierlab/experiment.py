"""Shared immutable inputs, profile admission, checkpoint identity and runtime evidence."""
from contextlib import contextmanager
import hashlib
import json
import math
import re
import os
from pathlib import Path
import time

from .data import INSTRUCTION, question_key
from .generate import ROOT
from .preflight import check_environment, code_revision
from .snapshots import SMOKE_MANIFEST_SHA256, file_identity, verify_snapshot, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inventory(directory):
    return {str(p.relative_to(directory)): file_identity(p)['sha256']
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def verify_inventory(directory, files):
    for name, digest in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or file_identity(Path(directory) / path)['sha256'] != digest:
            raise ValueError(f'artifact identity mismatch: {name}')


def read_json(path):
    return json.loads(Path(path).read_text())


def prepared_inputs(directory, profile):
    directory = Path(directory)
    report = read_json(directory / 'preparation.json')
    manifest = read_json(directory / 'data/manifest.json')
    sources = read_json(ROOT / 'configs/sources.json')
    config = read_json(ROOT / 'configs/data.yaml')
    if (config['seed'] != 42 or report['status'] != 'passed' or report['sources'] != sources
            or report['split_manifest_sha256'] != file_identity(directory / 'data/manifest.json')['sha256']
            or manifest['profile'] != profile or manifest['seed'] != config['seed']
            or manifest['sizes'] != config['profiles'][profile] or manifest['source'] != sources['dataset']):
        raise ValueError('preparation/profile/source mismatch')
    if profile == 'smoke' and report['split_manifest_sha256'] != SMOKE_MANIFEST_SHA256:
        raise ValueError('frozen smoke manifest mismatch')
    splits, all_ids, all_questions = {}, set(), set()
    for name, count in manifest['sizes'].items():
        path = directory / 'data' / f'{name}.jsonl'
        if file_identity(path)['sha256'] != manifest['files'][path.name]:
            raise ValueError('frozen split bytes mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        identity = [{'sample_id': r['sample_id'], 'question_hash': r['question_hash']} for r in rows]
        if len(rows) != count or identity != manifest['samples'][name]:
            raise ValueError('sample identity/count mismatch')
        for row in rows:
            prompt = row['prompt']
            if (len(prompt) != 2 or prompt[0] != {'role': 'system', 'content': INSTRUCTION}
                    or prompt[1]['role'] != 'user' or set(prompt[1]) != {'role', 'content'}
                    or question_key(prompt[1]['content']) != row['question_hash']
                    or row['sample_id'] in all_ids or row['question_hash'] in all_questions):
                raise ValueError('prompt/identity/leakage boundary violation')
            all_ids.add(row['sample_id'])
            all_questions.add(row['question_hash'])
        splits[name] = rows
    return splits, manifest, report


def checkpoint(directory, require_passed=True):
    directory = Path(directory)
    meta = read_json(directory / 'checkpoint.json')
    if require_passed:
        run = read_json(directory / 'run.json')
        verify_inventory(directory, run['artifacts'])
        if run['status'] != 'passed' or not run['trainer_validated'] or run['stage'] != meta['stage']:
            raise ValueError('checkpoint is not backed by a completed real Trainer run')
    if meta['sources'] != read_json(ROOT / 'configs/sources.json'):
        raise ValueError('checkpoint source revision mismatch')
    verify_inventory(directory / 'adapter', meta['files'])
    if inventory(directory / 'adapter') != meta['files'] or fingerprint(meta['files']) != meta['adapter_sha256']:
        raise ValueError('checkpoint inventory mismatch')
    return meta


def save_checkpoint(model, tokenizer, out, stage, sft_identity=None, contract=None):
    adapter = Path(out) / 'adapter'
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    files = inventory(adapter)
    meta = {'schema_version': 1, 'stage': stage, 'files': files,
            'adapter_sha256': fingerprint(files), 'sft_identity': sft_identity,
            'training_contract': contract, 'sources': read_json(ROOT / 'configs/sources.json')}
    write_json(Path(out) / 'checkpoint.json', meta)
    return meta


def load_policy(model_dir, checkpoint_dir=None, sft_dir=None, trainable=False, merge_e1=True, allow_unvalidated=False):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import torch
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa')
    if checkpoint_dir:
        meta = checkpoint(checkpoint_dir, require_passed=not allow_unvalidated)
        if meta['stage'] in ('E2', 'E3', 'E4'):
            if not sft_dir or checkpoint(sft_dir)['adapter_sha256'] != meta['sft_identity']:
                raise ValueError('RL checkpoint requires its exact original E1 checkpoint')
            if checkpoint(sft_dir)['stage'] != 'E1':
                raise ValueError('reference must be E1')
            model = PeftModel.from_pretrained(model, str(Path(sft_dir) / 'adapter'),
                                               local_files_only=True).merge_and_unload()
        elif meta['stage'] != 'E1':
            raise ValueError('unknown checkpoint stage')
        model = PeftModel.from_pretrained(model, str(Path(checkpoint_dir) / 'adapter'),
                                          local_files_only=True, is_trainable=trainable)
        if meta['stage'] == 'E1' and merge_e1:
            model = model.merge_and_unload()
    return model


class Session:
    def __init__(self, out, stage, profile, mode, model_dir, preparation_dir):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.report = {'schema_version': 2, 'stage': stage, 'profile': profile, 'mode': mode,
                       'status': 'running', 'model_generation_validated': False,
                       'trainer_validated': False, 'pilot_full_admitted': False}
        self.model_dir, self.preparation_dir = Path(model_dir), Path(preparation_dir)

    def initialize(self, settings, policy_identity=None):
        code = code_revision(ROOT)
        if code['dirty'] or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('clean committed single-process execution required')
        self.splits, self.manifest, prep = prepared_inputs(self.preparation_dir, self.report['profile'])
        if policy_identity and policy_identity['training_contract']['manifest'] != prep['split_manifest_sha256']:
            raise ValueError('checkpoint and evaluation/training preparation differ')
        sources = read_json(ROOT / 'configs/sources.json')
        extra = set(inventory(self.model_dir)) - (set(sources['model']['files']) | {'.gitattributes'})
        if extra:
            raise ValueError('extra unverified model files')
        model_files = verify_snapshot(self.model_dir, sources['model'])
        env = check_environment(ROOT / 'pyproject.toml')
        self.report.update(code=code, environment=env, settings=settings, sources=sources,
                           model_files=model_files, manifest_sha256=prep['split_manifest_sha256'],
                           preparation_sha256=file_identity(self.preparation_dir / 'preparation.json')['sha256'],
                           policy_identity=policy_identity)
        if not env['passed']:
            raise RuntimeError('target environment mismatch')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import set_seed, AutoTokenizer
        self.torch = torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 CUDA required')
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        gpu = {'name': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
               'total_bytes': total, 'free_bytes_before_load': free, 'runtime': torch.version.cuda}
        self.report['gpu'] = gpu
        import subprocess
        gpu['driver_versions'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()
        with (self.out / 'nvidia-smi.txt').open('x') as stream:
            subprocess.run(['nvidia-smi'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        set_seed(settings['seed'])
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                                                      trust_remote_code=False, padding_side='left')
        if not self.tokenizer.chat_template or self.tokenizer.eos_token_id is None:
            raise ValueError('checkpoint template and EOS required')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tree = {str(p.relative_to(ROOT)): file_identity(p)['sha256']
                for base in ('src', 'configs', 'requirements') for p in (ROOT / base).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts
                and p.suffix in {'.py', '.json', '.yaml', '.txt', '.in'}}
        tree['pyproject.toml'] = file_identity(ROOT / 'pyproject.toml')['sha256']
        identity = {'stage': self.report['stage'], 'profile': self.report['profile'],
                    'settings': settings, 'policy': policy_identity, 'source_tree': fingerprint(tree),
                    'manifest': prep['split_manifest_sha256'], 'model_files': model_files,
                    'packages': env['packages'], 'python': env['python'],
                    'gpu': {k: v for k, v in gpu.items() if k != 'free_bytes_before_load'}}
        self.report['signature'] = fingerprint(identity)
        self.report['identity'] = identity
        self.report['template_sha256'] = hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()

    def admit(self, approval):
        if self.report['mode'] == 'run' and self.report['profile'] != 'smoke':
            if not approval:
                raise ValueError('matching reviewed real GPU smoke admission required')
            proof = read_json(approval)
            smoke_path = Path(proof['run_dir'])
            smoke = read_json(smoke_path / 'run.json')
            validate_smoke(smoke_path, smoke)
            if (proof.get('approved') is not True or not proof.get('review_note')
                    or proof['run_report_sha256'] != file_identity(smoke_path / 'run.json')['sha256']
                    or smoke['mode'] != 'smoke' or smoke['status'] != 'passed'
                    or smoke['signature'] != self.report['signature']
                    or not (smoke.get('trainer_validated') or smoke.get('model_generation_validated'))):
                raise ValueError('stale/mismatched smoke admission')
            self.report['admission_sha256'] = file_identity(approval)['sha256']
            self.report['pilot_full_admitted'] = True

    def finish(self, error=None):
        secondary = None
        if error:
            self.report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                               error=f'{type(error).__name__}: {error}')
        elif self.report['status'] != 'passed':
            secondary = RuntimeError('stage did not complete its evidence checks')
        try:
            if hasattr(self, 'torch') and 'gpu' in self.report:
                self.report['gpu'].update(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                                          peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        except Exception as exc:
            self.report['resource_recording_error'] = f'{type(exc).__name__}: {exc}'
            secondary = exc
        self.report['elapsed_seconds'] = time.monotonic() - self.start
        try:
            self.report['artifacts'] = inventory(self.out)
        except Exception as exc:
            self.report['artifact_inventory_error'] = f'{type(exc).__name__}: {exc}'
            secondary = exc
        if secondary is not None and error is None:
            self.report.update(status='failed', error=f'{type(secondary).__name__}: {secondary}')
        write_json(self.out / 'run.json', self.report)
        if secondary is not None and error is None:
            raise RuntimeError('evidence finalization failed; inspect run.json') from secondary


@contextmanager
def session(*args, **kwargs):
    current = Session(*args, **kwargs)
    try:
        yield current
    except BaseException as exc:
        current.finish(exc)
        raise
    else:
        current.finish()


def approve(run_dir, output, review_note):
    report = read_json(Path(run_dir) / 'run.json')
    validate_smoke(run_dir, report)
    if (report['status'] != 'passed' or report['mode'] != 'smoke' or not review_note.strip()
            or not (report.get('trainer_validated') or report.get('model_generation_validated'))):
        raise ValueError('only a complete real smoke can be reviewed for admission')
    result = {'approved': True, 'review_note': review_note, 'run_dir': str(Path(run_dir).resolve()),
              'run_report_sha256': file_identity(Path(run_dir) / 'run.json')['sha256']}
    write_json(output, result)
    return result


def validate_smoke(directory, report):
    """Structural real-evidence requirements, followed by explicit human review; not attestation."""
    directory = Path(directory)
    if (report.get('schema_version') != 2 or report.get('mode') != 'smoke'
            or report.get('status') != 'passed' or report['code']['dirty']
            or not re.fullmatch('[0-9a-f]{40}', report['code']['commit'])
            or report['signature'] != fingerprint(report['identity'])):
        raise ValueError('not a complete identity-bound smoke report')
    verify_inventory(directory, report['artifacts'])
    gpu = report['gpu']
    if not all(type(gpu[k]) is int and gpu[k] > 0 for k in ('peak_allocated_bytes','peak_reserved_bytes','total_bytes')):
        raise ValueError('real GPU memory evidence missing')
    stage = report['stage']
    if stage in ('E1','E2','E3','E4'):
        required = {'checkpoint.json','gradients.jsonl','reload-logits.pt','trainer-log.json'}
        if not required <= set(report['artifacts']) or not report['trainer_validated'] or report['optimizer_steps'] != 2:
            raise ValueError('Trainer update/reload evidence missing')
        events = [json.loads(s) for s in (directory/'gradients.jsonl').read_text().splitlines()]
        norms = [e['gradient_norm'] for e in events]
        nonzero_steps = sum(norm > 0 for norm in norms)
        if (len(events) != 2 or [e['optimizer_step'] for e in events] != [1,2]
                or any(not math.isfinite(norm) or norm < 0 for norm in norms)
                or nonzero_steps == 0 or (stage == 'E1' and nonzero_steps != 2)
                or not any(report['max_parameter_deltas'].values())
                or any(not math.isfinite(v) for v in report['max_parameter_deltas'].values())):
            raise ValueError('effective finite optimizer updates missing')
        if (('nonzero_gradient_steps' in report and report['nonzero_gradient_steps'] != nonzero_steps)
                or ('zero_gradient_steps' in report and report['zero_gradient_steps'] != 2 - nonzero_steps)
                or any('has_gradient_signal' in e
                       and e['has_gradient_signal'] != (e['gradient_norm'] > 0) for e in events)):
            raise ValueError('gradient signal counts inconsistent with recorded events')
        if stage in ('E2','E3','E4') and not {'reference-logits.pt','reward-rollouts.jsonl','reward-groups.jsonl'} <= set(report['artifacts']):
            raise ValueError('GRPO reference/group evidence missing')
    elif stage in ('E0_eval','E1_eval','E2_eval','E3_eval','E4_eval','weak_audit'):
        if not {'raw.jsonl','scored.jsonl','metrics.json'} <= set(report['artifacts']) or not report['model_generation_validated']:
            raise ValueError('model generation evidence missing')
        raw = (directory/'raw.jsonl').read_text().splitlines()
        scored = (directory/'scored.jsonl').read_text().splitlines()
        if not raw or len(raw) != len(scored) or len(raw) != report['n']:
            raise ValueError('generation sample inventory mismatch')
    else:
        raise ValueError('unknown smoke stage')
