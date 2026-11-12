"""Read-only target readiness checks; never load models or run a Trainer."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def check_environment(project_file, probe_gpu=False):
    project_file = Path(project_file)
    with project_file.open('rb') as stream:
        project = tomllib.load(stream)
    errors = []
    versions = {}
    for requirement in project['project']['optional-dependencies']['train']:
        name, expected = requirement.split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {'expected': expected, 'installed': actual}
        # CUDA wheels can carry a local version such as +cu128.
        if actual is None or actual.split('+')[0] != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 11):
        errors.append('target requires Linux and Python 3.11')
    result = {'schema_version': 1, 'check': 'environment_preflight',
              'python': platform.python_version(), 'platform': platform.platform(),
              'packages': versions, 'gpu': {'status': 'not_requested'},
              'model_training_validated': False}
    if probe_gpu:
        try:
            # Import errors from the actual stack must be visible, not just metadata.
            for name in ('transformers', 'trl', 'peft', 'datasets', 'accelerate'):
                importlib.import_module(name)
            result['gpu'] = cuda_probe()
        except Exception as exc:
            result['gpu'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            errors.append('stack import or CUDA kernel check failed')
    result['errors'] = errors
    result['passed'] = not errors
    return result


def cuda_probe():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU/MPS fallback is forbidden')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported by current device/build')
    free, total = torch.cuda.mem_get_info()
    # Exercise allocation, matmul, autograd and synchronization on the selected GPU.
    with torch.enable_grad():
        x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16, requires_grad=True)
        loss = (x @ x).float().mean()
        loss.backward()
        torch.cuda.synchronize()
        if loss.item() != 32.0 or x.grad is None or not torch.isfinite(x.grad).all().item():
            raise RuntimeError('nonfinite or unexpected CUDA computation')
    return {'status': 'passed', 'name': torch.cuda.get_device_name(),
            'runtime': torch.version.cuda, 'capability': torch.cuda.get_device_capability(),
            'compiled_arches': torch.cuda.get_arch_list(), 'bf16': True,
            'free_bytes_before_probe': free, 'total_bytes': total}


def code_revision(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


def check_trainer_api(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    contracts = {}
    with tempfile.TemporaryDirectory(prefix='verifierlab-api-') as temp:
        for name, config_type, trainer_type in (
            ('sft', SFTConfig, SFTTrainer),
            ('grpo_strict', GRPOConfig, GRPOTrainer),
            ('grpo_weak', GRPOConfig, GRPOTrainer),
        ):
            source = json.loads((root / 'configs' / f'{name}.yaml').read_text())
            kwargs = dict(source['trainer'])
            kwargs.update(source['profiles']['smoke'])
            kwargs['output_dir'] = str(Path(temp) / name)
            if 'reward_weights' in source:
                kwargs['reward_weights'] = source['reward_weights']
            config_type(**kwargs)
            LoraConfig(**source['lora'])
            parameters = inspect.signature(trainer_type.__init__).parameters
            required = {'processing_class', 'peft_config', 'args', 'train_dataset'}
            if name.startswith('grpo'):
                required.add('reward_funcs')
            missing = required - parameters.keys()
            if missing:
                raise RuntimeError(f'{name}: missing Trainer arguments {sorted(missing)}')
            contracts[name] = {'config_constructed': True, 'lora_config_constructed': True,
                               'trainer_arguments_present': sorted(required)}
    return {'contracts': contracts, 'trainer_instantiated': False, 'trainer_smoke_passed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-gpu', action='store_true', help='run tiny CUDA BF16 autograd check')
    parser.add_argument('--check-trainer-api', action='store_true', help='construct pinned Trainer configs offline')
    parser.add_argument('--output', type=Path, help='new JSON evidence file; parent must exist')
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error('output already exists; refusing overwrite')
    root = Path(__file__).resolve().parents[2]
    project = root / 'pyproject.toml'
    result = check_environment(project, args.probe_gpu)
    result['code'] = code_revision(root)
    result['pyproject_sha256'] = hashlib.sha256(project.read_bytes()).hexdigest()
    if args.check_trainer_api:
        try:
            result['trainer_api'] = check_trainer_api(root)
        except Exception as exc:
            result['errors'].append(f'Trainer API check failed: {type(exc).__name__}: {exc}')
            result['passed'] = False
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
