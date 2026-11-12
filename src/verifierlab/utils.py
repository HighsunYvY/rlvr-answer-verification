"""Dependency-free config loading and optional runtime diagnostics."""
import importlib.metadata
import json
import platform
import shutil

PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "pytest", "ruff")


def load_config(path):
    # JSON is a YAML 1.2 subset; deliberately avoids a mandatory YAML dependency.
    with open(path) as f:
        return json.load(f)


def diagnostics(probe_gpu=False):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "packages": versions,
              "nvidia_smi": shutil.which("nvidia-smi"), "cuda": "not probed"}
    if probe_gpu and versions['torch']:
        import torch
        result['cuda'] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result['cuda'].update(name=torch.cuda.get_device_name(), free_bytes=free,
                                  total_bytes=total, capability=torch.cuda.get_device_capability(),
                                  compiled_arches=torch.cuda.get_arch_list(),
                                  bf16=torch.cuda.is_bf16_supported())
    return result


def validate_grpo(config, world_size=1):
    args = config['trainer']
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'num_generations'):
        if type(args[key]) is not int or args[key] < 1:
            raise ValueError(f"{key} must be positive integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be positive integer")
    micro = world_size * args['per_device_train_batch_size']
    effective = micro * args['gradient_accumulation_steps']
    generation = args.get('generation_batch_size')
    steps = args.get('steps_per_generation')
    if generation is not None and steps is not None:
        raise ValueError("generation_batch_size and steps_per_generation are mutually exclusive")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps_per_generation must be positive integer")
    if generation is None:
        generation = micro * (steps or args['gradient_accumulation_steps'])
    if type(generation) is not int or generation < 1 or generation % micro:
        raise ValueError("generation batch must divide into global microbatches")
    g = args['num_generations']
    if g < 2 or generation % g:
        raise ValueError("GRPO needs at least two generations and complete prompt groups")
    if args.get('eval_strategy', 'no') != 'no' and (world_size * args['per_device_eval_batch_size']) % g:
        raise ValueError("evaluation batch must contain complete groups")
    for key in ('max_prompt_length', 'max_completion_length'):
        if type(args[key]) is not int or args[key] < 1:
            raise ValueError(f"{key} must be a positive token limit")
    if args.get('remove_unused_columns') is not False:
        raise ValueError("preserve ground_truth for reward callbacks")
    return {"effective_completion_batch": effective, "generation_completion_batch": generation,
            "prompts_per_generation_batch": generation // g}
