"""Profiled SFT/strict-weak GRPO, retaining upstream Trainer optimization loops."""
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path

from .data import grpo_record
from .evaluate import score_prediction
from .experiment import (ROOT, checkpoint, file_identity, load_policy, read_json,
                         save_checkpoint, session, verify_inventory, write_json)
from .rewards import completion_text, format_reward
from .rollouts import group_signal, require_weak_admission
from .sft import inspect_labels, tokenize_completion
from .generate import completion_record
from .utils import validate_grpo


# Accelerate wraps Trainer forward in autocast; a freshly loaded model has no wrapper.
# Probe both through the same explicit context rather than comparing different arithmetic.
FORWARD_PROBE = {'device_type': 'cuda', 'autocast_dtype': 'bfloat16',
                 'use_cache': False, 'atol': 1e-4, 'rtol': 1e-4}


def forward_probe(model, input_ids, torch):
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       use_cache=False).logits[0, -1]
    return logits.detach().float().cpu()


def compare_probe(before, after, torch):
    finite = bool(torch.isfinite(before).all().item() and torch.isfinite(after).all().item())
    close = torch.isclose(before, after, atol=FORWARD_PROBE['atol'], rtol=FORWARD_PROBE['rtol'])
    return {'context': dict(FORWARD_PROBE), 'finite': finite,
            'passed': finite and bool(close.all().item()),
            'max_absolute_difference': (before - after).abs().max().item() if finite else None,
            'mismatched_logits': int((~close).sum().item()), 'logits': before.numel()}


@contextmanager
def initial_validation_context(trainer, grpo=False):
    """Bridge pinned TRL's loss divisor before Transformers initializes train state."""
    attribute = 'current_gradient_accumulation_steps'
    temporary = grpo and not hasattr(trainer, attribute)
    divisor = getattr(trainer, attribute, trainer.args.gradient_accumulation_steps) if grpo else 1
    if type(divisor) is not int or divisor < 1:
        raise ValueError('invalid initial validation loss divisor')
    if temporary:
        setattr(trainer, attribute, divisor)
    try:
        yield {'loss_divisor': divisor, 'temporary_train_state': temporary}
    finally:
        if temporary:
            delattr(trainer, attribute)


def evidence_callback(params, out, torch, allow_zero=False):
    from transformers import TrainerCallback

    class Evidence(TrainerCallback):
        def __init__(self):
            self.events = []
            self.baseline = {}

        def on_train_begin(self, args, state, control, **kwargs):
            self.baseline = {n: p.detach().float().cpu().clone() for n, p in params.items()}

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad for p in params.values() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError('missing/nonfinite gradients')
            norm = math.sqrt(sum(g.detach().float().norm().item() ** 2 for g in grads))
            if not math.isfinite(norm) or norm < 0 or (norm == 0 and not allow_zero):
                raise RuntimeError('zero/nonfinite gradient signal')
            # Equal-reward GRPO groups can legitimately yield zero advantage/gradient.
            # Retain the step; admission still requires signal and actual parameter changes.
            event = {'optimizer_step': state.global_step + 1, 'gradient_norm': norm,
                     'has_gradient_signal': norm > 0}
            self.events.append(event)
            with (out / 'gradients.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with (out / 'trainer-log.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': state.global_step, **logs}) + '\n')
                if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in logs.values()):
                    raise RuntimeError('nonfinite Trainer diagnostic')

    return Evidence()


def trace_reward(dataset_rows, out, verifier, generations, tokenizer=None, limit=None):
    lookup = {r['sample_id']: r for r in dataset_rows}
    calls = []

    def correctness(completions, ground_truth, sample_id, prompts, completion_ids, partition, **kwargs):
        n = len(completions)
        with (out / 'reward-inputs.jsonl').open('a') as stream:
            stream.write(json.dumps({'call': len(calls), 'optimizer_step': getattr(kwargs.get('trainer_state'), 'global_step', None), 'prompts': prompts, 'completions': completions,
                'ground_truth': ground_truth, 'sample_id': sample_id, 'partition': partition,
                'completion_ids': completion_ids}, ensure_ascii=False) + '\n')
        if n == 0 or not (n == len(sample_id) == len(ground_truth) == len(prompts) == len(completion_ids) == len(partition)) or n % generations:
            raise ValueError('GRPO gold/G metadata length mismatch')
        rows, groups = [], []
        for i in range(n):
            gold = lookup[sample_id[i]]
            if (ground_truth[i] != gold['ground_truth'] or prompts[i] != gold['prompt']
                    or partition[i] != gold['partition']):
                raise ValueError('GRPO forwarded gold/prompt mismatch')
            row = score_prediction(f'call{len(calls)}:{i}', completion_text(completions[i]), ground_truth[i],
                                   len(completion_ids[i]))
            if tokenizer is not None:
                prompt_ids = tokenizer.apply_chat_template(gold['prompt'], tokenize=True, add_generation_prompt=True)
                decoded = completion_record(row['sample_id'], completion_ids[i], tokenizer,
                                            [tokenizer.eos_token_id], limit, len(prompt_ids))
                if decoded['completion'] != row['completion']:
                    raise ValueError('Trainer completion text/token IDs differ')
                row.update(decoded)
            row.update(source_sample_id=sample_id[i], trainer_completion_ids=completion_ids[i])
            rows.append(row)
        for i in range(0, n, generations):
            if len(set(sample_id[i:i + generations])) != 1:
                raise ValueError('GRPO group ordering mismatch')
            members = rows[i:i + generations]
            total = [0.2 * r['format_reward'] + r[f'{verifier}_accuracy_reward'] for r in members]
            groups.append({'sample_id': sample_id[i], 'partition': partition[i], 'selected_total_rewards': total,
                           'variable_selected_total': len(set(total)) > 1,
                           'weak_signal': group_signal(members)})
        call = {'call': len(calls), 'optimizer_step': getattr(kwargs.get('trainer_state'), 'global_step', None), 'groups': groups}
        calls.append(call)
        with (out / 'reward-rollouts.jsonl').open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (out / 'reward-groups.jsonl').open('a') as stream:
            stream.write(json.dumps(call) + '\n')
        return [r[f'{verifier}_accuracy_reward'] for r in rows]

    correctness.__name__ = f'{verifier}_accuracy_reward_with_evidence'
    return correctness, calls


def train(stage, preparation_dir, model_dir, output_dir, profile='smoke', mode='smoke',
          sft_dir=None, approval=None, weak_admission=None, resume=None, max_steps=None):
    if stage not in ('E1', 'E2', 'E3'):
        raise ValueError('unknown optimization stage')
    if max_steps is not None and (stage == 'E1' or type(max_steps) is not int or max_steps < 2):
        raise ValueError('max_steps override requires E2/E3 and an integer of at least two')
    filename = {'E1': 'sft', 'E2': 'grpo_strict', 'E3': 'grpo_weak'}[stage]
    source = read_json(ROOT / f'configs/{filename}.yaml')
    trainer_settings = {**source['trainer'], **source['profiles'][profile]}
    budget = trainer_settings.pop('max_steps')
    if max_steps is not None:
        budget = max_steps
    settings = {'trainer': trainer_settings, 'lora': source['lora'], 'seed': source['seed'],
                'reward_weights': source.get('reward_weights'), 'max_steps': budget,
                'forward_probe': dict(FORWARD_PROBE)}
    # Step count belongs to the declared budget, not mode: smoke and run must match the same signature.
    if stage == 'E1' and sft_dir:
        raise ValueError('E1 initializes from original Instruct, not another adapter')
    sft_meta = checkpoint(sft_dir) if sft_dir else None
    if stage != 'E1' and (not sft_meta or sft_meta['stage'] != 'E1'):
        raise ValueError('E2/E3 must independently initialize from an E1 checkpoint')
    with session(output_dir, stage, profile, mode, model_dir, preparation_dir) as current:
        current.initialize(settings, sft_meta)
        sources = current.report['sources']
        if source['model'] != {'name': sources['model']['repo_id'], 'revision': sources['model']['revision'],
                               'tokenizer_revision': sources['tokenizer']['revision']} or source['seed'] != 42:
            raise ValueError('configured model/tokenizer/seed mismatch')
        if stage != 'E1':
            if source['correctness_verifier'] != ('strict' if stage == 'E2' else 'weak'):
                raise ValueError('condition/verifier mismatch')
            if profile != 'smoke' and (sft_meta['training_contract']['mode'] != 'run'
                    or sft_meta['training_contract']['effective_steps'] != sft_meta['training_contract']['settings']['max_steps']):
                raise ValueError('pilot/full GRPO requires the completed target-profile E1 budget')
        current.admit(approval)
        if stage == 'E3':
            require_weak_admission(weak_admission, current, sft_meta)
        torch, tokenizer = current.torch, current.tokenizer
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
        kwargs = {**trainer_settings, 'max_steps': 2 if mode == 'smoke' else budget,
                  'output_dir': str(current.out / 'trainer'), 'logging_nan_inf_filter': False}
        if mode == 'smoke':
            kwargs['save_strategy'] = 'no'
        validation = current.splits['validation'][:4] if mode == 'smoke' else current.splits['validation']
        kwargs.update(eval_strategy='steps', eval_steps=10, per_device_eval_batch_size=1)
        tokenizer.padding_side = 'right' if stage == 'E1' else 'left'
        reference_logits = None
        if stage == 'E1':
            rows = current.splits['sft'][:8] if mode == 'smoke' else current.splits['sft']
            features = [tokenize_completion(r, tokenizer, kwargs['max_length']) for r in rows]
            model = load_policy(model_dir)
            model.config.use_cache = False
            kwargs['dataset_kwargs'] = {'skip_prepare_dataset': True}
            trainer = SFTTrainer(model=model, args=SFTConfig(**kwargs), processing_class=tokenizer,
                                 peft_config=LoraConfig(**source['lora']), train_dataset=Dataset.from_list(features),
                                 eval_dataset=Dataset.from_list([tokenize_completion(r, tokenizer, kwargs['max_length']) for r in validation]))
            collated = {k: v.tolist() for k, v in trainer.data_collator(features).items()}
            masks = []
            for i, (row, feature) in enumerate(zip(rows, features, strict=True)):
                batch = {k: [v[i]] for k, v in collated.items()}
                if any(v != tokenizer.pad_token_id for v in batch['input_ids'][0][len(feature['input_ids']):]):
                    raise ValueError('collator padding ID mismatch')
                masks.append({'sample_id': row['sample_id'], **inspect_labels(feature, batch),
                              'input_ids': batch['input_ids'][0], 'labels': batch['labels'][0]})
            write_json(current.out / 'masks.json', masks)
            probe = torch.tensor([features[0]['input_ids']], device='cuda')
        else:
            rows = current.splits['grpo'][:8] if mode == 'smoke' else current.splits['grpo']
            rows = [{**grpo_record(r), 'partition': 'train'} for r in rows]
            validation = [{**grpo_record(r), 'partition': 'validation'} for r in validation]
            kwargs['per_device_eval_batch_size'] = kwargs['num_generations']
            for row in rows + validation:
                ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
                if len(ids) > kwargs['max_prompt_length']:
                    raise ValueError('GRPO prompt too long; silent prompt truncation prohibited')
            # Merge E1 first, then attach a new RL adapter. Disabling RL returns to E1.
            model = load_policy(model_dir, sft_dir).to('cuda').eval()
            probe = tokenizer.apply_chat_template(rows[0]['prompt'], tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to('cuda')
            reference_logits = forward_probe(model, probe, torch)
            effective = {**source, 'trainer': kwargs}
            current.report['batch_validation'] = validate_grpo(effective, 1)
            correctness, calls = trace_reward(rows + validation, current.out, source['correctness_verifier'], kwargs['num_generations'], tokenizer, kwargs['max_completion_length'])
            kwargs['reward_weights'] = source['reward_weights']
            kwargs['disable_dropout'] = True
            trainer = GRPOTrainer(model=model, args=GRPOConfig(**kwargs), processing_class=tokenizer,
                peft_config=LoraConfig(**source['lora']), train_dataset=Dataset.from_list(rows),
                reward_funcs=[format_reward, correctness], eval_dataset=Dataset.from_list(validation))
            if trainer.ref_model is not None or trainer.beta != source['trainer']['beta']:
                raise ValueError('unexpected TRL reference/KL configuration')
            trainer.model.eval()
            with trainer.model.disable_adapter():
                disabled = forward_probe(trainer.model, probe, torch)
            if not torch.allclose(reference_logits, disabled, atol=1e-4, rtol=1e-4):
                raise RuntimeError('disabled RL adapter is not the E1 reference')
            current.report['reference'] = {'strategy': 'merged_E1_plus_fresh_RL_LoRA',
                'sft_identity': sft_meta['adapter_sha256'], 'beta': trainer.beta,
                'initial_max_logit_difference': (reference_logits - disabled).abs().max().item()}
        if trainer.accelerator.num_processes != 1:
            raise ValueError('single process required')
        params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        if not params or any(not ('.lora_A.' in n or '.lora_B.' in n)
                             or not ('.q_proj.' in n or '.v_proj.' in n) for n in params):
            raise ValueError('unexpected trainable parameters')
        current.report['trainable'] = {n: list(p.shape) for n, p in params.items()}
        current.report['resolved_trainer'] = trainer.args.to_dict()
        current.report['training_sample_ids'] = [r['sample_id'] for r in rows]
        current.report['validation_sample_ids'] = [r['sample_id'] for r in validation]
        callback = evidence_callback(params, current.out, torch, allow_zero=stage != 'E1')
        trainer.add_callback(callback)
        resumed_step = 0
        if resume:
            old_root = Path(resume).parent.parent
            old = read_json(old_root / 'run.json')
            verify_inventory(old_root, old['artifacts'])
            if old['signature'] != current.report['signature'] or mode == 'smoke':
                raise ValueError('resume requires the same run signature and non-smoke mode')
            resumed_step = read_json(Path(resume) / 'trainer_state.json')['global_step']
            if not 0 < resumed_step < kwargs['max_steps']:
                raise ValueError('resume step outside remaining budget')
            current.report['resumed_report_sha256'] = file_identity(old_root / 'run.json')['sha256']
        current.report['initial_validation_context'] = 'fresh_initialization_not_resumed_weights' if resume else 'initial_policy'
        with initial_validation_context(trainer, grpo=stage != 'E1') as normalization:
            current.report['initial_validation_normalization'] = normalization
            current.report['initial_validation_metrics'] = trainer.evaluate()
        result = trainer.train(resume_from_checkpoint=resume)
        deltas = {n: (p.detach().float().cpu() - callback.baseline[n]).abs().max().item() for n, p in params.items()}
        nonzero_steps = sum(e['gradient_norm'] > 0 for e in callback.events)
        current.report.update(training_metrics=result.metrics, max_parameter_deltas=deltas,
                              nonzero_gradient_steps=nonzero_steps,
                              zero_gradient_steps=len(callback.events) - nonzero_steps)
        write_json(current.out / 'trainer-log.json', trainer.state.log_history)
        if (trainer.state.global_step != kwargs['max_steps']
                or len(callback.events) != kwargs['max_steps'] - resumed_step
                or nonzero_steps == 0
                or not math.isfinite(result.training_loss) or not any(deltas.values())
                or any(not math.isfinite(v) for v in deltas.values())):
            raise RuntimeError('finite loss/effective optimizer updates not established')
        if stage != 'E1':
            if not calls or not any(g['variable_selected_total'] and g['partition'] == 'train' for c in calls for g in c['groups']):
                raise RuntimeError('no effective within-group reward signal')
            trainer.model.eval()
            with trainer.model.disable_adapter():
                final_reference = forward_probe(trainer.model, probe, torch)
            torch.save({'E1': reference_logits, 'after_RL_disabled': final_reference}, current.out / 'reference-logits.pt')
            if not torch.allclose(reference_logits, final_reference, atol=1e-4, rtol=1e-4):
                raise RuntimeError('E1 reference changed during RL')
        trainer.model.eval()
        before = forward_probe(trainer.model, probe, torch)
        trained = {n: p.detach().cpu().clone() for n, p in params.items()}
        meta = save_checkpoint(trainer.model, tokenizer, current.out, stage,
                               sft_meta['adapter_sha256'] if sft_meta else None,
                               contract={'settings': settings, 'effective_steps': kwargs['max_steps'],
                                         'manifest': current.report['manifest_sha256'], 'mode': mode})
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        restored = load_policy(model_dir, current.out, sft_dir, merge_e1=False, allow_unvalidated=True).to('cuda').eval()
        after_params = dict(restored.named_parameters())
        if any(n not in after_params or not torch.equal(p, after_params[n].detach().cpu()) for n, p in trained.items()):
            raise RuntimeError('saved adapter parameter mismatch')
        after = forward_probe(restored, probe, torch)
        torch.save({'before': before, 'after': after}, current.out / 'reload-logits.pt')
        comparison = compare_probe(before, after, torch)
        current.report['reload_comparison'] = comparison
        current.report['reload_max_logit_difference'] = comparison['max_absolute_difference']
        write_json(current.out / 'reload-comparison.json', comparison)
        if not comparison['passed']:
            raise RuntimeError('checkpoint reload forward mismatch; inspect reload-comparison.json')
        current.report.update(status='passed', trainer_validated=True, optimizer_steps=kwargs['max_steps'],
                              checkpoint=meta, reload_max_logit_difference=(before - after).abs().max().item())
    return current.report
