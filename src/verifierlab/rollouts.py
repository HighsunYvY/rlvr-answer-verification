"""Shared evaluation, sampled validation, and preregistered E1 weak-verifier audit."""
import json
from pathlib import Path
import statistics
import time

from .evaluate import aggregate_metrics, score_prediction
from .experiment import ROOT, checkpoint, file_identity, fingerprint, load_policy, read_json, session, write_json
from .generate import completion_record


def group_signal(rows):
    total = [0.2 * r['format_reward'] + r['weak_accuracy_reward'] for r in rows]
    accuracy = [r['weak_accuracy_reward'] for r in rows]
    return {'total_rewards': total, 'accuracy_rewards': accuracy,
            'total_std': statistics.stdev(total), 'accuracy_std': statistics.stdev(accuracy),
            'variable_total': len(set(total)) > 1, 'variable_accuracy': len(set(accuracy)) > 1}


def run_rollouts(preparation_dir, model_dir, output_dir, profile='smoke', mode='smoke',
                 checkpoint_dir=None, sft_dir=None, split='test', approval=None, weak_audit=False,
                 sampled_validation=False, sampled_test=False):
    policy = checkpoint(checkpoint_dir) if checkpoint_dir else None
    stage = 'weak_audit' if weak_audit else (policy['stage'] if policy else 'E0') + '_eval'
    criteria = read_json(ROOT / 'configs/weak_audit.json') if weak_audit else None
    if weak_audit and (not policy or policy['stage'] != 'E1' or split != 'validation'):
        raise ValueError('weak audit requires E1 and validation only')
    if sampled_validation or sampled_test:
        expected_split = 'test' if sampled_test else 'validation'
        if (sampled_validation and sampled_test or split != expected_split
                or not policy or policy['stage'] not in ('E1', 'E2', 'E3', 'E4')):
            raise ValueError(f'sampled evaluation requires a trained checkpoint and the {expected_split} split')
    grouped = weak_audit or sampled_validation or sampled_test
    decode = read_json(ROOT / 'configs/evaluation.yaml')['generation'].copy()
    prompt_limit = None
    if grouped:
        source = read_json(ROOT / 'configs/grpo_weak.yaml')
        opts = {**source['trainer'], **source['profiles'][profile]}
        prompt_limit = opts['max_prompt_length']
        decode = {'do_sample': True, 'max_new_tokens': opts['max_completion_length'],
                  'num_return_sequences': criteria['num_generations'] if criteria else opts['num_generations'],
                  'temperature': opts['temperature'], 'top_p': opts['top_p'], 'top_k': None}
    settings = {'generation': decode, 'split': split, 'seed': 42, 'prompt_limit': prompt_limit,
                'criteria': criteria, 'validation_groups':
                read_json(ROOT / 'configs/data.yaml')['profiles'][profile]['validation'] if weak_audit else None}
    with session(output_dir, stage, profile, mode, model_dir, preparation_dir) as current:
        current.initialize(settings, policy)
        current.admit(approval)
        model = load_policy(model_dir, checkpoint_dir, sft_dir).to('cuda').eval()
        tokenizer, torch = current.tokenizer, current.torch
        from transformers import GenerationConfig
        eos = model.generation_config.eos_token_id
        eos_ids = [tokenizer.eos_token_id] if grouped else ([eos] if isinstance(eos, int) else eos)
        if not eos_ids or tokenizer.eos_token_id not in eos_ids:
            raise ValueError('model/tokenizer EOS mismatch')
        generation = GenerationConfig(**decode, num_beams=1, eos_token_id=eos_ids,
                                       pad_token_id=tokenizer.pad_token_id, use_cache=True)
        current.report['resolved_generation'] = generation.to_dict()
        rows = current.splits[split]
        # Smoke exercises a fixed prefix; final audit admission requires the entire preregistered pool.
        if mode == 'smoke':
            rows = rows[:4]
        scored, groups = [], []
        start = time.monotonic()
        with (current.out / 'raw.jsonl').open('x') as raw, (current.out / 'scored.jsonl').open('x') as stream:
            for frozen in rows:
                batch = tokenizer.apply_chat_template(frozen['prompt'], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
                width = batch['input_ids'].shape[1]
                if ((prompt_limit is not None and width > prompt_limit)
                        or width + decode['max_new_tokens'] > model.config.max_position_embeddings):
                    raise ValueError('prompt/context limit exceeded; no silent truncation')
                torch.cuda.synchronize()
                sample_start = time.monotonic()
                with torch.inference_mode():
                    sequences = model.generate(**batch, generation_config=generation)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - sample_start
                members = []
                for member, sequence in enumerate(sequences):
                    ids = sequence[width:].tolist()
                    identifier = f"{frozen['sample_id']}#g{member}" if grouped else frozen['sample_id']
                    raw.write(json.dumps({'sample_id': identifier, 'source_sample_id': frozen['sample_id'],
                                          'prompt_tokens': width, 'generated_token_ids': ids}) + '\n')
                    raw.flush()
                    if not torch.equal(sequence[:width], batch['input_ids'][0]):
                        raise ValueError('returned prefix mismatch')
                    prediction = completion_record(identifier, ids, tokenizer, eos_ids,
                                                   decode['max_new_tokens'], width)
                    prediction.update(source_sample_id=frozen['sample_id'], group_member=member,
                                      group_generation_seconds=elapsed)
                    result = {**prediction, **score_prediction(identifier, prediction['completion'],
                                                frozen['ground_truth'], prediction['generated_tokens'])}
                    scored.append(result)
                    members.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                if len(members) != decode['num_return_sequences']:
                    raise ValueError('incomplete generation group')
                if weak_audit:
                    groups.append({'sample_id': frozen['sample_id'], **group_signal(members)})
        elapsed = time.monotonic() - start
        metrics = aggregate_metrics(scored)
        metrics.update(truncation_rate=sum(r['truncated'] for r in scored) / len(scored),
                       sample_loop_seconds=elapsed,
                       tokens_per_second=sum(r['generated_tokens'] for r in scored) / elapsed)
        if weak_audit:
            variable = sum(g['variable_total'] for g in groups)
            eligible = (metrics['weak_false_positive_count'] >= criteria['min_false_positives']
                        and variable >= criteria['min_variable_total_reward_groups']
                        and variable / len(groups) >= criteria['min_variable_total_reward_fraction'])
            metrics.update(groups=len(groups), variable_total_groups=variable,
                           variable_accuracy_groups=sum(g['variable_accuracy'] for g in groups),
                           variable_total_fraction=variable / len(groups), eligible=eligible)
            write_json(current.out / 'groups.json', groups)
            current.report['criteria_sha256'] = fingerprint(criteria)
            # Negative admission is a completed audit, not a hidden failed experiment.
            current.report['eligible'] = eligible
        write_json(current.out / 'metrics.json', metrics)
        current.report.update(status='passed', model_generation_validated=True, n=len(scored),
                              metrics=metrics, manual_review_required=True)
    return current.report


def approve_weak(run_dir, output, review_note):
    from .experiment import verify_inventory
    report = read_json(Path(run_dir) / 'run.json')
    verify_inventory(run_dir, report['artifacts'])
    criteria = read_json(ROOT / 'configs/weak_audit.json')
    if (report['stage'] != 'weak_audit' or report['status'] != 'passed'
            or report.get('eligible') is not True or not review_note.strip()
            or report['criteria_sha256'] != fingerprint(criteria)
            or report['metrics']['groups'] != report['settings']['validation_groups']):
        raise ValueError('weak audit is incomplete/ineligible/stale; preserve the negative result')
    result = {'review_note': review_note, 'run_dir': str(Path(run_dir).resolve()),
              'run_report_sha256': file_identity(Path(run_dir) / 'run.json')['sha256']}
    write_json(output, result)
    return result


def require_weak_admission(path, current, policy):
    from .experiment import verify_inventory
    if not path:
        raise ValueError('weak-reward training requires a reviewed E1 validation audit')
    proof = read_json(path)
    root = Path(proof['run_dir'])
    report = read_json(root / 'run.json')
    verify_inventory(root, report['artifacts'])
    if (not proof.get('review_note') or proof['run_report_sha256'] != file_identity(root / 'run.json')['sha256']
            or report['stage'] != 'weak_audit' or report['status'] != 'passed' or not report['eligible']
            or report['metrics']['groups'] != report['settings']['validation_groups']
            or report['policy_identity'] != policy
            or report['manifest_sha256'] != current.report['manifest_sha256']
            or report['criteria_sha256'] != fingerprint(read_json(ROOT / 'configs/weak_audit.json'))
            or report['identity']['source_tree'] != current.report['identity']['source_tree']
            or report['profile'] != current.report['profile']
            or report['settings']['generation']['temperature'] != current.report['settings']['trainer']['temperature']
            or report['settings']['generation']['top_p'] != current.report['settings']['trainer']['top_p']
            or report['settings']['generation']['num_return_sequences'] != current.report['settings']['trainer']['num_generations']
            or report['settings']['generation']['max_new_tokens'] != current.report['settings']['trainer']['max_completion_length']):
        raise ValueError('weak audit does not match E1/manifest/budget/criteria')
    current.report['weak_admission_sha256'] = file_identity(path)['sha256']
