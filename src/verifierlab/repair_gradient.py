"""Measure how a single reward withdrawal changes the unstepped E1 LoRA policy gradient."""
import argparse
import json
from pathlib import Path

from .analysis import diagnose_scored_row
from .experiment import ROOT, checkpoint, file_identity, load_policy, read_json, session, verify_inventory
from .repair_replay import groups_from_run, repair_effects


def token_logps(model, tokenizer, question, row, temperature, torch):
    prompt = tokenizer.apply_chat_template(question['prompt'], tokenize=True, add_generation_prompt=True)
    tokens = row['generated_token_ids'][:row['generated_tokens']]
    if (not tokens or len(tokens) != row['generated_tokens'] or len(prompt) != row['prompt_tokens']
            or tokenizer.decode(tokens, skip_special_tokens=True) != row['completion']):
        raise ValueError('stored completion and frozen input tokenization disagree')
    ids = torch.tensor([prompt + tokens], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(tokens)+1).logits[:, :-1, :]
    scores = logits[0].float() / temperature
    return -torch.nn.functional.cross_entropy(scores, ids[0, -len(tokens):], reduction='none')


def flat_gradient(objective, params, torch):
    grads = torch.autograd.grad(objective, params)
    return torch.cat([g.detach().float().reshape(-1) for g in grads]).cpu()


def gradient_probe(preparation_dir, model_dir, checkpoint_dir, probe_dir, output_dir, seed, mode, admission):
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed

    path = Path(probe_dir)
    source = read_json(path / 'run.json')
    policy = checkpoint(checkpoint_dir)
    if (source['mode'] != 'run' or source['stage'] != 'repair_probe' or source['status'] != 'passed'
            or source['policy_identity'] != policy or policy['stage'] != 'E1'):
        raise ValueError('gradient probe requires complete rollouts from the same E1')
    verify_inventory(path, {'scored.jsonl': source['artifacts']['scored.jsonl']})
    settings = {'seed': seed, 'split': 'grpo', 'source_report': file_identity(path/'run.json')['sha256'],
                'temperature': source['settings']['generation']['temperature'],
                'lora': read_json(ROOT/'configs/grpo_weak.yaml')['lora']}
    with session(output_dir, 'repair_gradient', 'full', mode, model_dir, preparation_dir) as current:
        current.initialize(settings, policy)
        current.admit(admission)
        if source['manifest_sha256'] != current.report['manifest_sha256']:
            raise ValueError('gradient probe and generated bank require the same frozen questions')
        torch, tokenizer = current.torch, current.tokenizer
        base = load_policy(model_dir, checkpoint_dir).to('cuda')
        set_seed(seed)
        model = get_peft_model(base, LoraConfig(**settings['lora'])).eval()
        named = {k: p for k,p in model.named_parameters() if p.requires_grad}
        if not named or any('.lora_' not in k for k in named) or not any('.lora_B.' in k for k in named):
            raise ValueError('unexpected gradient parameter set')
        if any(torch.count_nonzero(p).item() for k,p in named.items() if '.lora_B.' in k):
            raise ValueError('fresh adapter must represent the unchanged E1 policy')
        params = tuple(named.values())
        by_id = {r['sample_id']: r for r in current.splits['grpo']}
        count = 0
        panel = source['settings']['repair_panel']
        schedule = [(q,d) for d in range(panel['draws']) for q in panel['sample_ids']]
        with (current.out/'group-gradients.jsonl').open('x') as stream:
            for (_, rows), (key,draw) in zip(groups_from_run(path,source),schedule,strict=True):
                if any(r['source_sample_id'] != key or r['panel_draw'] != draw for r in rows):
                    raise ValueError('group and frozen probe schedule differ')
                question = by_id[key]
                rows = [diagnose_scored_row(r, question['ground_truth']) for r in rows]
                context, events = repair_effects(rows)
                if not events or context == 'undetermined':
                    continue
                vectors = []
                for row in rows:
                    logps = token_logps(model, tokenizer, question, row, settings['temperature'], torch)
                    vectors.append(flat_gradient(logps.mean(), params, torch))
                vectors = torch.stack(vectors).double()
                gram = vectors @ vectors.T
                if not torch.isfinite(gram).all() or not (gram.diag() > 0).all():
                    raise RuntimeError('nonfinite or empty LoRA gradient')
                data = []
                for e in events:
                    gold = torch.tensor(e['final_only_advantages_with_format_reward'],dtype=torch.double)
                    before = (torch.tensor(e['before'],dtype=torch.double) - gold) / len(rows)
                    after = (torch.tensor(e['after'],dtype=torch.double) - gold) / len(rows)
                    errors = [float((v @ gram @ v).clamp_min(0).sqrt()) for v in (before,after)]
                    data.append({'member': e['member'], 'advantage_error_reduction': e['reduction_in_mean_absolute_advantage_error'],
                                 'gradient_error_before': errors[0], 'gradient_error_after': errors[1]})
                if count == 0:
                    # Independent backward pass through the clipped-ratio surrogate at ratio one.
                    coefs = torch.tensor(events[0]['after'],dtype=torch.double) - torch.tensor(events[0]['before'],dtype=torch.double)
                    direct = torch.zeros_like(vectors[0])
                    for coef, row in zip(coefs.tolist(), rows, strict=True):
                        logps = token_logps(model, tokenizer, question, row, settings['temperature'], torch)
                        ratio = (logps - logps.detach()).exp()
                        surrogate = torch.minimum(coef*ratio,coef*ratio.clamp(0.8,1.2)).mean()/len(rows)
                        direct += flat_gradient(surrogate, params, torch).double()
                    expected = (coefs @ vectors) / len(rows)
                    rel = float((direct-expected).norm()/expected.norm())
                    if not 0 <= rel < 0.025:
                        raise RuntimeError(f'direct gradient check failed: {rel}')
                    current.report['direct_gradient_relative_error'] = rel
                stream.write(json.dumps({'sample_id':key,'draw':draw,'context':context,'member_ids':[r['sample_id'] for r in rows],
                                         'gram':gram.tolist(),'events':data})+'\n')
                stream.flush()
                count += 1
                if mode == 'smoke' and count == 2:
                    break
        if count < 2:
            raise ValueError('insufficient confirmed groups for the gradient diagnostic')
        current.report.update(status='passed', gradient_validated=True, groups=count,
                              trainable_coordinates=sum(p.numel() for p in params))
    return current.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-dir','model-dir','checkpoint-dir','probe-dir','output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--mode', choices=['smoke','run'], required=True)
    parser.add_argument('--admission')
    result = gradient_probe(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('status','groups','direct_gradient_relative_error')}))
