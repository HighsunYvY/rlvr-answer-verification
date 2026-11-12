"""Completion-only token/label validation and the SFT smoke compatibility entry point."""
import argparse
import json

from .verifier import strict_verifier


def tokenize_completion(row, tokenizer, max_length):
    completion = row['completion']
    if (len(completion) != 1 or completion[0]['role'] != 'assistant'
            or not strict_verifier(completion[0]['content'], row['ground_truth'])):
        raise ValueError('invalid frozen demonstration')
    prompt = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row['prompt'] + completion, tokenize=True,
                                         add_generation_prompt=False)
    if not prompt or full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError('chat template prompt/completion boundary mismatch')
    if len(full) > max_length:
        raise ValueError('SFT smoke refuses truncated demonstrations')
    if tokenizer.eos_token_id not in full[len(prompt):]:
        raise ValueError('assistant completion must include EOS')
    return {'input_ids': full, 'completion_mask': [0] * len(prompt) + [1] * (len(full) - len(prompt))}


def inspect_labels(feature, batch):
    ids, mask = feature['input_ids'], feature['completion_mask']
    expected = [token if keep else -100 for token, keep in zip(ids, mask, strict=True)]
    if (batch['input_ids'][0][:len(ids)] != ids or batch['labels'][0][:len(ids)] != expected
            or any(v != -100 for v in batch['labels'][0][len(ids):])
            or batch['attention_mask'][0][:len(ids)] != [1] * len(ids)
            or any(batch['attention_mask'][0][len(ids):])):
        raise ValueError('Trainer collator labels/padding do not match completion mask')
    if not any(label != -100 for label in expected[1:]):
        raise ValueError('no shifted supervised targets')
    return {'prompt_tokens': mask.count(0), 'completion_tokens': mask.count(1),
            'shifted_targets': sum(v != -100 for v in expected[1:])}



def run(preparation_dir, model_dir, output_dir):
    from .training import train
    return train('E1', preparation_dir, model_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Two-step LoRA SFT smoke; no implicit downloads")
    parser.add_argument('--preparation-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.preparation_dir, args.model_dir, args.output_dir), indent=2))
