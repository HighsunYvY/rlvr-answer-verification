"""Pure GSM8K parsing and deterministic, content-deduplicated splits."""
import hashlib
import json
from pathlib import Path
import random
import re
from .verifier import normalize_number

INSTRUCTION = "Solve the problem step by step. End with exactly one <answer>NUMBER</answer> tag."
TEST_AUDIT_PATH = Path(__file__).resolve().parents[2] / 'configs/gsm8k_test_audit.json'


def test_exclusion_indices(audit):
    external = audit['external_review']
    indices = (external['removed_indices'] + external['revised_indices']
               + [int(index) for index in audit['additional_exclusions']])
    if (len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < audit['test_row_count'] for index in indices)):
        raise ValueError('invalid or duplicate test exclusion index')
    return set(indices)


def read_test_audit(dataset, revision):
    audit = json.loads(TEST_AUDIT_PATH.read_text())
    if (audit['dataset'], audit['revision']) != (dataset, revision):
        raise ValueError('test audit was prepared for a different dataset revision')
    return audit


def require_audited_test(manifest):
    source = manifest['source']
    audit = read_test_audit(source['repo_id'], source['revision'])
    if manifest['profile'] != 'smoke' and manifest.get('test_audit') != audit:
        raise ValueError('prepare a new split with the current GSM8K test audit before evaluation')
    excluded_ids = {f'test:{index}' for index in test_exclusion_indices(audit)}
    if excluded_ids.intersection(row['sample_id'] for row in manifest['samples']['test']):
        raise ValueError('frozen test contains an excluded GSM8K question')


def question_key(question):
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def parse_sample(row, source_split, index):
    if source_split not in {"train", "test"} or type(index) is not int or index < 0:
        raise ValueError("invalid source identity")
    question, answer = row.get("question"), row.get("answer")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
        raise ValueError("question/answer must be nonempty strings")
    if answer.count("####") != 1:
        raise ValueError("expected exactly one GSM8K final-answer delimiter")
    reasoning, gold = answer.split("####")
    gold = gold.strip()
    if normalize_number(gold) is None:
        raise ValueError("invalid GSM8K numeric gold")
    # Remove dataset calculator annotations but retain the written reasoning.
    reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
    prompt = [{"role": "system", "content": INSTRUCTION},
              {"role": "user", "content": question.strip()}]
    return {"sample_id": f"{source_split}:{index}", "source_split": source_split,
            "source_index": index, "question_hash": question_key(question),
            "prompt": prompt, "ground_truth": gold,
            "completion": [{"role": "assistant", "content":
                            f"{reasoning}\n<answer>{gold}</answer>".strip()}]}


def grpo_record(row):
    # Never give GRPO the SFT demonstration; gold is metadata for reward callbacks only.
    return {k: row[k] for k in ("sample_id", "prompt", "ground_truth")}


def make_splits(train, test, sizes, seed=42, *, excluded_test_indices=()):
    """Input is ordered raw rows. First occurrence wins, official test has priority.

    Duplicate questions in train/test are removed from train, even if the test row
    is not sampled. Fail on insufficient unique data; never silently change sizes.
    """
    names = ("sft", "grpo", "validation", "test")
    if set(sizes) != set(names) or any(type(v) is not int or v < 0 for v in sizes.values()):
        raise ValueError("sizes must specify four nonnegative integer counts")
    seen, pools, dropped = set(), {}, {"train": 0, "test": 0}
    excluded = set(excluded_test_indices)
    for split, rows in (("test", test), ("train", train)):
        pool = []
        for i, row in enumerate(rows):
            parsed = parse_sample(row, split, i)
            key = parsed["question_hash"]
            if key in seen:
                dropped[split] += 1
                continue
            seen.add(key)
            if split == 'test' and i in excluded:
                continue
            pool.append(parsed)
        # Independent streams keep test selection independent of train sizes.
        random.Random(f"{seed}:{split}").shuffle(pool)
        pools[split] = pool
    if sum(sizes[k] for k in names[:3]) > len(pools['train']) or sizes['test'] > len(pools['test']):
        raise ValueError("insufficient unique samples for requested sizes")
    result, offset = {}, 0
    for key in names[:3]:
        result[key] = pools['train'][offset:offset + sizes[key]]
        offset += sizes[key]
    result['test'] = pools['test'][:sizes['test']]
    manifest = {"schema_version": 1, "seed": seed, "sizes": sizes,
                "dedup_policy": "normalized question; official test priority; first occurrence",
                "dropped_duplicates": dropped,
                "samples": {k: [{f: r[f] for f in ("sample_id", "question_hash")} for r in v]
                            for k, v in result.items()}}
    return result, manifest
