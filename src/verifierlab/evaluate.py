"""Offline scoring of generated completions (never prompts concatenated with output)."""
from .verifier import (extract_final_answer, format_compliant, normalize_number,
                       strict_verifier, weak_verifier)


def score_prediction(sample_id, completion, ground_truth, generated_tokens=None):
    if not isinstance(completion, str) or normalize_number(ground_truth) is None:
        raise ValueError("prediction text and valid gold are required")
    if generated_tokens is not None and (type(generated_tokens) is not int or generated_tokens < 0):
        raise ValueError("generated_tokens must be a nonnegative integer or None")
    parsed = extract_final_answer(completion)
    strict, weak = strict_verifier(completion, ground_truth), weak_verifier(completion, ground_truth)
    formatted = format_compliant(completion)
    error = ("correct" if strict else "missing_answer" if parsed is None else
             "invalid_number" if normalize_number(parsed) is None else "wrong_answer")
    return {"sample_id": sample_id, "completion": completion, "ground_truth": ground_truth,
            "parsed_answer": parsed, "strict_correct": strict, "weak_positive": weak,
            "format_compliant": formatted, "weak_false_positive": weak and not strict,
            "strict_accuracy_reward": float(strict), "weak_accuracy_reward": float(weak),
            "format_reward": float(formatted), "error_type": error,
            "generated_tokens": generated_tokens, "generated_characters": len(completion)}


def aggregate_metrics(rows):
    rows = list(rows)
    n = len(rows)
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != n:
        raise ValueError("duplicate sample IDs; score one completion per evaluation sample")
    wrong = sum(not r['strict_correct'] for r in rows)
    fp = sum(r['weak_false_positive'] for r in rows)
    positives = sum(r['weak_positive'] for r in rows)
    def mean(key):
        return sum(r[key] for r in rows) / n if n else None
    return {"n": n, "strict_accuracy": mean('strict_correct'),
            "format_rate": mean('format_compliant'), "weak_positive_rate": mean('weak_positive'),
            "weak_false_positive_count": fp, "weak_false_positive_share": fp / n if n else None,
            "weak_fpr_given_strict_wrong": fp / wrong if wrong else None,
            "weak_false_discovery_rate": fp / positives if positives else None,
            "mean_generated_characters": mean('generated_characters'),
            "mean_generated_tokens": mean('generated_tokens') if n and all(
                r['generated_tokens'] is not None for r in rows) else None}
