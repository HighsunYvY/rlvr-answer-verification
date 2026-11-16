"""Offline scoring of generated completions (never prompts concatenated with output)."""
import re
from collections import Counter

from .verifier import (extract_final_answer, format_compliant, normalize_number,
                       strict_verifier, weak_verifier)


EXPLICIT_FINAL_VERSION = 1
_ANSWER_MARKER = re.compile(
    r'(?P<tag><answer>(?P<tag_value>[^<>]*)</answer>)'
    r'|(?P<boxed>\\boxed\s*\{(?P<boxed_value>[^{}]*)\})'
    r'|(?P<cue>(?P<bold>\*\*)?\b(?:the\s+)?(?:final\s+)?answer'
    r'(?:\s+is\b\s*:?|\s*:)(?P<bold_end>\*\*)?|^[ \t]*####[ \t]*)',
    re.IGNORECASE | re.MULTILINE,
)
_TAG_START = re.compile(r'</?\s*answer\b', re.IGNORECASE)
_BOX_START = re.compile(r'\\boxed\b', re.IGNORECASE)
_ALTERNATIVE = re.compile(r'\s*(?:[,;]\s*|(?:or|and)\s+)[+\-−]?(?:[0-9]|\.[0-9])', re.IGNORECASE)


def extract_explicit_final(completion):
    """Return (answer, source, reason) using explicit markers or a wholly numeric response."""
    markers = list(_ANSWER_MARKER.finditer(completion))
    if (len(_TAG_START.findall(completion)) != 2 * sum(m['tag'] is not None for m in markers)
            or len(_BOX_START.findall(completion)) != sum(m['boxed'] is not None for m in markers)):
        return None, None, 'malformed_marker'
    if not markers:
        bare = completion.strip().removesuffix('.').rstrip()
        if normalize_number(bare) is not None:
            return bare, 'bare_number', None
        return None, None, 'missing_explicit_answer'

    candidates = []
    for i, marker in enumerate(markers):
        if marker['cue'] is not None:
            following = markers[i + 1] if i + 1 < len(markers) else None
            end = following.start() if following else len(completion)
            value = completion[marker.end():end].strip()
            # A label may introduce a tagged or boxed answer directly.
            if not value and following and following['cue'] is None:
                continue
            if marker['bold'] and not marker['bold_end']:
                closing = value.removesuffix('.')
                if closing.endswith('**'):
                    value = closing[:-2].strip()
                else:
                    return None, None, 'malformed_marker'
            if value.startswith('**'):
                wrapped = value.removesuffix('.')
                if wrapped.endswith('**'):
                    value = wrapped[2:-2].strip()
            value = value.removesuffix('.').rstrip()
            source = 'cue'
        elif marker['tag'] is not None:
            value, source = marker['tag_value'].strip(), 'tag'
        else:
            value, source = marker['boxed_value'].strip(), 'boxed'
        if source != 'cue' and _ALTERNATIVE.match(completion, marker.end()):
            return None, None, 'conflicting_explicit_answers'
        number = normalize_number(value)
        if number is None:
            return None, None, 'unsupported_explicit_answer'
        candidates.append((number, value, source))

    if len({number for number, _, _ in candidates}) != 1:
        return None, None, 'conflicting_explicit_answers'
    _, answer, source = candidates[-1]
    return answer, source, None


def diagnose_final_answer(row):
    """Add independent evaluation fields to a legacy score without changing its rewards."""
    if row.get('truncated') is True:
        answer, source, reason = None, None, 'truncated'
    else:
        answer, source, reason = extract_explicit_final(row['completion'])
    correct = None if answer is None else normalize_number(answer) == normalize_number(row['ground_truth'])
    disagreement = None
    if row['weak_false_positive']:
        disagreement = {True: 'correct_explicit_final', False: 'wrong_explicit_final',
                        None: 'undetermined_explicit_final'}[correct]
    return {'explicit_final_answer': answer, 'explicit_final_source': source,
            'explicit_final_correct': correct, 'explicit_final_uncertainty': reason,
            'weak_disagreement': disagreement}


def explicit_final_metrics(rows):
    rows = list(rows)
    counts = Counter(r['explicit_final_correct'] for r in rows)
    reasons = Counter(r['explicit_final_uncertainty'] for r in rows if r['explicit_final_correct'] is None)
    disagreements = Counter(r['weak_disagreement'] for r in rows if r['weak_disagreement'] is not None)
    n = sum(counts.values())
    covered = counts[True] + counts[False]
    wrong_rewarded = sum(r['weak_positive'] and r['explicit_final_correct'] is False for r in rows)
    return {'version': EXPLICIT_FINAL_VERSION, 'correct': counts[True], 'wrong': counts[False],
            'undetermined': counts[None], 'coverage': covered / n if n else None,
            'accuracy_given_determined': counts[True] / covered if covered else None,
            'uncertainty_counts': dict(reasons), 'weak_disagreement_counts': dict(disagreements),
            'weak_rewarded_wrong_final_count': wrong_rewarded,
            'weak_rewarded_wrong_final_share': wrong_rewarded / n if n else None}


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
