"""Exact rational numeric comparison; no float tolerance or expression evaluation."""
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

# Thousands groups are validated before commas are removed.
_NUMBER = r"[+\-−]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?"
_NUMERIC = re.compile(rf"(?:{_NUMBER})(?:\s*/\s*(?:{_NUMBER}))?")
# Boundaries prevent extracting 42 from 142, -42, 42.5, 1,042, 42/7 or identifiers.
_TOKEN = re.compile(rf"(?<![\w.,/+\-−]){_NUMERIC.pattern}(?![\w.,/]|[+\-−][0-9])")


def normalize_number(value):
    """Return Fraction or None. Supports signed decimal, grouping, exponent, fraction.

    Reject units, currency, percentages, NaN, expressions and excessive exponents.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) > 128 or not _NUMERIC.fullmatch(value):
        return None
    try:
        parts = value.replace('−', '-').replace(',', '').split('/')
        decimals = [Decimal(p.strip()) for p in parts]
        if any(not d.is_finite() or abs(d.as_tuple().exponent) > 100 for d in decimals):
            return None
        numbers = [Fraction(d) for d in decimals]
        return numbers[0] if len(numbers) == 1 else numbers[0] / numbers[1]
    except (InvalidOperation, ZeroDivisionError, ValueError, OverflowError):
        return None


def answer_tags(text):
    """Complete, nonnested exact-case tags, with nonempty plain-text content."""
    if not isinstance(text, str):
        return []
    # Reject a tag if it is enclosed in an unmatched/nested answer tag.
    result, depth, nested, start = [], 0, False, 0
    for match in re.finditer(r'</?answer>', text):
        if match.group() == '<answer>':
            if depth:
                nested = True
            else:
                nested, start = False, match.end()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                content = text[start:match.start()]
                if not nested and content.strip() and '<' not in content and '>' not in content:
                    result.append((content.strip(), match.end()))
    return result


def extract_final_answer(text):
    """Last syntactically legal tag, even when its content is not a number.

    A later malformed/unclosed tag does not erase an earlier complete tag.
    Never fall back to an earlier numerically correct tag.
    """
    tags = answer_tags(text)
    return tags[-1][0] if tags else None


def format_compliant(text):
    """One numeric answer tag at the end; reasoning may precede it."""
    if not isinstance(text, str):
        return False
    tags = answer_tags(text)
    return (len(tags) == 1 and text.count('<answer>') == 1
            and text.count('</answer>') == 1 and not text[tags[0][1]:].strip()
            and normalize_number(tags[0][0]) is not None)


def strict_verifier(text, ground_truth):
    gold = normalize_number(ground_truth)
    return gold is not None and normalize_number(extract_final_answer(text)) == gold


def weak_match_spans(text, ground_truth):
    """Yield original character spans for the weak verifier's numeric matches."""
    gold = normalize_number(ground_truth)
    if gold is None or not isinstance(text, str):
        return
    # A sentence-ending period/comma is punctuation, not part of the value.
    # Substituting a space for the punctuation keeps the offsets unchanged.
    scanned = re.sub(r'([0-9])[.,](?=\s|$|<)', r'\1 ', text)
    for match in _TOKEN.finditer(scanned):
        if normalize_number(match.group()) == gold:
            yield match.span()


def weak_verifier(text, ground_truth):
    """Intentionally flawed baseline: accepts a gold-valued numeric token anywhere."""
    return next(weak_match_spans(text, ground_truth), None) is not None
