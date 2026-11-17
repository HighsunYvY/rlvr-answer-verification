"""Question-paired attribution of weak rewards assigned to explicit wrong answers."""
import json
import random
import re
from collections import Counter
from pathlib import Path

from .analysis import diagnose_scored_row
from .experiment import prepared_inputs, read_json, verify_inventory, write_json
from .verifier import weak_match_spans


_EQUALS_BEFORE_NUMBER = re.compile(r'=\s*(?:\\?[$£€]\s*)?\Z')


def follows_equals(completion, start):
    return bool(_EQUALS_BEFORE_NUMBER.search(completion[:start]))


def paired_prompt_interval(deltas):
    rng = random.Random(42)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
    return [draws[250], draws[9749]]


def attribute_answers(left, right, preparation_dir, split, output_dir, left_stage='E2'):
    if left_stage not in ('E2', 'E4'):
        raise ValueError('left condition must be E2 or E4 for comparison against E3')
    difference_key = f'difference_E3_minus_{left_stage}'
    directories = {left_stage: Path(left), 'E3': Path(right)}
    reports = {stage: read_json(path / 'run.json') for stage, path in directories.items()}
    reference = reports[left_stage]
    settings = reference['settings']
    generations = settings['generation']['num_return_sequences']
    if split not in ('validation', 'test') or not settings['generation']['do_sample'] or generations < 2:
        raise ValueError('attribution requires grouped sampling on a frozen evaluation split')
    inputs, _, preparation = prepared_inputs(preparation_dir, reference['profile'])
    if reference['manifest_sha256'] != preparation['split_manifest_sha256']:
        raise ValueError('frozen preparation differs from the evaluated manifest')
    refs = {row['sample_id']: row for row in inputs[split]}
    expected = [(q, member) for q in refs for member in range(generations)]
    policy = reference['policy_identity']
    if not policy['sft_identity'] or policy['training_contract'] != reports['E3']['policy_identity']['training_contract']:
        raise ValueError('attribution requires a common SFT parent and paired training contract')

    questions, cases, summaries = {}, [], {}
    for key, row in refs.items():
        question = row['prompt'][1]['content']
        gold = row['ground_truth']
        questions[key] = {'sample_id': key, 'question': question, 'ground_truth': gold,
                          'gold_in_question': bool(list(weak_match_spans(question, gold))),
                          left_stage: 0, 'E3': 0, f'{left_stage}_after_equals': 0, 'E3_after_equals': 0}
    for stage, path in directories.items():
        report = reports[stage]
        if (report['status'] != 'passed' or report['mode'] != 'run'
                or report['stage'] != f'{stage}_eval' or report['settings']['split'] != split
                or report['n'] != len(expected)
                or any(report[key] != reference[key] for key in (
                    'settings', 'profile', 'manifest_sha256', 'template_sha256', 'model_files',
                    'resolved_generation'))
                or report['policy_identity']['sft_identity'] != policy['sft_identity']):
            raise ValueError(f'{stage} is not a complete, matched evaluation for attribution')
        verify_inventory(path, {'scored.jsonl': report['artifacts']['scored.jsonl']})
        totals, patterns = Counter(), Counter()
        with (path / 'scored.jsonl').open() as stream:
            for line, (key, member) in zip(stream, expected, strict=True):
                row = json.loads(line)
                if (row['source_sample_id'], row['group_member'], row['sample_id']) != (key, member, f'{key}#g{member}'):
                    raise ValueError(f'{stage} sample identity mismatch for {key}#{member}')
                gold = refs[key]['ground_truth']
                row = diagnose_scored_row(row, gold)
                correct = row['explicit_final_correct']
                totals['explicit_final_correct'] += correct is True
                totals['explicit_final_wrong'] += correct is False
                totals['explicit_final_undetermined'] += correct is None
                if correct is not False or not row['weak_positive']:
                    continue
                completion = row['completion']
                tag_start = completion.rfind('<answer>') if row['format_compliant'] else None
                matches = []
                for start, end in weak_match_spans(completion, gold):
                    matches.append({'start': start, 'end': end, 'text': completion[start:end],
                                    'after_equals': follows_equals(completion, start),
                                    'before_terminal_tag': None if tag_start is None else end <= tag_start})
                after_equals = any(m['after_equals'] for m in matches)
                in_question = questions[key]['gold_in_question']
                pattern = ('gold_in_question' if in_question else 'gold_absent_from_question') + (
                    '_matched_after_equals' if after_equals else '_no_match_after_equals')
                patterns[pattern] += 1
                totals['weak_rewarded_wrong'] += 1
                totals['format_compliant_weak_rewarded_wrong'] += row['format_compliant']
                totals['weak_rewarded_wrong_after_equals'] += after_equals
                totals['weak_rewarded_wrong_gold_in_question'] += in_question
                questions[key][stage] += 1
                questions[key][f'{stage}_after_equals'] += after_equals
                cases.append({'stage': stage, 'source_sample_id': key, 'sample_id': row['sample_id'],
                              'explicit_final_answer': row['explicit_final_answer'], 'gold': gold,
                              'pattern': pattern, 'matches': matches, 'completion': completion})
        summaries[stage] = {'completions': len(expected),
                            'affected_questions': sum(row[stage] > 0 for row in questions.values()),
                            'counts': dict(totals), 'patterns': dict(patterns)}

    for row in questions.values():
        row[difference_key] = row['E3'] - row[left_stage]
    deltas = [questions[q][difference_key] / generations for q in sorted(questions)]
    equals_deltas = [(questions[q]['E3_after_equals'] - questions[q][f'{left_stage}_after_equals']) / generations
                     for q in sorted(questions)]
    summary = {'split': split, 'questions': len(questions), 'completions_per_question': generations,
               'training_seed': policy['training_contract']['settings']['seed'],
               'gold_in_question_count': sum(row['gold_in_question'] for row in questions.values()),
               'conditions': summaries, 'question_changes': dict(Counter(
                   'increase' if d > 0 else 'decrease' if d < 0 else 'unchanged' for d in deltas)),
               difference_key: sum(deltas) / len(deltas),
               'paired_prompt_bootstrap_ci95': paired_prompt_interval(deltas),
               'after_equals': {difference_key: sum(equals_deltas) / len(equals_deltas),
                                'paired_prompt_bootstrap_ci95': paired_prompt_interval(equals_deltas)},
               'bootstrap_repetitions': 10000, 'bootstrap_seed': 42,
               'source_runs': {stage: str(path.resolve()) for stage, path in directories.items()}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for filename, rows in (('cases.jsonl', cases), ('questions.jsonl', questions.values())):
        with (out / filename).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'metrics.json', summary)
    return summary
