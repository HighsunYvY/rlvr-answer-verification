"""Project contracts and target regressions; run only within authorized scope."""
from contextlib import redirect_stdout
import copy
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
from verifierlab.analysis import audit_answers, paired_difference, replay_group_rewards, wilson
from verifierlab.audit import reconcile_rows
from verifierlab.data import grpo_record, make_splits, parse_sample
from verifierlab.evaluate import (aggregate_metrics, diagnose_final_answer, explicit_final_metrics,
                                 score_prediction)
from verifierlab.experiment import Session, approve, fingerprint, inventory, verify_inventory
from verifierlab.generate import completion_record, load_frozen, run, score_frozen
from verifierlab.preflight import check_environment, main as preflight_main
from verifierlab.rewards import format_reward, strict_accuracy_reward, weak_accuracy_reward
from verifierlab.rollouts import group_signal
from verifierlab.sft import inspect_labels, tokenize_completion
from verifierlab.snapshots import file_identity, freeze_data, prepare, validate_sources, verify_snapshot
from verifierlab.training import compare_probe, forward_probe, initial_validation_context, trace_reward
from verifierlab.utils import load_config, validate_grpo
from verifierlab.verifier import extract_final_answer, format_compliant, normalize_number, strict_verifier, weak_verifier


class AuditTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 100

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if messages != [{'role': 'user', 'content': 'question'}]:
            raise ValueError('unexpected prompt payload')
        return [8, 9]

    def decode(self, ids, skip_special_tokens):
        return ''.join({1: '<answer>42</answer>', 2: '42 then <answer>7</answer>',
                        99: '' if skip_special_tokens else '<eos>'}[i] for i in ids)


def fixture():
    frozen, raw, scored = [], [], []
    for index, token in enumerate((1, 2)):
        gold = {'sample_id': f'test:{index}', 'ground_truth': '42',
                'prompt': [{'role': 'user', 'content': 'question'}]}
        prediction = completion_record(gold['sample_id'], [token, 99], AuditTokenizer(), [99], 4, 2)
        prediction['generation_seconds'] = 0.1
        frozen.append(gold)
        raw.append({'sample_id': gold['sample_id'], 'prompt_tokens': 2,
                    'generated_token_ids': [token, 99]})
        scored.append(score_frozen(prediction, gold))
    return raw, scored, frozen


class AuditTests(unittest.TestCase):
    def test_recomputed_verifier_metrics(self):
        metrics = reconcile_rows(*fixture(), AuditTokenizer(), [99], 4)
        self.assertEqual(metrics['n'], 2)
        self.assertEqual(metrics['strict_accuracy'], 0.5)
        self.assertEqual(metrics['weak_false_positive_count'], 1)
        self.assertEqual(metrics['truncation_rate'], 0)

    def test_rejects_tampered_scored_fields(self):
        changes = {'ground_truth': '7', 'completion': '<answer>7</answer>',
                   'strict_correct': False, 'generated_tokens': 1, 'truncated': True,
                   'termination': 'max_new_tokens', 'prompt_tokens': 7,
                   'generation_seconds': float('nan')}
        for key, value in changes.items():
            raw, scored, frozen = fixture()
            scored[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reconcile_rows(raw, scored, frozen, AuditTokenizer(), [99], 4)

    def test_rejects_missing_duplicate_reordered_or_extra_rows(self):
        for mutation in ('missing', 'duplicate', 'reorder', 'extra'):
            raw, scored, frozen = fixture()
            if mutation == 'missing':
                raw.pop()
            elif mutation == 'duplicate':
                raw[1] = copy.deepcopy(raw[0])
            elif mutation == 'reorder':
                raw.reverse()
            else:
                scored.append(copy.deepcopy(scored[0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reconcile_rows(raw, scored, frozen, AuditTokenizer(), [99], 4)

    def test_rejects_bad_token_evidence(self):
        for ids in ([True, 99], [-1, 99], [100, 99], [], [1], [1, 99, 2]):
            raw, scored, frozen = fixture()
            raw[0]['generated_token_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                reconcile_rows(raw, scored, frozen, AuditTokenizer(), [99], 4)


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_module(self, module, *args):
        return subprocess.run(
            [sys.executable, '-m', f'verifierlab.{module}', *args],
            cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
            text=True, capture_output=True, timeout=20,
        )

    def test_help(self):
        commands = [('runner',), ('snapshots',), ('preflight',), ('generate',), ('audit',),
                    ('cli',), ('cli', 'prepare-data'), ('cli', 'preview-config'),
                    ('cli', 'summarize-results'), ('cli', 'diagnose')]
        for command in commands:
            with self.subTest(command=command):
                result = self.run_module(*command, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_paired_plan_profiles_and_output_isolation(self):
        for stage in ('grpo_strict', 'grpo_weak'):
            for profile in ('smoke', 'pilot', 'full'):
                with self.subTest(stage=stage, profile=profile):
                    result = self.run_module('cli', 'preview-config', '--config', f'configs/{stage}.yaml',
                                             '--profile', profile)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertFalse(plan['execution_available'])
                    self.assertEqual(plan['mode'], 'plan_only')
                    self.assertEqual(plan['config']['trainer']['output_dir'],
                                     f'outputs/{stage}/{profile}/RUN_ID')
                    self.assertEqual(plan['batch_check']['generation_completion_batch'], 8)

    def test_preview_and_diagnostics(self):
        for config in ('sft', 'evaluation'):
            result = self.run_module('cli', 'preview-config', '--config', f'configs/{config}.yaml')
            self.assertEqual(result.returncode, 0, result.stderr)
            preview = json.loads(result.stdout)
            self.assertFalse(preview['execution_available'])
            self.assertEqual(preview['mode'], 'plan_only')
        result = self.run_module('cli', 'diagnose')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['cuda'], 'not probed')

    def test_local_split_scoring_and_overwrite_rejection(self):
        with tempfile.TemporaryDirectory(prefix='verifierlab-test-') as directory:
            tmp = Path(directory)
            for split, count in [('train', 24), ('test', 8)]:
                rows = [{'question': f'{split} question {i}', 'answer': 'Reasoning #### 42'}
                        for i in range(count)]
                (tmp / f'{split}.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            args = ('--train-jsonl', str(tmp / 'train.jsonl'), '--test-jsonl',
                    str(tmp / 'test.jsonl'), '--source-revision', 'synthetic-local',
                    '--output-dir', str(tmp / 'split'))
            result = self.run_module('cli', 'prepare-data', *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = tmp / 'split' / 'manifest.json'
            before = manifest_path.read_bytes()
            manifest = json.loads(before)
            self.assertEqual(manifest['sizes'], dict(sft=8, grpo=8, validation=4, test=4))
            self.assertEqual(manifest['source_revision'], 'synthetic-local')
            for split, count in manifest['sizes'].items():
                self.assertEqual(len((tmp / 'split' / f'{split}.jsonl').read_text().splitlines()),
                                 count)
            self.assertNotEqual(self.run_module('cli', 'prepare-data', *args).returncode, 0)
            self.assertEqual(manifest_path.read_bytes(), before)

            prediction = tmp / 'prediction.jsonl'
            prediction.write_text(json.dumps({
                'sample_id': 'synthetic:0', 'completion': '42 then <answer>7</answer>',
                'ground_truth': '42', 'generated_tokens': 10,
            }) + '\n', encoding='utf-8')
            args = ('--predictions', str(prediction), '--output-dir', str(tmp / 'scored'))
            result = self.run_module('cli', 'summarize-results', *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            scored_path = tmp / 'scored' / 'scored.jsonl'
            scored_before = scored_path.read_bytes()
            scored = json.loads(scored_before)
            self.assertTrue(scored['weak_false_positive'])
            self.assertFalse(scored['strict_correct'])
            metrics = json.loads((tmp / 'scored' / 'metrics.json').read_text())
            self.assertEqual(metrics['n'], 1)
            self.assertEqual(metrics['weak_fpr_given_strict_wrong'], 1.0)
            self.assertNotEqual(self.run_module('cli', 'summarize-results', *args).returncode, 0)
            self.assertEqual(scored_path.read_bytes(), scored_before)


class ConfigTests(unittest.TestCase):
    def test_paired_controls_and_profiles(self):
        strict = load_config(ROOT / 'configs/grpo_strict.yaml')
        weak = load_config(ROOT / 'configs/grpo_weak.yaml')
        strict.pop('stage')
        weak.pop('stage')
        strict.pop('correctness_verifier')
        weak.pop('correctness_verifier')
        self.assertEqual(strict, weak)
        for profile in strict['profiles'].values():
            config = copy.deepcopy(strict)
            config['trainer'].update(profile)
            self.assertEqual(validate_grpo(config)['prompts_per_generation_batch'], 2)

    def test_invalid_batches(self):
        c = load_config(ROOT / 'configs/grpo_strict.yaml')
        c['trainer']['num_generations'] = 3
        with self.assertRaises(ValueError):
            validate_grpo(c)
        c['trainer']['num_generations'] = 1
        with self.assertRaises(ValueError):
            validate_grpo(c)


def raw(prefix, n):
    return [{'question': f'{prefix} problem {i}?', 'answer': 'Compute <<1+1=2>>2. #### 2'} for i in range(n)]


class DataTests(unittest.TestCase):
    def test_parse_and_prompt_isolation(self):
        row = parse_sample({'question': 'What is six times seven?', 'answer': 'Six times seven. #### 42'}, 'train', 0)
        self.assertEqual(row['ground_truth'], '42')
        self.assertNotIn('42', str(row['prompt']))
        self.assertEqual(row['completion'][0]['content'], 'Six times seven.\n<answer>42</answer>')
        self.assertNotIn('completion', grpo_record(row))
        self.assertNotIn('<<', parse_sample(raw('a', 1)[0], 'train', 0)['completion'][0]['content'])

    def test_reproducibility_disjointness(self):
        sizes = dict(sft=5, grpo=5, validation=3, test=4)
        a, m = make_splits(raw('train', 30), raw('test', 10), sizes)
        self.assertEqual((a, m), make_splits(raw('train', 30), raw('test', 10), sizes))
        self.assertNotEqual(m, make_splits(raw('train', 30), raw('test', 10), sizes, 43)[1])
        ids = [r['sample_id'] for rows in a.values() for r in rows]
        hashes = [r['question_hash'] for rows in a.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual({k: len(v) for k, v in a.items()}, sizes)

    def test_duplicate_questions_removed_across_sources(self):
        train, test = raw('train', 20), raw('test', 10)
        train += [test[0], {'question': '  TEST   problem 0? ', 'answer': 'Same #### 2'}]
        _, manifest = make_splits(train, test, dict(sft=5, grpo=5, validation=3, test=4))
        self.assertEqual(manifest['dropped_duplicates']['train'], 2)

    def test_invalid_and_short_inputs(self):
        for answer in ['', '42', 'x #### nan', 'x #### 2 #### 3']:
            with self.assertRaises(ValueError):
                parse_sample({'question': 'Q', 'answer': answer}, 'train', 0)
        with self.assertRaises(ValueError):
            make_splits([], [], dict(sft=1, grpo=0, validation=0, test=0))


class EvaluationTests(unittest.TestCase):
    def test_denominators(self):
        rows = [score_prediction('a', '<answer>42</answer>', '42', 10),
                score_prediction('b', '42 then <answer>7</answer>', '42', 20),
                score_prediction('c', '', '42', 0)]
        result = aggregate_metrics(rows)
        self.assertEqual(result['strict_accuracy'], 1/3)
        self.assertEqual(result['weak_false_positive_share'], 1/3)
        self.assertEqual(result['weak_fpr_given_strict_wrong'], 1/2)
        self.assertEqual(result['weak_false_discovery_rate'], 1/2)
        self.assertEqual(result['mean_generated_tokens'], 10)
        self.assertEqual(rows[2]['error_type'], 'missing_answer')

    def test_empty_missing_lengths_duplicate_ids(self):
        self.assertIsNone(aggregate_metrics([])['strict_accuracy'])
        row = score_prediction('a', '<answer>2</answer>', '2')
        self.assertIsNone(aggregate_metrics([row])['mean_generated_tokens'])
        self.assertIsNone(aggregate_metrics([row])['weak_fpr_given_strict_wrong'])
        with self.assertRaises(ValueError):
            aggregate_metrics([row, row])
        with self.assertRaises(ValueError):
            score_prediction('b', '', 'invalid')

    def test_explicit_answer_keeps_conflicts_and_truncation_unresolved(self):
        cases = [
            ('The answer is 42.', False, True, None),
            ('**Final answer:** 84/2.', False, True, None),
            ('The answer is 42.5.', False, False, None),
            ('The answer is 4.2e1.', False, True, None),
            ('Answer: 42. <answer>84/2</answer>', False, True, None),
            (r'Thus the total is \boxed{42} items.', False, True, None),
            ('Considered 42. <answer>7</answer>', False, False, None),
            ('The subtotal is 42.', False, None, 'missing_explicit_answer'),
            ('The answer is 42 or 7.', False, None, 'unsupported_explicit_answer'),
            ('The answer is 42. <answer>7</answer>', False, None, 'conflicting_explicit_answers'),
            (r'\boxed{7} or 42', False, None, 'conflicting_explicit_answers'),
            ('<answer>7</answer><answer>42', False, None, 'malformed_marker'),
            ('<answer>42</answer>', True, None, 'truncated'),
        ]
        for completion, truncated, correct, reason in cases:
            with self.subTest(completion=completion, truncated=truncated):
                row = {**score_prediction('a', completion, '42'), 'truncated': truncated}
                diagnosis = diagnose_final_answer(row)
                self.assertIs(diagnosis['explicit_final_correct'], correct)
                self.assertEqual(diagnosis['explicit_final_uncertainty'], reason)

    def test_replay_grouped_legacy_rows_against_frozen_gold(self):
        references = {'validation': [{'sample_id': 'train:1', 'ground_truth': '42'}]}
        texts = ('Final answer: 42', 'Considered 42. <answer>7</answer>', 'The subtotal is 42.')
        rows = [{**score_prediction(f'train:1#g{i}', text, '42', 20),
                 'source_sample_id': 'train:1', 'group_member': i, 'truncated': False}
                for i, text in enumerate(texts)]
        with tempfile.TemporaryDirectory() as tmp, patch('verifierlab.analysis.prepared_inputs',
                return_value=(references, {}, {'split_manifest_sha256': 'fixture'})):
            source = Path(tmp) / 'scored.jsonl'
            output = Path(tmp) / 'diagnosis'
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            summary = audit_answers(source, 'fixture', 'full', 'validation', 3, output)
            metrics = summary['metrics']['explicit_final']
            self.assertEqual(metrics['accuracy_given_determined'], 0.5)
            self.assertEqual(metrics['coverage'], 2 / 3)
            self.assertEqual(metrics['weak_disagreement_counts'], {
                'correct_explicit_final': 1, 'wrong_explicit_final': 1, 'undetermined_explicit_final': 1})
            self.assertEqual(metrics['weak_rewarded_wrong_final_count'], 1)
            diagnosed = [json.loads(line) for line in (output / 'diagnosed.jsonl').read_text().splitlines()]
            self.assertEqual([r['weak_accuracy_reward'] for r in diagnosed], [1.0, 1.0, 1.0])
            self.assertIsNone(explicit_final_metrics([])['coverage'])
            for invalid_rows in (rows[:2], [*rows[:2], rows[1]],
                                 [{**rows[0], 'ground_truth': '7'}, *rows[1:]]):
                source.write_text(''.join(json.dumps(r) + '\n' for r in invalid_rows))
                with self.assertRaises(ValueError):
                    audit_answers(source, 'fixture', 'full', 'validation', 3, Path(tmp) / 'invalid')

    def test_group_reward_replay_gains_and_loses_signal(self):
        rows = []
        rewarded_wrong = 'Considered 42. <answer>7</answer>'
        gained_signal = [rewarded_wrong] + ['<answer>7</answer>'] * 3
        lost_signal = ['<answer>42</answer>'] + [rewarded_wrong] * 3
        for q, texts in enumerate((gained_signal, lost_signal)):
            for i, completion in enumerate(texts):
                row = score_prediction(f'train:{q}#g{i}', completion, '42')
                rows.append({**row, **diagnose_final_answer(row),
                             'source_sample_id': f'train:{q}', 'group_member': i})
        summary, groups = replay_group_rewards(rows, 'full', 4)
        self.assertEqual(summary['group_signal_counts']['gained_variable_total'], 1)
        self.assertEqual(summary['group_signal_counts']['lost_variable_total'], 1)
        self.assertEqual(summary['wrong_explicit_final_disagreement_transitions'], {
            'zero->positive': 1, 'negative->zero': 3})
        self.assertEqual(summary['wrong_explicit_final_becomes_positive'], 1)
        self.assertAlmostEqual(groups[0]['weak']['sample_std'], .5)
        self.assertAlmostEqual(rows[0]['grpo_replay']['weak']['advantage'], .75 / .5001)
        self.assertEqual(rows[5]['grpo_replay']['weak']['advantage'], 0)


class ExperimentTests(unittest.TestCase):
    def test_identity_is_order_stable_and_changes_with_controls(self):
        self.assertEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'b': 2, 'a': 1}))
        self.assertNotEqual(fingerprint({'beta': .04}), fingerprint({'beta': 0}))

    def test_inventory_detects_tampering_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x'
            path.write_text('one')
            files = inventory(tmp)
            verify_inventory(tmp, files)
            path.write_text('two')
            with self.assertRaises(ValueError):
                verify_inventory(tmp, files)
            with self.assertRaises(ValueError):
                verify_inventory(tmp, {'../escape': '0' * 64})

    def test_pilot_requires_reviewed_matching_real_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = Session(Path(tmp)/'new', 'E2', 'pilot', 'run', '.', '.')
            current.report['signature'] = 'expected'
            with self.assertRaises(ValueError):
                current.admit(None)
            root = Path(tmp)/'old'
            root.mkdir()
            run = {'status':'failed', 'mode':'smoke', 'trainer_validated':False, 'artifacts':{}}
            (root/'run.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                approve(root, Path(tmp)/'proof.json', 'reviewed')
            run.update(status='passed', trainer_validated=True, signature='different')
            (root/'run.json').write_text(json.dumps(run))
            with self.assertRaises((ValueError, KeyError)):
                approve(root, Path(tmp)/'proof.json', 'reviewed')

    def test_weak_signal_keeps_negative_groups(self):
        rows = [score_prediction(str(i), '<answer>7</answer>', '42') for i in range(4)]
        signal = group_signal(rows)
        self.assertFalse(signal['variable_total'])
        rows[0] = score_prediction('0', '42 then <answer>7</answer>', '42')
        self.assertTrue(group_signal(rows)['variable_accuracy'])

    def test_grpo_forwarding_and_group_order(self):
        rows = [{'sample_id': str(i), 'ground_truth':'42', 'prompt': [], 'partition':'train'} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            reward, calls = trace_reward(rows, Path(tmp), 'weak', 2)
            result = reward(completions=['42 then <answer>7</answer>']*2,
                            ground_truth=['42']*2, sample_id=['0']*2, prompts=[[],[]],
                            completion_ids=[[1],[1]], partition=['train']*2)
            self.assertEqual(result, [1.,1.])
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                reward(completions=['x']*2, ground_truth=['42']*2, sample_id=['0','1'],
                       prompts=[[],[]], completion_ids=[[1],[1]], partition=['train']*2)
            with self.assertRaises(ValueError):
                reward(completions=['x']*2, ground_truth=['7']*2, sample_id=['0']*2,
                       prompts=[[],[]], completion_ids=[[1],[1]], partition=['train']*2)

    def test_statistics_identical_and_paired_changes(self):
        same = paired_difference([True,False], [True,False], repetitions=100)
        self.assertEqual(same['difference_right_minus_left'], 0)
        self.assertEqual(same['exact_mcnemar_p_two_sided'], 1)
        changed = paired_difference([False]*8, [True]*8, repetitions=100)
        self.assertEqual(changed['paired_bootstrap_ci95'], [1,1])
        self.assertLess(changed['exact_mcnemar_p_two_sided'], .05)
        ci = wilson(0,4)
        self.assertAlmostEqual(ci[0], 0)
        self.assertGreater(ci[1], 0)

    def test_failed_checkpoint_is_not_promoted(self):
        from verifierlab.experiment import checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'checkpoint.json').write_text(json.dumps({'stage':'E1'}))
            (root/'run.json').write_text(json.dumps({'stage':'E1','status':'failed',
                                                     'trainer_validated':False,'artifacts':{}}))
            with self.assertRaises(ValueError):
                checkpoint(root)

    def test_partial_analysis_and_gold_anchor(self):
        from unittest.mock import patch
        from verifierlab.analysis import analyze
        frozen = [{'sample_id': str(i), 'ground_truth':'42',
                   'prompt':[{'role':'system','content':'instruction'},{'role':'user','content':'question'}]} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for stage in ('E0','E1'):
                root = Path(tmp)/stage
                root.mkdir()
                data = [{**score_prediction(str(i), '<answer>42</answer>', '42', 2),
                         'truncated':False} for i in range(4)]
                (root/'scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
                report = {'stage':stage+'_eval','status':'passed','mode':'smoke','profile':'smoke',
                          'manifest_sha256':'manifest','settings':{'split':'test'},
                          'template_sha256':'template','model_files':{},
                          'policy_identity':None if stage=='E0' else {'adapter_sha256':'e1'},
                          'artifacts':inventory(root)}
                (root/'run.json').write_text(json.dumps(report))
                paths[stage] = root
            with patch('verifierlab.analysis.prepared_inputs', return_value=(
                    {'test':frozen}, {}, {'split_manifest_sha256':'manifest'})):
                summary = analyze(paths, Path(tmp)/'analysis', preparation_dir='fixture')
                self.assertEqual(summary['missing_conditions'], ['E2','E3'])
                self.assertEqual(summary['conditions']['E1']['strict_accuracy'], 1)
                self.assertEqual(summary['paired_explicit_final']['E0->E1']['jointly_determined'], 4)
                self.assertEqual(summary['paired_weak_rewarded_wrong_final']['E0->E1']['gains'], 0)
            wrong = [{'sample_id':str(i),'ground_truth':'7'} for i in range(4)]
            with patch('verifierlab.analysis.prepared_inputs', return_value=(
                    {'test':wrong}, {}, {'split_manifest_sha256':'manifest'})):
                with self.assertRaises(ValueError):
                    analyze(paths, Path(tmp)/'bad', preparation_dir='fixture')


class Decoder:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens):
        return '|'.join(str(i) for i in ids if not skip_special_tokens or i != 99)


class GenerationTests(unittest.TestCase):
    def test_eos_budget_and_preservation(self):
        row = completion_record('id', [7, 99, 0], Decoder(), [99], 4, 120)
        self.assertEqual(row['completion'], '7')
        self.assertEqual(row['raw_completion'], '7|99')
        self.assertEqual(row['generated_tokens'], 2)
        self.assertEqual(row['generated_token_ids'], [7, 99, 0])
        self.assertFalse(row['truncated'])
        self.assertEqual(row['termination'], 'eos')
        capped = completion_record('id', [7, 8], Decoder(), [99], 2, 120)
        self.assertTrue(capped['truncated'])
        self.assertEqual(capped['termination'], 'max_new_tokens')
        # EOS at the exact limit takes precedence over length truncation.
        self.assertFalse(completion_record('id', [7, 99], Decoder(), [99], 2, 120)['truncated'])
        for ids in ([], [1], [1, 2, 3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                completion_record('id', ids, Decoder(), [99], 2, 120)

    def test_rejects_nonpadding_after_eos(self):
        with self.assertRaisesRegex(ValueError, 'non-padding'):
            completion_record('id', [7, 99, 8], Decoder(), [99], 4, 120)

    def test_gold_is_joined_by_frozen_identity(self):
        prediction = {'sample_id': 'test:5', 'completion': '<answer>42</answer>',
                      'generated_tokens': 7, 'termination': 'eos'}
        frozen = {'sample_id': 'test:5', 'ground_truth': '42'}
        scored = score_frozen(prediction, frozen)
        self.assertTrue(scored['strict_correct'])
        self.assertEqual(scored['termination'], 'eos')
        self.assertNotIn('ground_truth', prediction)
        for invalid in ({**prediction, 'sample_id': 'test:6'},
                        {**prediction, 'ground_truth': '1'}):
            with self.assertRaises(ValueError):
                score_frozen(invalid, frozen)

    def test_frozen_file_corruption_and_manifest_anchor(self):
        config = {'seed': 42, 'profiles': {'smoke': dict(sft=8, grpo=8, validation=4, test=4)}}
        def rows(prefix, n):
            return [{'question': f'{prefix} {i}', 'answer': 'Reasoning #### 42'} for i in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'data'
            freeze_data(rows('train', 24), rows('test', 8), config, 'smoke', directory, {})
            digest = file_identity(directory / 'manifest.json')['sha256']
            frozen = load_frozen(directory, digest)
            self.assertEqual(len(frozen), 4)
            self.assertEqual(len(load_frozen(directory, digest, split='sft')), 8)
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                load_frozen(directory, '0' * 64)
            path = directory / 'test.jsonl'
            path.write_text(path.read_text().replace('42', '43'))
            with self.assertRaisesRegex(ValueError, 'split hash'):
                load_frozen(directory, digest)

    def test_failure_evidence_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            with patch('verifierlab.generate.code_revision', return_value={'dirty': True}):
                with self.assertRaisesRegex(ValueError, 'clean committed'):
                    run(Path(tmp) / 'missing', Path(tmp) / 'model', out)
            report = json.loads((out / 'run.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['model_generation_validated'])
            self.assertFalse(report['trainer_validated'])
            before = (out / 'run.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run(Path(tmp) / 'missing', Path(tmp) / 'model', out)
            self.assertEqual(before, (out / 'run.json').read_bytes())


class GRPOInitialValidationTests(unittest.TestCase):
    def test_evaluate_before_train_initializes_and_restores_loss_divisor(self):
        try:
            import torch
            from datasets import Dataset
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
            from trl import GRPOConfig, GRPOTrainer
        except ImportError:
            self.skipTest('requires the locked target environment')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest('requires BF16 CUDA')
        raw = Tokenizer(WordLevel({'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3,
                                   'question': 4, 'one': 5, 'two': 6, 'answer': 7},
                                  unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', padding_side='left',
                    model_input_names=['input_ids', 'attention_mask'])
        self.assertEqual(set(tokenizer(['question one'], return_tensors='pt')),
                         {'input_ids', 'attention_mask'})
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=32,
                    intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=32,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2)).to(dtype=torch.bfloat16)
        rows = Dataset.from_list([{'prompt': 'question one'}, {'prompt': 'question two'}])

        def reward(completions, **kwargs):
            return [float(len(text)) for text in completions]

        with tempfile.TemporaryDirectory() as tmp:
            trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward],
                train_dataset=rows, eval_dataset=rows,
                args=GRPOConfig(output_dir=tmp, per_device_train_batch_size=1,
                    gradient_accumulation_steps=2, per_device_eval_batch_size=2,
                    num_generations=2, max_prompt_length=8, max_completion_length=4,
                    bf16=True, report_to='none', max_steps=1, seed=42,
                    remove_unused_columns=False, use_vllm=False))
            self.assertFalse(hasattr(trainer, 'current_gradient_accumulation_steps'))
            with initial_validation_context(trainer, grpo=True) as normalization:
                self.assertEqual(normalization, {'loss_divisor': 2, 'temporary_train_state': True})
                metrics = trainer.evaluate()
            self.assertTrue(math.isfinite(metrics['eval_loss']))
            self.assertFalse(hasattr(trainer, 'current_gradient_accumulation_steps'))
            trainer.current_gradient_accumulation_steps = 3
            with initial_validation_context(trainer, grpo=True) as normalization:
                self.assertEqual(normalization, {'loss_divisor': 3, 'temporary_train_state': False})
            self.assertEqual(trainer.current_gradient_accumulation_steps, 3)
            del trainer.current_gradient_accumulation_steps
            with self.assertRaisesRegex(RuntimeError, 'interrupted validation'):
                with initial_validation_context(trainer, grpo=True):
                    raise RuntimeError('interrupted validation')
            self.assertFalse(hasattr(trainer, 'current_gradient_accumulation_steps'))


class PreflightTests(unittest.TestCase):
    def test_missing_dependencies_fail_without_gpu_import(self):
        with patch('verifierlab.preflight.importlib.metadata.version',
                   side_effect=importlib.metadata.PackageNotFoundError), patch(
                       'verifierlab.preflight.cuda_probe') as probe:
            result = check_environment(ROOT / 'pyproject.toml')
        self.assertFalse(result['passed'])
        self.assertFalse(result['model_training_validated'])
        probe.assert_not_called()

    def test_target_and_cuda_failure(self):
        versions = {'torch': '2.8.0+cu128', 'transformers': '4.57.1', 'trl': '0.24.0',
                    'peft': '0.17.1', 'datasets': '4.3.0', 'accelerate': '1.11.0'}
        with patch('verifierlab.preflight.platform.system', return_value='Linux'), patch(
            'verifierlab.preflight.sys.version_info', (3, 11)), patch(
            'verifierlab.preflight.importlib.metadata.version', side_effect=versions.get), patch(
            'verifierlab.preflight.cuda_probe', side_effect=RuntimeError('kernel failed')), patch(
            'verifierlab.preflight.importlib.import_module'):
            self.assertTrue(check_environment(ROOT / 'pyproject.toml')['passed'])
            result = check_environment(ROOT / 'pyproject.toml', True)
        self.assertFalse(result['passed'])
        self.assertIn('kernel failed', result['gpu']['error'])

    def test_cli_records_failure_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'evidence.json'
            args = ['--output', str(output)]
            revision = {'commit': '0' * 40, 'dirty': False}
            with patch('verifierlab.preflight.code_revision', return_value=revision), redirect_stdout(io.StringIO()):
                status = preflight_main(args)
            result = json.loads(output.read_text())
            self.assertEqual(status, 0 if result['passed'] else 1)
            self.assertEqual(result['code'], revision)
            self.assertEqual(len(result['pyproject_sha256']), 64)
            original = output.read_bytes()
            with self.assertRaises(SystemExit) as error:
                preflight_main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)


class RewardTests(unittest.TestCase):
    def test_trl_keyword_contract(self):
        completions = [[{'role': 'assistant', 'content': '42 then <answer>7</answer>'}],
                       '<answer>42</answer>', None, '', {'content': '42'}]
        kwargs = dict(prompts=['unused'] * 5, completions_ids=[[]] * 5, trainer_state=None,
                      sample_id=list(range(5)), ground_truth=['42'] * 5)
        self.assertEqual(strict_accuracy_reward(completions=completions, **kwargs), [0., 1., 0., 0., 0.])
        self.assertEqual(weak_accuracy_reward(completions=completions, **kwargs), [1., 1., 0., 0., 0.])
        self.assertEqual(format_reward(completions=completions, **kwargs), [1., 1., 0., 0., 0.])

    def test_alignment_errors(self):
        for gold in [None, '42', [], ['42', '7']]:
            with self.assertRaises(ValueError):
                strict_accuracy_reward(['<answer>42</answer>'], gold)
        with self.assertRaises(TypeError):
            format_reward('not a batch')
        self.assertEqual(strict_accuracy_reward([], []), [])
        self.assertEqual(weak_accuracy_reward(['42'], [None]), [0.])


class SFTTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [4, 5] if add_generation_prompt else [4, 5, 6, 99]


class SFTTests(unittest.TestCase):
    def row(self):
        return {'prompt': [{'role': 'user', 'content': 'question'}], 'ground_truth': '42',
                'completion': [{'role': 'assistant', 'content': 'Reason <answer>42</answer>'}]}

    def test_completion_boundary(self):
        feature = tokenize_completion(self.row(), SFTTokenizer(), 4)
        self.assertEqual(feature, {'input_ids': [4, 5, 6, 99], 'completion_mask': [0, 0, 1, 1]})
        checked = inspect_labels(feature, {'input_ids': [[4, 5, 6, 99, 0]],
                                          'labels': [[-100, -100, 6, 99, -100]],
                                          'attention_mask': [[1, 1, 1, 1, 0]]})
        self.assertEqual(checked['shifted_targets'], 2)

    def test_rejects_truncation_bad_demo_and_boundary_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            tokenize_completion(self.row(), SFTTokenizer(), 3)
        row = self.row()
        row['completion'][0]['content'] = '<answer>7</answer>'
        with self.assertRaisesRegex(ValueError, 'demonstration'):
            tokenize_completion(row, SFTTokenizer(), 4)
        class WrongPrefix(SFTTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [4, 5] if add_generation_prompt else [4, 7, 6, 99]
        with self.assertRaisesRegex(ValueError, 'boundary'):
            tokenize_completion(self.row(), WrongPrefix(), 4)

    def test_rejects_supervised_prompt_or_padding(self):
        feature = {'input_ids': [4, 5, 6, 99], 'completion_mask': [0, 0, 1, 1]}
        for labels in ([4, -100, 6, 99, -100], [-100, -100, 6, 99, 0]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                inspect_labels(feature, {'input_ids': [[4, 5, 6, 99, 0]], 'labels': [labels],
                                         'attention_mask': [[1, 1, 1, 1, 0]]})


SOURCES = json.loads((ROOT / 'configs/sources.json').read_text())


CONFIG = json.loads((ROOT / 'configs/data.yaml').read_text())


class SnapshotTests(unittest.TestCase):
    def test_pins_and_path_safety(self):
        validate_sources(SOURCES)
        for change in ('branch', 'traversal', 'tokenizer'):
            sources = copy.deepcopy(SOURCES)
            if change == 'branch':
                sources['model']['revision'] = 'main'
            elif change == 'traversal':
                sources['dataset']['files']['../escape'] = next(iter(
                    sources['dataset']['files'].values()))
            else:
                sources['tokenizer']['revision'] = '0' * 40
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sources(sources)

    def test_byte_integrity_for_git_and_lfs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file'
            path.write_bytes(b'hello\n')
            identity = file_identity(path)
            self.assertEqual(identity['git-blob-sha1'], 'ce013625030ba8dba906f756967f9e9ca394464a')
            for algorithm in ('sha256', 'git-blob-sha1'):
                source = {'files': {'file': {'size': 6, 'algorithm': algorithm,
                                           'digest': identity[algorithm]}}}
                verify_snapshot(directory, source)
                path.write_bytes(b'Hello\n')  # equal length, altered bytes
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    verify_snapshot(directory, source)
                path.write_bytes(b'hello\n')
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_snapshot(directory, source)

    def test_frozen_splits_hashes_determinism_and_overwrite(self):
        def rows(prefix, n):
            return [{'question': f'{prefix} {i}', 'answer': 'reason #### 42'} for i in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('a', 'b')]
            for path in paths:
                manifest = freeze_data(rows('train', 24), rows('test', 8), CONFIG, 'smoke',
                                       path, SOURCES['dataset'])
                for name, digest in manifest['files'].items():
                    self.assertEqual(file_identity(path / name)['sha256'], digest)
            self.assertEqual((paths[0] / 'manifest.json').read_bytes(),
                             (paths[1] / 'manifest.json').read_bytes())
            before = (paths[0] / 'test.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):
                freeze_data(rows('train', 24), rows('test', 8), CONFIG, 'smoke',
                            paths[0], SOURCES['dataset'])
            self.assertEqual((paths[0] / 'test.jsonl').read_bytes(), before)

    def test_failure_is_retained_and_revision_mismatch_fails_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            config = {**CONFIG, 'revision': 'main'}
            with self.assertRaises(ValueError):
                prepare(SOURCES, config, 'smoke', out)
            self.assertFalse(out.exists())
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                with self.assertRaises(ImportError):
                    prepare(SOURCES, CONFIG, 'smoke', out)
            report = json.loads((out / 'preparation.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['download_enabled'])
            with self.assertRaises(FileExistsError):
                prepare(SOURCES, CONFIG, 'smoke', out)

    @unittest.skipUnless(importlib.util.find_spec('datasets'), 'requires existing target environment')
    def test_real_local_parquet_to_frozen_manifest(self):
        from datasets import Dataset
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            sources = copy.deepcopy(SOURCES)
            model = tmp / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            data = tmp / 'dataset'
            (data / 'main').mkdir(parents=True)
            for split, count in (('train', 7473), ('test', 1319)):
                Dataset.from_dict({'question': [f'{split} {i}' for i in range(count)],
                                   'answer': ['reason #### 42'] * count}).to_parquet(
                    data / f'main/{split}-00000-of-00001.parquet')
            for key, path in (('model', model), ('dataset', data)):
                sources[key]['files'] = {
                    str(p.relative_to(path)): {'size': p.stat().st_size, 'algorithm': 'sha256',
                                              'digest': file_identity(p)['sha256']}
                    for p in path.rglob('*') if p.is_file()}
            def local_snapshot(**kwargs):
                self.assertTrue(kwargs['local_files_only'])
                self.assertEqual(kwargs['revision'], sources[kwargs['repo_type']]['revision'])
                self.assertEqual(set(kwargs['allow_patterns']),
                                 set(sources[kwargs['repo_type']]['files']))
                return str(model if kwargs['repo_type'] == 'model' else data)
            with patch('huggingface_hub.snapshot_download', side_effect=local_snapshot):
                report = prepare(sources, CONFIG, 'smoke', tmp / 'run')
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['row_counts'], {'train': 7473, 'test': 1319})
            self.assertEqual(report['split_sizes'], CONFIG['profiles']['smoke'])
            self.assertEqual(set(report['dataset_fingerprints']), {'train', 'test'})


class TrainingPrecisionTests(unittest.TestCase):
    def test_accelerate_wrapped_and_reloaded_peft_share_probe_precision(self):
        try:
            import torch
            from accelerate import Accelerator
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest('requires the locked target environment')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest('requires BF16 CUDA')
        accelerator = Accelerator(mixed_precision='bf16')
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=32)
        base = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
        base_state = {name: p.detach().cpu().clone() for name, p in base.state_dict().items()}
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if '.lora_' in name:
                    parameter.uniform_(-0.1, 0.1)
        adapter_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                         if '.lora_' in name}
        prepared = accelerator.prepare_model(model)
        probe = torch.tensor([[1, 2, 3, 4]], device=accelerator.device)
        before = forward_probe(prepared, probe, torch)
        with tempfile.TemporaryDirectory() as tmp:
            prepared.save_pretrained(tmp, safe_serialization=True)
            reloaded_base = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
            reloaded_base.load_state_dict(base_state)
            restored = PeftModel.from_pretrained(reloaded_base, str(Path(tmp)),
                                                local_files_only=True).to(accelerator.device)
            restored_parameters = dict(restored.named_parameters())
            for name, parameter in adapter_state.items():
                self.assertTrue(torch.equal(parameter, restored_parameters[name].detach().cpu()))
            after = forward_probe(restored, probe, torch)
        comparison = compare_probe(before, after, torch)
        self.assertTrue(comparison['passed'], comparison)
        self.assertEqual(comparison['mismatched_logits'], 0)
        self.assertFalse(compare_probe(before, after + 1, torch)['passed'])
        invalid = after.clone()
        invalid[0] = float('nan')
        comparison = compare_probe(before, invalid, torch)
        self.assertFalse(comparison['passed'])
        self.assertIsNone(comparison['max_absolute_difference'])


class VerifierTests(unittest.TestCase):
    def test_numeric_equivalence(self):
        for value in ['1,200', '+1200.00', '1.2e3', '2400/2']:
            with self.subTest(value=value):
                self.assertEqual(normalize_number(value), Fraction(1200))
        self.assertEqual(normalize_number('−0.5'), Fraction(-1, 2))
        self.assertEqual(normalize_number('-0'), Fraction(0))

    def test_invalid_numbers(self):
        for value in [None, 42, '', '1,20', 'NaN', 'inf', '1/0', '42 dollars', '$42',
                      '50%', '2+2', '1e999999', '<answer>42</answer>']:
            with self.subTest(value=value):
                self.assertIsNone(normalize_number(value))

    def test_last_tag(self):
        self.assertFalse(strict_verifier('<answer>42</answer><answer>7</answer>', '42'))
        self.assertFalse(strict_verifier('<answer>42</answer><answer>oops</answer>', '42'))
        self.assertEqual(extract_final_answer('<answer>42</answer><answer>unfinished'), '42')
        self.assertEqual(extract_final_answer('<answer>bad</answer><answer>-2</answer>'), '-2')

    def test_malformed_tags(self):
        for value in [None, '', '<answer>42', '<answer></answer>', '<Answer>42</Answer>',
                      '<answer><answer>42</answer></answer>', '<answer>x<b>42</b></answer>',
                      '<answer><answer>1</answer><answer>42</answer></answer>']:
            self.assertIsNone(extract_final_answer(value))
            self.assertFalse(format_compliant(value))
            self.assertFalse(strict_verifier(value, '42'))

    def test_format_separate_from_correctness(self):
        self.assertTrue(format_compliant('Reasoning\n<answer>7</answer>'))
        self.assertFalse(strict_verifier('<answer>7</answer>', '42'))
        self.assertTrue(strict_verifier('<answer>42</answer> trailing', '42'))
        self.assertFalse(format_compliant('<answer>42</answer> trailing'))
        self.assertFalse(format_compliant('<answer>42</answer><answer>42</answer>'))

    def test_weak_counterexamples(self):
        for text in ['I considered 42 but choose <answer>7</answer>',
                     '<answer>42</answer><answer>7</answer>', 'Possible answers: 7, 42, 99.',
                     '42 is incorrect. <answer>0</answer>']:
            self.assertTrue(weak_verifier(text, '42'), text)
            self.assertFalse(strict_verifier(text, '42'), text)

    def test_weak_numeric_boundaries(self):
        for text in ['142', '-42', '42.5', '1,042', '42/7', 'x42', '42x', '0.42', '42e2']:
            self.assertFalse(weak_verifier(text, '42'), text)
        for text in ['42.', 'Value: 42, next', '<answer>42</answer>', '+42', '84/2', '4.2e1']:
            self.assertTrue(weak_verifier(text, '42'), text)
        self.assertTrue(weak_verifier('Answer −1,200.00', '-1200'))
        self.assertFalse(weak_verifier('42', None))
        self.assertFalse(weak_verifier(None, '42'))
