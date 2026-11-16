from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import adaptation_common as common
import adaptation_ledger as ledger


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scope binding ')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'source'
        self.repo.mkdir()
        (self.repo / 'CMakeLists.txt').write_text('project(x LANGUAGES HIP)\n')
        self.path = self.base / 'ledger.json'
        self.book = ledger.init_ledger(self.repo, self.path, features=['basic', 'original-platform'])
        self.artifact = self.base / 'fixture.bin'
        self.artifact.write_bytes(b'synthetic fixture, not GPU evidence')
        sha = self.book['source_sha256']
        self.receipt = {
            'schema_version': common.SCHEMA_VERSION, 'kind': 'adaptation-result', 'status': 'completed',
            'route': 'native-hip', 'source_sha256': sha, 'covered_entries': ['feature:basic'],
            'stages': {s:'passed' for s in ('build','load','gpu','correctness')},
            'tests': {'total':1,'passed':1,'failed':0,'skipped':0,'gpu_passed':1},
            'oracle': 'synthetic fixture', 'validation_contract': {'required_stages':['build','gpu']},
            'commands': [{'phase':p,'argv':['synthetic'],'exit_code':0} for p in ('build','correctness')],
            'artifacts': [{'path':str(self.artifact),'sha256':common.file_sha256(self.artifact),
                           'source_sha256':sha,'loaded':True}],
        }
        self.rpath = self.base / 'receipt.json'

    def save_receipt(self):
        self.rpath.write_text(json.dumps(self.receipt), encoding='utf-8')
        return self.rpath

    def test_scope_missing_partial_and_legacy_labels_are_rejected(self):
        original = dict(self.receipt)
        for status, scope in [('partial', ['feature:basic']), ('completed', None), ('completed', ['basic'])]:
            self.receipt = {**original, 'status':status, 'covered_entries':scope, 'target_features':['basic']}
            with self.subTest(status=status, scope=scope), self.assertRaises(common.AdaptationError):
                ledger.set_status(self.path, 'feature:basic', 'covered', receipt=self.save_receipt())

    def test_cli_cannot_widen_receipt_scope(self):
        receipt = self.save_receipt()
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, 'feature:original-platform', 'covered', receipt=receipt, target_features=['feature:basic'])
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, 'feature:basic', 'covered', receipt=receipt, target_features=['feature:basic','feature:other'])
        entry = ledger.set_status(self.path, 'feature:basic', 'covered', receipt=receipt, target_features=['feature:basic'])
        self.assertEqual(entry['route'], 'native-hip')
        self.assertEqual(ledger.check_ledger(self.path)['problems'], [])

    def test_named_stage_is_bound_to_entry_not_self_declared_receipt_contract(self):
        self.receipt['covered_entries'] = ['feature:original-platform']
        with self.assertRaisesRegex(common.AdaptationError, 'original-platform'):
            ledger.set_status(self.path, 'feature:original-platform', 'covered', receipt=self.save_receipt())
        self.receipt['stages']['original-platform'] = 'passed'
        ledger.set_status(self.path, 'feature:original-platform', 'covered', receipt=self.save_receipt())

    def test_cli_rejects_explicit_null_or_nonmapping_requirements(self):
        requirements = self.base / 'requirements.json'
        script = Path(ledger.__file__)
        for index, value in enumerate((None, [], False, 7, 'integration')):
            requirements.write_text(json.dumps(value), encoding='utf-8')
            output = self.base / f'rejected-{index}.json'
            done = subprocess.run([sys.executable, '-B', '-I', str(script), 'init', '--repo', str(self.repo),
                                   '--out', str(output), '--feature', 'basic', '--requirements', str(requirements)],
                                  capture_output=True, text=True, timeout=30)
            with self.subTest(value=value):
                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                self.assertFalse(output.exists())
        requirements.write_text(json.dumps({'feature:basic': ['integration']}), encoding='utf-8')
        output = self.base / 'valid-cli.json'
        done = subprocess.run([sys.executable, '-B', '-I', str(script), 'init', '--repo', str(self.repo),
                               '--out', str(output), '--feature', 'basic', '--requirements', str(requirements)],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('integration', json.loads(output.read_text())['entries'][0]['required_stages'])

    def test_additional_entry_requirements(self):
        path = self.base / 'extra.json'
        ledger.init_ledger(self.repo, path, features=['basic'], requirements={'feature:basic':['integration']})
        with self.assertRaisesRegex(common.AdaptationError, 'integration'):
            ledger.set_status(path, 'feature:basic', 'covered', receipt=self.save_receipt())

    def test_check_reapplies_scope_validation(self):
        ledger.set_status(self.path, 'feature:basic', 'covered', receipt=self.save_receipt())
        self.receipt['covered_entries'] = ['feature:other']
        self.save_receipt()
        data = json.loads(self.path.read_text())
        data['entries'][0]['receipt']['sha256'] = common.file_sha256(self.rpath)
        self.path.write_text(json.dumps(data))
        report = ledger.check_ledger(self.path)
        self.assertTrue(report['problems'])
        self.assertFalse(report['complete'])

    def test_repository_summary_changes_invalidate_covered_entries(self):
        ledger.set_status(self.path, 'feature:basic', 'covered', receipt=self.save_receipt())
        self.assertFalse(ledger.check_ledger(self.path)['source_changed'])
        (self.repo / 'ADAPTATION_SUMMARY.md').write_text('Final report written after validation\n')
        report = ledger.check_ledger(self.path)
        self.assertTrue(report['source_changed'])
        self.assertTrue(any('stale' in problem['problem'] for problem in report['problems']))
        with self.assertRaisesRegex(common.AdaptationError, 'Source changed'):
            ledger.set_status(self.path, 'feature:basic', 'covered', receipt=self.rpath)

    def test_three_identical_names_have_stable_unique_ids(self):
        items = [{'type':'test','name':'same','file':'CMakeLists.txt','line':1}] * 3
        taken = set()
        ids = [ledger._entry(item, taken)['id'] for item in items]
        self.assertEqual(len(set(ids)), 3)
        again = set()
        self.assertEqual(ids, [ledger._entry(item, again)['id'] for item in items])
        data = json.loads(self.path.read_text())
        data['entries'].append(data['entries'][0])
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(common.AdaptationError, 'unique'):
            ledger.check_ledger(self.path)

    def test_template_filter_does_not_remove_real_target_names(self):
        (self.repo / 'CMakeLists.txt').write_text('add_library(ALIAS_helper x.cpp)\nadd_library(real_SOURCES x.cpp)\nadd_library("prefix_${name}" x.cpp)\n')
        names = {e['name'] for e in ledger.enumerate_components(self.repo, None)}
        self.assertEqual(names, {'ALIAS_helper', 'real_SOURCES'})
