from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts'))
import adaptation_common as common
import route_preflight
import build_skill_packages as exporter


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='independent skill packages ')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / 'export'
        self.summary = exporter.export_packages(self.output)
        self.project = self.base / 'project'
        self.project.mkdir()
        (self.project / 'CMakeLists.txt').write_text('project(example LANGUAGES CXX HIP)\n')
        (self.project / 'kernel.cu').write_text('// CUDA source selected by project HIP backend\n')
        for module in ('torch.py', 'fastpt.py', 'json.py'):
            (self.project / module).write_text('raise RuntimeError("target code must never be imported")\n')
        self.request = self.base / 'request.json'
        self.request.write_text(json.dumps({'repo': str(self.project), 'arch': 'gfx936', 'build_command': ['cmake', '--build', 'build'], 'capabilities': {'native_backend': {'state': 'available'}, 'hip': {'installed': True, 'compiler': 'hipcc'}, 'fastpt': {'installed': True, 'version_compatible': False, 'invocable': False}, 'converters': {'hipify-clang': {key: True for key in ('installed', 'version_compatible', 'invocable', 'feature_probed')}}}}), encoding='utf-8')

    def run_package(self, folder, *args):
        return subprocess.run([sys.executable, '-B', '-I', '-X', 'utf8', str(folder / 'scripts/route_preflight.py'), *args], cwd=self.project, capture_output=True, text=True, encoding='utf-8', timeout=30, check=False)

    def test_four_packages_have_closed_resources(self):
        self.assertFalse(self.summary['installed'])
        self.assertEqual(set(self.summary['packages']), set(exporter.NAMES))
        for name in exporter.NAMES:
            folder = self.output / name
            exporter.check_links(folder)
            manifest = route_preflight.verify_package(folder)
            self.assertEqual(manifest['skill_name'], name)
            result = self.run_package(folder, '--verify-package-only')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['valid'])

    def test_each_skill_runs_alone_after_relocation(self):
        expected = {'hcu-native-adaptation': 'native-hip', 'fastpt-non-transcode': 'non-transcode', 'hcu-transcode': 'transcode', 'hcu-porting-workflow': 'transcode'}
        # A standalone parent contains exactly one package, no sibling skill.
        for index, name in enumerate(exporter.NAMES):
            isolated = self.base / f'isolated parent {index}' / 'renamed skill with spaces'
            isolated.parent.mkdir()
            shutil.copytree(self.output / name, isolated)
            result = self.run_package(isolated, '--request', str(self.request))
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['route'], expected[name])
            self.assertEqual(plan['selection_mode'], 'auto' if name == 'hcu-porting-workflow' else 'explicit')
            self.assertFalse(plan['executed'])
            self.assertTrue(all(value == 'not-run' for value in plan['stages'].values()))
            if name == 'hcu-native-adaptation':
                self.assertFalse((isolated / 'scripts/routes/non_transcode.py').exists())
                self.assertFalse((isolated / 'scripts/fastpt_non_transcode_prepare.py').exists())
                self.assertFalse((isolated / 'scripts/classify_fastpt_failure.py').exists())
                self.assertEqual(plan['plan_status'], 'planned')
            elif name == 'fastpt-non-transcode':
                self.assertEqual(plan['plan_status'], 'blocked')

    def test_route_lock_and_strict_exit_are_enforced(self):
        native = self.output / 'hcu-native-adaptation'
        result = self.run_package(native, '--request', str(self.request), '--route', 'transcode')
        self.assertEqual(result.returncode, 2)
        self.assertIn('locked', result.stderr)
        result = self.run_package(self.output / 'fastpt-non-transcode', '--request', str(self.request), '--strict')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)['plan_status'], 'blocked')

    def test_missing_manifest_or_resource_is_an_error(self):
        folder = self.output / 'hcu-native-adaptation'
        (folder / 'skill-package.json').unlink()
        result = self.run_package(folder, '--request', str(self.request))
        self.assertEqual(result.returncode, 2)
        self.assertIn('manifest missing', result.stderr)
        other = self.output / 'hcu-transcode'
        (other / 'docs/contract.md').unlink()
        self.assertEqual(self.run_package(other, '--verify-package-only').returncode, 2)

    def test_modified_resource_and_incompatible_version_fail(self):
        folder = self.output / 'hcu-native-adaptation'
        with (folder / 'docs/contract.md').open('a', encoding='utf-8') as stream:
            stream.write('\nmodified\n')
        self.assertEqual(self.run_package(folder, '--verify-package-only').returncode, 2)
        other = self.output / 'hcu-transcode'
        path = other / 'skill-package.json'
        data = json.loads(path.read_text())
        data['contract_version'] = '999'
        path.write_text(json.dumps(data))
        self.assertEqual(self.run_package(other, '--verify-package-only').returncode, 2)

    def test_unmanifested_code_and_malformed_routes_fail_cleanly(self):
        folder = self.output / 'hcu-native-adaptation'
        (folder / 'scripts/unknown.py').write_text('raise RuntimeError()\n')
        result = self.run_package(folder, '--verify-package-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('unmanifested', result.stderr)
        other = self.output / 'hcu-transcode'
        path = other / 'skill-package.json'
        data = json.loads(path.read_text())
        data['available_routes'] = [{}]
        path.write_text(json.dumps(data))
        result = self.run_package(other, '--verify-package-only')
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('Traceback', result.stderr)

    def test_export_never_overwrites_or_nests_in_source(self):
        sentinel = self.output / 'sentinel'
        sentinel.write_text('keep')
        with self.assertRaises(common.AdaptationError):
            exporter.export_packages(self.output)
        self.assertEqual(sentinel.read_text(), 'keep')
        with self.assertRaises(common.AdaptationError):
            exporter.export_packages(ROOT / 'should-never-exist')
        self.assertFalse((ROOT / 'should-never-exist').exists())

    def test_common_component_is_identical_in_all_packages(self):
        hashes = {common.file_sha256(self.output / name / 'scripts/adaptation_common.py') for name in exporter.NAMES}
        self.assertEqual(len(hashes), 1)
        self.assertEqual(next(iter(hashes)), common.file_sha256(ROOT / 'scripts/adaptation_common.py'))

    def test_investigation_tools_ship_and_run_in_every_package(self):
        tools = ('probe_environment.py', 'source_risk_scan.py', 'adaptation_ledger.py',
                 'inspect_build_context.py', 'record_command.py')
        for index, name in enumerate(exporter.NAMES):
            isolated = self.base / f'tools parent {index}' / 'renamed skill'
            isolated.parent.mkdir()
            shutil.copytree(self.output / name, isolated)
            for document in ('docs/tools.md', 'docs/pre-build-analysis.md', 'references/build-adaptation.md', 'references/instruction-porting.md'):
                self.assertTrue((isolated / document).is_file(), f'{name}: {document}')
            self.assertEqual(self.run_package(isolated, '--verify-package-only').returncode, 0)
            for script in tools:
                path = isolated / 'scripts' / script
                helped = subprocess.run([sys.executable, '-B', '-I', str(path), '--help'], cwd=self.project, capture_output=True, text=True, timeout=30, check=False)
                self.assertEqual(helped.returncode, 0, f'{name}/{script}: {helped.stderr}')
                unisolated = subprocess.run([sys.executable, '-B', str(path), '--help'], cwd=self.project, capture_output=True, text=True, timeout=30, check=False)
                self.assertEqual(unisolated.returncode, 2, f'{name}/{script} must refuse non-isolated start')
                self.assertIn('requires python -I', unisolated.stderr)
            scan = subprocess.run([sys.executable, '-B', '-I', str(isolated / 'scripts/source_risk_scan.py'), '--repo', str(self.project)], cwd=self.project, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(scan.returncode, 0, scan.stderr)
            self.assertEqual(json.loads(scan.stdout)['kind'], 'risk-scan')
            environment = subprocess.run([sys.executable, '-B', '-I', str(isolated / 'scripts/probe_environment.py')], cwd=self.project, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(environment.returncode, 0, environment.stderr)
            report = json.loads(environment.stdout)
            self.assertTrue(report['read_only'])
            # Native package does not collect FastPT capabilities

    def test_cold_start_clis_execute_after_relocation(self):
        database = self.base / 'compile_commands.json'
        database.write_text(json.dumps([{'directory': str(self.project), 'file': 'kernel.cu',
                                         'arguments': ['hipcc', '-std=c++17']}]))
        for index, name in enumerate(exporter.NAMES):
            folder = self.base / f'cold-start-{index}' / 'relocated package'
            shutil.copytree(self.output / name, folder)
            self.assertFalse((folder / 'docs/evals').exists())
            for document in ('build-adaptation', 'code-conversion', 'instruction-porting', 'common-errors-and-solutions'):
                self.assertTrue((folder / f'references/{document}.md').is_file())
            inspected = subprocess.run([sys.executable, '-B', '-I', str(folder / 'scripts/inspect_build_context.py'),
                                        '--repo', str(self.project), '--compile-commands', str(database)],
                                       cwd=self.project, capture_output=True, text=True, timeout=30)
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertFalse(json.loads(inspected.stdout)['executed'])
            evidence = self.base / f'command-{index}'
            recorded = subprocess.run([sys.executable, '-B', '-I', str(folder / 'scripts/record_command.py'),
                                       '--cwd', str(self.project), '--out-dir', str(evidence), '--',
                                       sys.executable, '-B', '-I', '-c', 'print("fixture only")'],
                                      cwd=self.project, capture_output=True, text=True, timeout=30)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            self.assertFalse(json.loads(recorded.stdout)['gpu_validated'])
            self.assertTrue((evidence / 'command.json').is_file())
            # A missing experience resource must block the recorder before spawning.
            (folder / 'references/instruction-porting.md').unlink()
            rejected = subprocess.run([sys.executable, '-B', '-I', str(folder / 'scripts/record_command.py'),
                                       '--cwd', str(self.project), '--out-dir', str(self.base / f'not-run-{index}'),
                                       '--', sys.executable, '-B', '-I', '-c', 'print("must not run")'],
                                      cwd=self.project, capture_output=True, text=True, timeout=30)
            self.assertEqual(rejected.returncode, 2)
            self.assertFalse((self.base / f'not-run-{index}').exists())
            self.assertFalse(list(folder.rglob('__pycache__')))

    def test_corrupted_tool_is_rejected_before_it_runs(self):
        folder = self.output / 'fastpt-non-transcode'
        with (folder / 'scripts/source_risk_scan.py').open('a', encoding='utf-8') as stream:
            stream.write('\n# tampered\n')
        result = subprocess.run([sys.executable, '-B', '-I', str(folder / 'scripts/source_risk_scan.py'), '--repo', str(self.project)], cwd=self.project, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('hash mismatch', result.stderr)

    def test_legacy_entrypoints_work_with_isolated_python(self):
        entries = {'fastpt-non-transcode': ['fastpt_non_transcode_prepare.py', 'classify_fastpt_failure.py'], 'hcu-transcode': ['cmake_fastpt_convert.py', 'cutlass_to_hytlass_prepare.py', 'classify_fastpt_failure.py']}
        for name, scripts in entries.items():
            for script in scripts:
                result = subprocess.run([sys.executable, '-B', '-I', str(self.output / name / 'scripts' / script), '--help'], cwd=self.project, capture_output=True, text=True, timeout=30, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
