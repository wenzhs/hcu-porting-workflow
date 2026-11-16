from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import adaptation_common as common
import inspect_build_context as context
import probe_environment as probe
import record_command as recorder
import source_risk_scan as scanner


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cold start tests ')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'project'
        self.repo.mkdir()
        (self.repo / 'kernel.cu').write_text('__global__ void k() {}\n', encoding='utf-8')

    def json_file(self, name, data):
        path = self.base / name
        path.write_text(json.dumps(data), encoding='utf-8')
        return path


class ContextTests(Workspace):
    def test_empty_macros_and_contexts_remain_separate(self):
        text = '#define EMPTY\r\n#define NEXT 1\r\n#define LAST\n#define FN(x) x\n'
        self.assertEqual(common.parse_predefined_macros(text), {'EMPTY': '', 'NEXT': '1', 'LAST': ''})
        host, device = self.base / 'host.txt', self.base / 'device.txt'
        host.write_text('#define __FPT_DTK_ARCH__ 520\n#define __HIP_DEVICE_COMPILE__ 0\n')
        device.write_text('#define __FPT_DTK_ARCH__ 520\n#define __HIP_DEVICE_COMPILE__ 1\n')
        report = context.inspect_context(self.repo, host_macros=host, device_macros=device)
        self.assertEqual(report['device_context_comparison']['host'], '0')
        self.assertEqual(report['device_context_comparison']['device'], '1')
        self.assertTrue(report['warnings'])
        self.assertFalse(report['executed'])

    def test_header_standard_and_last_explicit_flag(self):
        header = self.base / 'ATen.h'
        header.write_text('#error C++20 or later compatible compiler is required\n')
        row = {'directory': str(self.repo), 'file': 'kernel.cu',
               'arguments': ['hipcc', '-std=c++17', '-DUSE_ROCM=1', '--offload-arch=gfx936']}
        db = self.json_file('compile_commands.json', [row])
        report = context.inspect_context(self.repo, compile_commands=db, requirement_header=header)
        self.assertEqual(report['units'][0]['header_comparison'], 'possible-standard-mismatch')
        row['arguments'].append('-std=c++20')
        db = self.json_file('compile_commands.json', [row])
        unit = context.inspect_context(self.repo, compile_commands=db, requirement_header=header)['units'][0]
        self.assertEqual(unit['explicit_standard_year'], 2020)
        self.assertEqual(unit['defines'], {'USE_ROCM': '1'})
        self.assertEqual(unit['architectures'], ['gfx936'])
        self.assertFalse(unit['compiled'])

    def test_standard_aliases_and_separate_values_follow_argument_order(self):
        header = self.base / 'requirements.h'
        header.write_text('#error C++20 or later compatible compiler is required\n')
        for override in (['--std=c++14'], ['--std', 'c++14'], ['-std', 'c++14'], ['/std:c++14']):
            row = {'directory': str(self.repo), 'file': 'kernel.cu',
                   'arguments': ['nvcc', '-std=c++20', *override, '-c', 'kernel.cu']}
            unit = context.inspect_context(self.repo, compile_commands=self.json_file('std.json', [row]),
                                           requirement_header=header)['units'][0]
            with self.subTest(override=override):
                self.assertEqual(unit['explicit_standard_year'], 2014)
                self.assertEqual(unit['header_comparison'], 'possible-standard-mismatch')
        for override in (['--std'], ['--std=c++future'], ['-Xcompiler=-std=c++14']):
            row['arguments'] = ['nvcc', '-std=c++20', *override]
            unit = context.inspect_context(self.repo, compile_commands=self.json_file('std.json', [row]),
                                           requirement_header=header)['units'][0]
            with self.subTest(override=override):
                self.assertEqual(unit['header_comparison'], 'unknown')
                self.assertTrue(unit['warnings'])

    @unittest.skipUnless(os.name == 'nt', 'Windows case-insensitive path selection')
    def test_selected_compilation_files_use_windows_path_equality(self):
        row = {'directory': str(self.repo).upper(), 'file': 'KERNEL.CU', 'arguments': ['hipcc']}
        database = self.json_file('case.json', [row])
        report = context.inspect_context(self.repo, compile_commands=database, files=['kernel.cu'])
        self.assertEqual(len(report['units']), 1)
        self.assertTrue(Path(report['units'][0]['file']).samefile(self.repo / 'kernel.cu'))

    def test_generated_root_cannot_have_a_regular_file_ancestor(self):
        blocker = self.repo / 'build-slot'
        blocker.write_text('not a directory')
        with self.assertRaises((common.AdaptationError, OSError)):
            context.inspect_context(self.repo, transform_map={'kernel.cu': 'k.hip'},
                                    generated_root=blocker / 'generated')

    def test_unknown_flags_and_shell_text_are_not_executed(self):
        marker = self.base / 'must-not-exist'
        db = self.json_file('commands.json', [{'directory': str(self.repo), 'file': 'kernel.cu',
                                            'command': f'compiler @flags ; touch {marker}'}])
        report = context.inspect_context(self.repo, compile_commands=db)
        self.assertEqual(report['units'][0]['header_comparison'], 'unknown')
        self.assertTrue(report['units'][0]['warnings'])
        self.assertFalse(marker.exists())
        with self.assertRaises(common.AdaptationError):
            context.inspect_context(self.repo, compile_commands=db, files=['missing.cu'])

    def test_transform_plan_rejects_escape_and_overwrite(self):
        generated = self.base / 'generated'
        self.assertEqual(common.validate_transform_map(self.repo, generated, {'kernel.cu': 'kernel.hip'}), {'kernel.cu': 'kernel.hip'})
        for mapping in ({'../other.cu': 'x.hip'}, {'kernel.cu': '../x.hip'}, {'kernel.cu': 'C:/shared/x.hip'}):
            with self.subTest(mapping=mapping), self.assertRaises(common.AdaptationError):
                common.validate_transform_map(self.repo, generated, mapping)
        (self.repo / 'other.cu').write_text('// other')
        with self.assertRaises(common.AdaptationError):
            common.validate_transform_map(self.repo, generated, {'kernel.cu': 'x.hip', 'other.cu': 'x.hip'})
        generated.mkdir()
        (generated / 'kernel.hip').write_text('// existing')
        with self.assertRaises(common.AdaptationError):
            common.validate_transform_map(self.repo, generated, {'kernel.cu': 'kernel.hip'})
        common.validate_transform_map(self.repo, generated, {'kernel.cu': 'kernel.hip'}, require_outputs=True)

    def test_transform_rejects_file_directory_collisions(self):
        (self.repo / 'second.cu').write_text('// second')
        generated = self.base / 'generated'
        for first, second in [('parent', 'parent/child.hip'), ('parent/child.hip', 'parent')]:
            with self.subTest(first=first), self.assertRaisesRegex(common.AdaptationError, 'collision'):
                common.validate_transform_map(self.repo, generated, {'kernel.cu': first, 'second.cu': second})
        generated.write_text('not a directory')
        with self.assertRaisesRegex(common.AdaptationError, 'directory'):
            common.validate_transform_map(self.repo, generated, {'kernel.cu': 'k.hip'})

    def test_ambiguous_transform_arguments_and_empty_database_are_rejected(self):
        host = self.base / 'host.txt'
        host.write_text('#define __HIPCC__ 1\n')
        for kwargs in ({'generated_root': self.base / 'out'}, {'outputs_exist': True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(common.AdaptationError, 'transform map'):
                context.inspect_context(self.repo, host_macros=host, **kwargs)
        with self.assertRaisesRegex(common.AdaptationError, 'nonempty'):
            context.inspect_context(self.repo, compile_commands=self.json_file('empty.json', []))
        with contextlib.redirect_stderr(io.StringIO()):
            rc = context.main(['--repo', str(self.repo), '--host-macros', str(host),
                               '--transform-map', str(self.json_file('null.json', None))])
        self.assertEqual(rc, 2)

    def test_cli_isolated_from_target_modules(self):
        (self.repo / 'json.py').write_text('raise RuntimeError("must not import target")')
        db = self.json_file('db.json', [{'directory': str(self.repo), 'file': 'kernel.cu', 'arguments': ['hipcc']}])
        done = subprocess.run([sys.executable, '-B', '-I', str(ROOT / 'scripts/inspect_build_context.py'),
                               '--repo', str(self.repo), '--compile-commands', str(db)], cwd=self.repo,
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse(json.loads(done.stdout)['executed'])


class RecorderTests(Workspace):
    def run_child(self, code, **kwargs):
        return recorder.record_command([sys.executable, '-B', '-I', '-c', code], cwd=self.repo,
                                       out_dir=self.base / 'evidence', **kwargs)

    def test_complete_streams_and_actual_rc(self):
        report = self.run_child('import sys; print("x"*6000); print("y"*7000,file=sys.stderr); sys.exit(7)')
        self.assertEqual(report['exit_code'], 7)
        self.assertFalse(report['command_succeeded'])
        self.assertGreater(Path(report['stderr']).stat().st_size, 7000)
        self.assertFalse(report['gpu_validated'])
        with self.assertRaises(common.AdaptationError):
            self.run_child('pass')

    def test_rc_zero_without_artifact_is_not_successful_build_evidence(self):
        report = self.run_child('pass', expected_artifacts=['result.o'])
        self.assertTrue(report['command_succeeded'])
        self.assertFalse(report['expected_artifacts_present'])
        self.assertFalse(report['gpu_validated'])

    def test_fresh_artifact_identity_and_existing_artifact_rejected(self):
        report = self.run_child('from pathlib import Path; Path("result.o").write_bytes(b"synthetic")', expected_artifacts=['result.o'])
        self.assertTrue(report['expected_artifacts_present'])
        self.assertEqual(report['artifacts'][0]['sha256'], common.file_sha256(self.repo / 'result.o'))
        with self.assertRaises(common.AdaptationError):
            recorder.record_command([sys.executable, '-I', '-c', 'pass'], cwd=self.repo,
                                    out_dir=self.base / 'second', expected_artifacts=['result.o'])

    def test_recorder_logs_cannot_be_expected_build_artifacts(self):
        for name in ('stdout.log', 'stderr.log', 'command.json'):
            with self.subTest(name=name), self.assertRaisesRegex(common.AdaptationError, 'overlap'):
                self.run_child('print("not a build")', expected_artifacts=[str(self.base / 'evidence' / name)])
        self.assertFalse((self.base / 'evidence').exists())

    def test_no_requested_artifact_check_remains_unknown(self):
        report = self.run_child('pass')
        self.assertTrue(report['command_succeeded'])
        self.assertFalse(report['artifact_check_requested'])
        self.assertIsNone(report['expected_artifacts_present'])

    def test_recorder_cli_preserves_rc_and_missing_artifact_failure(self):
        def run(label, code, *artifact_args):
            return subprocess.run([sys.executable, '-B', '-I', str(ROOT / 'scripts/record_command.py'),
                                   '--cwd', str(self.repo), '--out-dir', str(self.base / label), *artifact_args,
                                   '--', sys.executable, '-B', '-I', '-c', code],
                                  capture_output=True, text=True, timeout=30)
        failed = run('failed', 'import sys; sys.exit(9)')
        self.assertEqual(failed.returncode, 9, failed.stderr)
        self.assertEqual(json.loads(failed.stdout)['exit_code'], 9)
        missing = run('missing', 'pass', '--expected-artifact', 'missing.o')
        self.assertEqual(missing.returncode, 2, missing.stderr)
        self.assertEqual(json.loads(missing.stdout)['exit_code'], 0)
        successful = run('success', 'print("fixture only")')
        self.assertEqual(successful.returncode, 0, successful.stderr)

    @unittest.skipUnless(os.name == 'nt', 'Windows direct-child-only timeout handling')
    def test_descendant_timeout_evidence_is_explicitly_unfinalized(self):
        import ctypes
        from ctypes import wintypes
        release, pid_file = self.base / 'release', self.base / 'child.pid'
        late = self.repo / 'late.o'
        child_code = '\n'.join([
            'from pathlib import Path', 'import time',
            f'release = Path({str(release)!r})',
            'deadline = time.monotonic() + 8',
            'while not release.exists() and time.monotonic() < deadline: time.sleep(0.02)',
            f'Path({str(late)!r}).write_bytes(b"fixture after timeout")',
            'print("late descendant output", flush=True)',
        ])
        parent_code = '\n'.join([
            'import subprocess, sys, time', 'from pathlib import Path',
            f'child = subprocess.Popen([sys.executable, "-B", "-I", "-c", {child_code!r}])',
            f'Path({str(pid_file)!r}).write_text(str(child.pid))',
            'time.sleep(10)',
        ])
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = None
        try:
            report = self.run_child(parent_code, timeout=2, expected_artifacts=['late.o'])
            self.assertTrue(pid_file.is_file(), 'Fixture parent must launch its owned descendant')
            handle = kernel.OpenProcess(0x00100000, False, int(pid_file.read_text()))
            self.assertTrue(handle, f'Cannot wait for fixture child: {ctypes.get_last_error()}')
            self.assertEqual(kernel.WaitForSingleObject(handle, 0), 258, 'Descendant should still be waiting')
            self.assertTrue(report['timed_out'])
            self.assertFalse(report['evidence_finalized'])
            self.assertIsNone(report['expected_artifacts_present'])
            self.assertTrue(all(log['sha256'] is None and log['bytes'] is None for log in report['logs']))
            release.write_text('finish fixture')
            self.assertEqual(kernel.WaitForSingleObject(handle, 10000), 0)
            self.assertTrue(late.is_file())
            self.assertIn('late descendant output', Path(report['stdout']).read_text())
            self.assertIsNone(report['artifacts'][0]['nonempty'])
        finally:
            release.write_text('finish fixture')
            if handle:
                kernel.WaitForSingleObject(handle, 10000)
                kernel.CloseHandle(handle)

    def test_timeout_is_distinct_from_exit_zero(self):
        report = self.run_child('import time; time.sleep(5)', timeout=0.1)
        self.assertTrue(report['timed_out'])
        self.assertFalse(report['command_succeeded'])
        self.assertIsNotNone(report['exit_code'])
        self.assertEqual(report['evidence_finalized'], os.name == 'posix')


class ProbeEvidenceTests(Workspace):
    def runner(self, argv, timeout):
        if '-dM' in argv:
            stdout, stderr = '#define __HIP_PLATFORM_AMD__\n#define __HIP_PLATFORM_HCC__\n', ''
        else:
            stdout, stderr = 'compiler output', 'diagnostic ' * 800
        return {'argv': argv, 'exit_code': 0, 'output': (stdout + stderr)[:1500],
                '_full': stdout, '_stderr': stderr, 'error': None}

    def test_false_success_compile_and_persisted_inputs(self):
        directory = self.base / 'probe-evidence'
        report = probe.probe(arch='gfx936', compile_probe=True, evidence_dir=directory,
                             runner=self.runner, environ={}, distributions=lambda: [],
                             which=lambda n: '/fake/hipcc' if n == 'hipcc' else None)
        self.assertTrue(all(not p['passed'] for p in report['compile_probes']))
        self.assertTrue(all(Path(p['source_path']).is_file() for p in report['compile_probes']))
        self.assertEqual(report['macros_defined']['__HIP_PLATFORM_AMD__'], '')
        self.assertGreater((directory / '0002.stderr.log').stat().st_size, 1500)
        self.assertNotIn('_stderr', json.dumps(report))
        self.assertIsNone(report['devices']['kernel_launch_limits'])
        with self.assertRaises(common.AdaptationError):
            probe.probe(evidence_dir=directory)

    def test_invalid_probe_timeouts_do_not_start_commands(self):
        calls = []
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(common.AdaptationError):
                probe.probe(timeout=value, runner=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_real_rocminfo_field_spellings(self):
        info = 'Agent 1\n  Name: gfx936\n  Wavefront Size: 64\n  Workgroup Max Size: 1024\n  Compute Unit: 60\n  Max Waves Per CU: 32\n'
        limits = probe.parse_launch_limits(info)['gfx936']
        self.assertEqual(limits['max_workgroup_size'], 1024)
        self.assertEqual(limits['max_compute_units'], 60)


class FocusedScanTests(Workspace):
    def test_grouped_scan_and_python_backend_format_hints(self):
        (self.repo / 'ops.py').write_text('import triton\nx = torch.float8_e4m3fnuz\ny = tl.float8e4nv\n')
        (self.repo / 'kernel.cu').write_text('int x = CUDA_VERSION;\nint y = CUDA_VERSION;\n')
        report = scanner.scan_risks(self.repo, files_from=['kernel.cu', 'ops.py'], max_findings=1)
        self.assertEqual(report['files_scanned'], 2)
        self.assertEqual(report['files_with_findings'], 2)
        self.assertTrue(report['truncated'])
        self.assertEqual(sum(g['count'] for g in report['groups']), report['total_findings'])
        self.assertTrue(all(g['status'] == 'unreviewed' for g in report['groups']))
        self.assertIn('backend-dependency', report['summary']['by_category'])
        self.assertEqual(report['summary']['by_category']['low-precision-type'], 2)

    def test_summary_and_mapping_cli_preserve_scope_and_counts(self):
        (self.repo / 'kernel.cu').write_text('int a = CUDA_VERSION;\nint b = CUDA_VERSION;\n')
        mapping = self.json_file('map.json', {'kernel.cu': 'other.hip'})
        detail = self.base / 'detail.json'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = scanner.main(['--repo', str(self.repo), '--transform-map', str(mapping),
                               '--summary', '--max-findings', '1', '--json-out', str(detail)])
        self.assertEqual(rc, 0)
        summary, full = json.loads(stdout.getvalue()), json.loads(detail.read_text())
        self.assertNotIn('findings', summary)
        self.assertTrue(summary['truncated'])
        self.assertEqual(summary['total_findings'], full['total_findings'])
        self.assertEqual(summary['scope']['selected_file_count'], 1)
        self.assertTrue(all('files' not in g and 'file_count' in g for g in summary['groups']))
        for args in (['--files-from', str(self.json_file('null-files.json', None))],
                     ['--map-side', 'generated'], ['--max-findings', '0']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(scanner.main(['--repo', str(self.repo), *args]), 2)
        generated = self.base / 'generated'
        generated.mkdir()
        (generated / 'other.hip').write_text('int a = CUDA_VERSION;\n')
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = scanner.main(['--repo', str(generated), '--transform-map', str(mapping), '--map-side', 'generated'])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue())['findings'][0]['file'], 'other.hip')

    def test_selection_and_exclusion_are_explicit(self):
        (self.repo / 'build').mkdir()
        (self.repo / 'build/owned.cu').write_text('int x = CUDA_VERSION;\n')
        selected = scanner.scan_risks(self.repo, files_from=['kernel.cu'])
        self.assertEqual(selected['files_scanned'], 1)
        excluded = scanner.scan_risks(self.repo, exclude_dirs=['build'])
        self.assertEqual(excluded['files_scanned'], 1)
        for files in ([], ['../other.cu'], ['missing.cu']):
            with self.subTest(files=files), self.assertRaises(common.AdaptationError):
                scanner.scan_risks(self.repo, files_from=files)
