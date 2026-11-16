"""Offline resource/CLI consistency checks, not model or GPU evaluations."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_skill_packages as exporter
import source_risk_scan as scanner


class DocumentationTests(unittest.TestCase):
    def test_reference_guides_and_contracts_are_self_contained(self):
        expected = {'resources.md', 'build-adaptation.md', 'code-conversion.md', 'instruction-porting.md', 'common-errors-and-solutions.md', 'fastpt-non-transcode.md', 'pytorch-hipify.md'}
        self.assertEqual({path.name for path in (ROOT / 'references').glob('*.md')}, expected)
        for package in exporter.NAMES:
            mapping = exporter.sources_for(package)
            with self.subTest(package=package):
                for name in expected - {'fastpt-non-transcode.md'}:
                    self.assertIn('references/' + name, mapping)
                self.assertEqual('references/fastpt-non-transcode.md' in mapping,
                                 package in {'hcu-porting-workflow', 'fastpt-non-transcode'})
                for name in ('contract', 'tools', 'pre-build-analysis', 'adaptation-summary'):
                    self.assertIn(f'docs/{name}.md', mapping)
                self.assertNotIn('docs/README.md', mapping)
                self.assertNotIn('docs/sources.md', mapping)

    def test_exported_shell_examples_use_shipped_scripts_and_valid_flags(self):
        checked = 0
        for package in exporter.NAMES:
            mapping = exporter.sources_for(package)
            flags = {}
            for relative, source in mapping.items():
                if relative in exporter.PUBLIC_CLIS:
                    tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
                    flags[Path(relative).name] = {
                        a.value for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'add_argument'
                        for a in node.args if isinstance(a, ast.Constant)
                        and isinstance(a.value, str) and a.value.startswith('--')
                    } | {'--help'}
            for source in mapping.values():
                if not source.endswith('.md'):
                    continue
                text = (ROOT / source).read_text(encoding='utf-8')
                for block in re.findall(r'```(?:bash|sh|shell)\n(.*?)```', text, re.S):
                    block = block.replace('\\\n', ' ')
                    pattern = r'(?m)^python(?:3)?(?:[ \t]+-\S+)*[ \t]+"?(?:<skill-root>/)?scripts/([\w-]+\.py)"?([^\n]*)'
                    for match in re.finditer(pattern, block):
                        script, tail = match.groups()
                        with self.subTest(package=package, source=source, script=script):
                            self.assertIn(script, flags, 'Example refers to an unavailable CLI')
                            options = re.findall(r'(?<!\S)(--[a-z][\w-]*)', tail.split(' -- ', 1)[0])
                            self.assertFalse(set(options) - flags[script], f'Unknown CLI flags: {options}')
                        checked += 1
        self.assertGreater(checked, 5, 'Checks must exercise actual shell examples')

    def test_scanner_cards_and_guidance_resolve_in_every_package(self):
        script = (ROOT / 'scripts/source_risk_scan.py').read_text(encoding='utf-8')
        cards = set(re.findall(r'\b[WS]-\d{2}\b', script))
        documented = {f'W-{i:02d}' for i in range(1, 14)} | {f'S-{i:02d}' for i in range(1, 12)}
        self.assertFalse(cards - documented)
        # Several manual-only cards describe cross-kernel semantics that a
        # heuristic text scanner cannot establish; all must remain accessible.
        targets = set(scanner.GUIDANCE.values()) | {scanner.card_reference(card) for card in documented}
        for target in targets:
            path, anchor = target.split('#')
            with self.subTest(target=target):
                self.assertIn(f'id="{anchor}"', (ROOT / path).read_text(encoding='utf-8'))
                for package in exporter.NAMES:
                    self.assertIn(path, exporter.sources_for(package))

    def test_maintainer_evaluation_is_not_in_application_packages(self):
        for package in exporter.NAMES:
            self.assertFalse(any(p.startswith('docs/evals/') for p in exporter.sources_for(package)))
        samples = {}
        for name in ('round-input.example.json', 'round-metrics.example.json'):
            data = json.loads((ROOT / 'docs/evals' / name).read_text(encoding='utf-8'))
            self.assertIsInstance(data, dict)
            self.assertTrue(data['example_only'])
            samples[name] = data
        inputs, metrics = samples['round-input.example.json'], samples['round-metrics.example.json']
        self.assertTrue(inputs['independence']['freeze_skill_during_run'])
        self.assertFalse(inputs['execution_context']['inherited_component_context'])
        self.assertFalse(inputs['user_authorization']['remote_execution_authorized'])
        self.assertEqual(metrics['status'], 'not-run')
        self.assertFalse(metrics['measured'])
        self.assertTrue(all(status == 'not-run' for status in metrics['stages'].values()))
        self.assertIsNone(metrics['tests']['required_gpu_passed'])
        self.assertFalse(metrics['merge_gate']['ready_for_merge'])


if __name__ == '__main__':
    unittest.main()
