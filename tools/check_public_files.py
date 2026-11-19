"""Scan publication files without printing potential credential values.

This is a small local guard, not a replacement for reviewing new data or commits.
Uses only Python's standard library; no network access or API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
PATTERNS = {
    'private_key_material': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'provider_key_shape': re.compile(r'\bsk-[A-Za-z0-9_-]{16,}\b'),
    'github_token_shape': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,})\b'),
    'aws_access_key_shape': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    'personal_windows_path': re.compile(r'[A-Za-z]:(?:\\\\|\\|/)(?:csrse|Users[/\\]+ASUS)(?:[/\\]|\b)', re.I),
    'rented_server_reference': re.compile(r'\bconnect\.[A-Za-z0-9.-]*seetacloud\.com\b', re.I),
    'credential_in_url': re.compile(r'https?://[^\s/:\"<>]+:[^\s/@\"<>]{8,}@'),
    'bearer_token_literal': re.compile(r'(?i)bearer\s+[A-Za-z0-9._-]{24,}'),
}
CREDENTIAL_FIELDS = {'api_key', 'apikey', 'authorization', 'password', 'passwd', 'access_token', 'client_secret'}
PLACEHOLDERS = ('your-', 'your_', 'example', 'placeholder', 'dummy', 'mock', 'native-', 'test-', 'not-a-real')


def credential_values(obj, prefix=''):
    if isinstance(obj, dict):
        for key, val in obj.items():
            field = str(key).lower().replace('-', '_')
            if field in CREDENTIAL_FIELDS and isinstance(val, str):
                if len(val) >= 12 and not any(x in val.lower() for x in PLACEHOLDERS):
                    yield prefix + '/' + str(key)
            if isinstance(val, (dict, list)):
                yield from credential_values(val, prefix + '/' + str(key))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, (dict, list)):
                yield from credential_values(val, prefix + '/' + str(i))


def scan(root: Path):
    findings = []
    text_count = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts) or rel.parts[0] == 'results':
            continue
        name = path.name.lower()
        detectors = []
        if ((name.startswith('api_key') and path.suffix.lower() == '.txt')
                or (name.startswith('.env') and name != '.env.example')
                or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}
                or name in {'id_rsa', 'id_ed25519', 'credentials.json'}):
            detectors.append('credential_filename')
        if path.stat().st_size >= 100 * 1024 * 1024:
            detectors.append('github_100_mib_file_limit')
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            text = None
        if text is not None:
            text_count += 1
            detectors.extend(label for label, pattern in PATTERNS.items() if pattern.search(text))
            if path.suffix.lower() == '.json':
                try:
                    fields = list(credential_values(json.loads(text)))
                except json.JSONDecodeError:
                    fields = []
                if fields:
                    detectors.append('nonplaceholder_credential_value')
        if detectors:
            findings.append({'path': rel.as_posix(), 'detectors': sorted(set(detectors))})
    return {'status': 'passed' if not findings else 'needs_review',
            'text_files_scanned': text_count, 'findings': findings,
            'pattern_counts': dict(Counter(x for row in findings for x in row['detectors'])),
            'potential_secret_values_printed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Optional JSON report; store generated output outside the tracked release')
    args = parser.parse_args(argv)
    result = scan(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
