"""Pinned Hub acquisition and byte-verified local GSM8K/model preparation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import time

from .data import make_splits
from .preflight import code_revision

# Frozen smoke manifest identity for the pinned Qwen/GSM8K inputs.
SMOKE_MANIFEST_SHA256 = 'e3af538beefb55d867be23c5270947d4d5b5ec2851995bd97f98ca2a38aebf6c'


def file_identity(path):
    path = Path(path)
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
            blob.update(chunk)
    return {'size': size, 'sha256': sha.hexdigest(), 'git-blob-sha1': blob.hexdigest()}


def validate_sources(sources):
    for key in ('model', 'dataset'):
        source = sources[key]
        if not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
            raise ValueError('source revision must be a full commit SHA')
        if source['repo_type'] != key or not source['files']:
            raise ValueError('invalid repository type or empty file selection')
        for name, expected in source['files'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('unsafe snapshot filename')
            algorithm = expected['algorithm']
            length = {'sha256': 64, 'git-blob-sha1': 40}.get(algorithm)
            if (length is None or not re.fullmatch(f'[0-9a-f]{{{length}}}', expected['digest'])
                    or type(expected['size']) is not int or expected['size'] <= 0):
                raise ValueError('invalid expected file identity')
    if sources['tokenizer'] != {k: sources['model'][k] for k in ('repo_id', 'revision')}:
        raise ValueError('this acquisition unit requires a co-revisioned tokenizer')


def verify_snapshot(directory, source):
    result = {}
    for name, expected in source['files'].items():
        identity = file_identity(Path(directory) / name)
        if (identity['size'] != expected['size'] or
                identity[expected['algorithm']] != expected['digest']):
            raise ValueError(f'snapshot integrity mismatch: {name}')
        result[name] = {'size': identity['size'], 'sha256': identity['sha256']}
    return result


def freeze_data(train, test, config, profile, output, source):
    """Use the existing split contract; record hashes of the actual serialized gold files."""
    splits, manifest = make_splits(train, test, config['profiles'][profile], config['seed'])
    output = Path(output)
    output.mkdir(exist_ok=False)
    manifest.update(source=source, profile=profile, data_config=config, files={})
    for name, rows in splits.items():
        path = output / f'{name}.jsonl'
        with path.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['files'][path.name] = file_identity(path)['sha256']
    write_json(output / 'manifest.json', manifest)
    return manifest


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def prepare(sources, data_config, profile, output, download=False, cache_dir=None):
    validate_sources(sources)
    if (data_config['revision'] != sources['dataset']['revision']
            or data_config['dataset'] != sources['dataset']['repo_id']
            or data_config['name'] != 'main'):
        raise ValueError('dataset config/source revision mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    report = {'schema_version': 1, 'status': 'running', 'sources': sources,
              'code': code_revision(root), 'download_enabled': download,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'lock_sha256': file_identity(root / 'requirements/linux-py311-cu128.txt')['sha256'],
              'data_config': data_config, 'profile': profile, 'snapshots': {}}
    try:
        from huggingface_hub import snapshot_download
        from datasets import Dataset
        report['packages'] = {name: importlib.metadata.version(name)
                              for name in ('huggingface-hub', 'datasets', 'pyarrow')}
        for key in ('model', 'dataset'):
            source = sources[key]
            directory = snapshot_download(
                repo_id=source['repo_id'], repo_type=source['repo_type'],
                revision=source['revision'], allow_patterns=list(source['files']),
                local_files_only=not download, cache_dir=cache_dir, max_workers=2)
            report['snapshots'][key] = {'path': str(Path(directory).resolve()),
                                       'files': verify_snapshot(directory, source)}
        directory = Path(report['snapshots']['dataset']['path'])
        tables, fingerprints = {}, {}
        for split, count in (('train', 7473), ('test', 1319)):
            table = Dataset.from_parquet(str(directory / f'main/{split}-00000-of-00001.parquet'),
                                         cache_dir=str(output / 'arrow-cache'))
            if (len(table) != count or set(table.column_names) != {'question', 'answer'}
                    or any(table.features[name].dtype != 'string' for name in table.column_names)):
                raise ValueError(f'unexpected GSM8K main schema/count: {split}')
            fingerprints[split] = table._fingerprint
            tables[split] = list(table)
        manifest = freeze_data(tables['train'], tables['test'], data_config, profile,
                               output / 'data', sources['dataset'])
        report.update(status='passed', dataset_fingerprints=fingerprints,
                      row_counts={key: len(rows) for key, rows in tables.items()},
                      split_manifest_sha256=file_identity(output / 'data/manifest.json')['sha256'],
                      split_sizes=manifest['sizes'])
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', default='configs/sources.json')
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='smoke')
    parser.add_argument('--plan', action='store_true', help='show file/byte scope; no Hub access')
    parser.add_argument('--download', action='store_true', help='explicitly allow pinned Hub download')
    parser.add_argument('--cache-dir', help='reuse an existing Hub cache; defaults to HF_HOME')
    parser.add_argument('--output-dir', help='new evidence directory; refuses overwrite')
    args = parser.parse_args()
    sources = json.loads(Path(args.sources).read_text())
    validate_sources(sources)
    if args.plan:
        print(json.dumps({'sources': sources, 'selected_bytes': sum(
            f['size'] for key in ('model', 'dataset') for f in sources[key]['files'].values())},
            indent=2))
        return
    if not args.output_dir:
        parser.error('--output-dir is required for preparation')
    report = prepare(sources, json.loads(Path(args.data_config).read_text()), args.profile,
                     args.output_dir, args.download, args.cache_dir)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
