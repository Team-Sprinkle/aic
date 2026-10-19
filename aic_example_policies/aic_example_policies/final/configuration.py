"""Portable, checksum-verified configuration for the selected controller."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

CONFIG = Path(__file__).with_name('config')
_applied = False


def verify_artifacts(root):
    root = Path(root).resolve()
    manifest = json.loads((CONFIG / 'artifacts.json').read_text())
    for relative, expected in manifest['files'].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing final-controller artifact: {relative}; see docs/FINAL_SUBMISSION.md')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {relative}')
    return manifest


def apply_configuration():
    global _applied
    if _applied:
        return
    root = Path(os.environ.get('AIC_ARTIFACT_ROOT', '/opt/aic/artifacts')).resolve()
    verify_artifacts(root)
    config = json.loads((CONFIG / 'runtime.json').read_text())
    for key, value in config['environment'].items():
        os.environ[key] = value
    for key, relative in config['artifact_environment'].items():
        os.environ[key] = str(root / relative)
    os.environ['AIC_CONNECTOR_MODEL_ROOT'] = str(root / 'connector_models')
    os.environ['AIC_SHARED_PORT_TCP_DEVICE'] = os.environ.get('AIC_DEVICE', 'cuda:0')
    # Only traces and effective config are written; model artifacts stay read-only.
    spec = json.loads((CONFIG / 'transport.json').read_text())
    spec['bundle'] = {k: str(root / p) for k, p in spec['bundle'].items()}
    directory = Path(os.environ.get('AIC_RUNTIME_DIR', tempfile.gettempdir()))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix='aic-final-transport-', suffix='.json', dir=directory, delete=False) as stream:
        json.dump(spec, stream)
        os.environ['AIC_RULE_TRANSPORT_TEMPLATE'] = stream.name
    _applied = True
