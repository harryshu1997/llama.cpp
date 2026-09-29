"""Record task-local before-images and the complete native build source inventory."""

import hashlib
import json
from pathlib import Path
import subprocess

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[5]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (REPORT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


changes = []
patches = []
for snapshot in sorted((REPORT / 'source-final').rglob('*')):
    if not snapshot.is_file():
        continue
    relative = snapshot.relative_to(REPORT / 'source-final')
    current, before = REPO / relative, REPORT / 'before' / relative
    assert sha(current) == sha(snapshot), relative
    result = subprocess.run(['diff', '-u', '--label', 'before/' + str(relative),
                             '--label', 'after/' + str(relative),
                             str(before) if before.exists() else '/dev/null', str(current)],
                            capture_output=True, text=True)
    assert result.returncode == 1, (relative, result.stderr)
    patches.append(result.stdout)
    changes.append({'path': str(relative), 'kind': 'modified' if before.exists() else 'added',
                    'before_sha256': sha(before) if before.exists() else None,
                    'after_sha256': sha(current),
                    'added_lines': sum(line.startswith('+') and not line.startswith('+++')
                                       for line in result.stdout.splitlines()),
                    'removed_lines': sum(line.startswith('-') and not line.startswith('---')
                                         for line in result.stdout.splitlines())})
with (REPORT / 'IMPLEMENTATION.patch').open('x') as stream:
    stream.write(''.join(patches))
save('CHANGES.json', {'schema': 's42-fp16-ffn-fusion-changes-v1', 'files': changes,
                      'patch_sha256': sha(REPORT / 'IMPLEMENTATION.patch'),
                      'no_scheduler_or_worker_protocol_changes': True})
paths = subprocess.check_output(['rg', '--files', 'ggml', 'src', 'include', 'common', 'cmake',
                                  'examples/layersplit', 'tests/test-backend-ops.cpp',
                                  'CMakeLists.txt', 'tests/CMakeLists.txt'], cwd=REPO, text=True).splitlines()
save('BUILD_SOURCE_MANIFEST.json', {
    'schema': 's42-native-build-source-inventory-v1',
    'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'dirty_tree': True,
    'source_sha256': {name: sha(REPO / name) for name in sorted(set(paths))},
    'cmake_cache_sha256': sha(REPO / 'build-ffn-fused-android/CMakeCache.txt'),
    'build_ninja_sha256': sha(REPO / 'build-ffn-fused-android/build.ninja'),
    'toolchain_image': 'snapdragon-toolchain-hostgcc:v0.3',
    'toolchain_image_id': subprocess.check_output(['docker', 'image', 'inspect',
        'snapdragon-toolchain-hostgcc:v0.3', '--format', '{{.Id}}'], text=True).strip(),
})
print(len(changes), 'implementation files', len(paths), 'native source inputs')
