from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/physical')
files = {}
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name == 'ARCHIVE_SHA256.json':
        continue
    with path.open('rb') as stream:
        files[str(path.relative_to(root))] = {
            'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
        }
result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'files': files}
with (root / 'ARCHIVE_SHA256.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({'files': len(files), 'bytes': sum(row['bytes'] for row in files.values())}))
