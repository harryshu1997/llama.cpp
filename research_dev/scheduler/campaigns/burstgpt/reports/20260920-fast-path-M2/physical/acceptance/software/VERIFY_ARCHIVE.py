from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
manifest = json.loads((root / 'ARCHIVE_SHA256.json').read_text())
remaining = dict(manifest['files'])
verified = {}
started = time.monotonic()
while remaining:
    for name, record in list(remaining.items()):
        path = root / name
        if not path.is_file() or path.stat().st_size != record['bytes']:
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != record['sha256']:
            raise RuntimeError('Archive hash mismatch: ' + name)
        verified[name] = record
        del remaining[name]
        if record['bytes'] > 100_000_000:
            print('Verified', name, flush=True)
    if remaining:
        if time.monotonic() - started > 2400:
            raise TimeoutError('Archive still incomplete: ' + ', '.join(remaining))
        time.sleep(5)
result = {'status': 'PASS', 'at_utc': datetime.now(timezone.utc).isoformat(),
    'source': manifest['root'], 'files': len(verified),
    'bytes': sum(row['bytes'] for row in verified.values()),
    'manifest_sha256': hashlib.sha256((root / 'ARCHIVE_SHA256.json').read_bytes()).hexdigest()}
with (root / 'ARCHIVE_VERIFIED.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result), flush=True)
