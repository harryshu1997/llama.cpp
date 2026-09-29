"""Run the existing backend test executable under the rig's idle/ownership lock."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, '/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2')
from src.run_direct_usb_device import ADB, execution_lock, preflight, shell


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--binary-dir', required=True)
parser.add_argument('--fusion', type=int, default=1)
args = parser.parse_args()
args.output.mkdir()
binary = args.binary_dir + '/test-backend-ops'
command = ['taskset', 'c0', 'env', 'LD_LIBRARY_PATH=' + args.binary_dir,
           'ADSP_LIBRARY_PATH=' + args.binary_dir + ';/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp',
           'GGML_HEXAGON_PROFILE=1', 'GGML_HEXAGON_NHVX=4', 'GGML_HEXAGON_OPFUSION=' + str(args.fusion),
           binary, 'test', '-b', 'HTP0', '-o', 'MUL_MAT_VEC_FUSION', '-p', 'k=(3840|5120),']
save(args.output / 'COMMAND.json', command)
with execution_lock():
    idle = args.output / 'idle'
    idle.mkdir()
    boot = preflight(idle)
    save(args.output / 'HASHES.json', shell('sha256sum ' + shlex.quote(binary) + ' ' +
                                          shlex.quote(args.binary_dir) + '/lib*.so'))
    with (args.output / 'test.log').open('x') as log:
        remote = 'echo TEST_PID=$$; exec ' + shlex.join(command)
        process = subprocess.Popen(ADB + ['shell', '-T', 'su -mm -c ' + shlex.quote(remote)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=600)
        finally:
            if process.poll() is None:
                lines = (args.output / 'test.log').read_text().splitlines()
                pid = next((line.split('=', 1)[1] for line in lines if line.startswith('TEST_PID=')), '')
                if pid.isdigit() and binary in shell('cat /proc/' + pid + '/cmdline'):
                    shell('kill -TERM ' + pid)
                process.wait(timeout=10)
            after = args.output / 'postflight'
            after.mkdir()
            assert preflight(after) == boot
    save(args.output / 'RESULT.json', {'returncode': code})
    print(args.output, code, flush=True)
    raise SystemExit(code)
