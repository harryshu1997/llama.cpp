"""Bounded real-weight FFN protocol probe; no scheduler or residency policy."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, '/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2')
from src.run_direct_usb_device import ADB, execution_lock, preflight, shell


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def receive(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.recv(count - len(data))
        if not part:
            raise RuntimeError('worker disconnected')
        data.extend(part)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary-dir', required=True)
    parser.add_argument('--profile', type=int, default=1)
    parser.add_argument('--fusion', type=int, default=1)
    parser.add_argument('--hvx-threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 4, 137, 512])
    parser.add_argument('--columns', type=int, nargs='+', default=[3840, 7680, 11520, 15360])
    args = parser.parse_args()
    args.output.mkdir()
    parent = 'ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf'
    shard = '/data/local/tmp/s42-ffn-shards-20260904-v2/gemma24/HTP0.ffn.gguf'
    worker = args.binary_dir + '/llama-ffn-split-worker'
    cases = [(rows, columns, rep) for rows in args.rows
             for columns in args.columns
             for rep in range(args.repeats)]
    environment = {'LD_LIBRARY_PATH': args.binary_dir,
                   'ADSP_LIBRARY_PATH': args.binary_dir + ';/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp',
                   'GGML_HEXAGON_PROFILE': str(args.profile),
                   'GGML_HEXAGON_OPFUSION': str(args.fusion),
                   'GGML_HEXAGON_NHVX': str(args.hvx_threads),
                   'S42_RESIDENCY_SESSION_ID': 'HTP0',
                   'S42_RESIDENCY_SESSION_GENERATION': '1'}
    command = ['taskset', 'c0', 'env', *[k + '=' + v for k, v in environment.items()],
               worker, '-m', shard, '--artifact-sha256', 'sha256:' + parent,
               '--layer', '0', '--columns', '15360', '--column-quantum', '1280',
               '--backend', 'HTP0', '--port', '19643', '--bind', '127.0.0.1',
               '--f16-io', '--max-tokens', '512', '--max-requests', str(len(cases))]
    save(args.output / 'COMMAND.json', command)
    records = []
    with execution_lock():
        idle = args.output / 'idle'
        idle.mkdir()
        boot = preflight(idle)
        hashes = shell('sha256sum ' + shlex.quote(worker) + ' ' +
                       shlex.quote(args.binary_dir) + '/libggml*.so ' + shlex.quote(shard), timeout=120)
        save(args.output / 'IDENTITY.json', {'boot': boot, 'hashes': hashes})
        forward = subprocess.check_output(ADB + ['forward', 'tcp:0', 'tcp:19643'], text=True).strip()
        process = None
        stream = None
        try:
            with (args.output / 'worker.log').open('x') as log:
                remote = 'echo PROFILE_PID=$$; exec ' + shlex.join(command)
                process = subprocess.Popen(ADB + ['shell', '-T', 'su -mm -c ' + shlex.quote(remote)],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('worker failed during startup')
                    if '[ffn-worker] ready backend=' in (args.output / 'worker.log').read_text():
                        break
                    time.sleep(0.25)
                else:
                    raise TimeoutError('worker startup')
                stream = socket.create_connection(('127.0.0.1', int(forward)), timeout=30)
                stream.settimeout(30)
                stream.sendall(struct.pack('<IHHQIIHH32s4x', 0x46534631, 6, 1, 1, 3840,
                                           15360, 1, 512, bytes.fromhex(parent)))
                hello = receive(stream, 96)
                save(args.output / 'HELLO.json', {'hex': hello.hex()})
                assert struct.unpack_from('<IHHH', hello) == (0x46534631, 6, 2, 0)
                assert hello[64:] == bytes.fromhex(parent)
                for ident, (rows, columns, rep) in enumerate(cases, 1):
                    data = np.random.default_rng(1234 + rows).normal(0, 0.1, rows * 3840).astype('<f2').tobytes()
                    request = struct.pack('<IHHIiIIIII', 0x46534631, 6, 3, ident, 0,
                                          rows * 3840, len(data), fnv(data), columns, rows)
                    started = time.monotonic_ns()
                    stream.sendall(request + data)
                    header = struct.unpack('<IHHHHIiIIIIIQ', receive(stream, 48))
                    result = receive(stream, header[8])
                    rpc_us = (time.monotonic_ns() - started) / 1000
                    assert header[:4] == (0x46534631, 6, 4, 0)
                    assert (header[5], header[6], header[7], header[10], header[11]) == (ident, 0, rows * 3840, columns, rows)
                    assert header[9] == fnv(result)
                    values = np.frombuffer(result, dtype='<f2')
                    assert np.isfinite(values).all()
                    with (args.output / f'output-{ident}.npy').open('xb') as array:
                        np.save(array, values)
                    record = {'id': ident, 'rows': rows, 'columns': columns, 'repeat': rep,
                              'rpc_us': rpc_us, 'compute_us': header[12],
                              'input_sha256': hashlib.sha256(data).hexdigest(),
                              'output_sha256': hashlib.sha256(result).hexdigest()}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                stream.close()
                stream = None
                assert process.wait(timeout=20) == 0
        except BaseException as error:
            save(args.output / 'FAILURE.json', {'error': repr(error), 'completed': records})
            raise
        finally:
            if stream:
                stream.close()
            if process and process.poll() is None:
                lines = (args.output / 'worker.log').read_text().splitlines()
                pid = next((line.split('=', 1)[1] for line in lines if line.startswith('PROFILE_PID=')), '')
                if pid.isdigit() and worker in shell('cat /proc/' + pid + '/cmdline'):
                    shell('kill -TERM ' + pid)
                process.wait(timeout=10)
            subprocess.run(ADB + ['forward', '--remove', 'tcp:' + forward], check=True)
            save(args.output / 'RESULT.json', records)
            after = args.output / 'postflight'
            after.mkdir()
            assert preflight(after) == boot


if __name__ == '__main__':
    main()
