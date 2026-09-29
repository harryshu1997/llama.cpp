"""Journal the documented, RAM-only boot of the exact qualified OP15 image."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/storage/s42-phone-kernel-restore-20260910-v1')
RIG = Path('/mnt/storage/s42-cuda-graph-v1-20260909/reference-inputs/matched-rig.json')
SERIAL = '3C15AU002CL00000'
IMAGE_SHA256 = '26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d'
KERNEL = '6.12.23-android16-5-o-g227664cbe007-4k'
sequence = 0


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write('\n')


def command(args, timeout=15, required=True):
    global sequence
    sequence += 1
    started = time.time_ns()
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        record = {'command': args, 'returncode': result.returncode,
                  'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        record = {'command': args, 'returncode': None, 'failure': 'TIMEOUT',
                  'stdout': decoded(error.stdout), 'stderr': decoded(error.stderr)}
        result = None
    record.update(started_epoch_ns=started, finished_epoch_ns=time.time_ns())
    write(ROOT / 'commands' / f'{sequence:03d}.json', record)
    if required and (result is None or result.returncode != 0):
        raise RuntimeError(record)
    return record


def main():
    rig = json.loads(RIG.read_text())
    phone = rig['phone']
    assert phone['serial'] == SERIAL and phone['kernel_release'] == KERNEL
    assert phone['boot_image_sha256'] == 'sha256:' + IMAGE_SHA256
    image = Path(rig['binaries']['phone_boot_image'])
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    assert image_sha == IMAGE_SHA256, 'Qualified boot image hash mismatch'
    ROOT.mkdir()
    (ROOT / 'commands').mkdir()
    adb = [rig['binaries']['adb'], '-P', str(phone['adb_port']), '-s', SERIAL]
    fastboot = [rig['binaries']['fastboot'], '-s', SERIAL]
    before = {key: command(adb + ['shell', 'getprop', key])['stdout'].strip() for key in (
        'ro.product.device', 'ro.product.model', 'ro.build.fingerprint',
        'ro.boot.slot_suffix', 'ro.boot.flash.locked', 'ro.boot.verifiedbootstate')}
    before['kernel_release'] = command(adb + ['shell', 'uname', '-r'])['stdout'].strip()
    processes = command(adb + ['shell', 'ps', '-A'])['stdout']
    assert not any(name in processes.lower() for name in ('llama-', 'ffn-split', 'resident-router')), 'Phone workers active'
    assert before['ro.boot.flash.locked'] == '0', 'Bootloader is not already unlocked'
    write(ROOT / 'BOOT_PLAN.json', {
        'serial': SERIAL, 'method': 'fastboot boot; RAM only', 'image_path': str(image),
        'image_sha256': image_sha, 'required_kernel': KERNEL, 'before': before,
        'forbidden_operations': ['flash', 'erase', 'format', 'unlock', 'set_active'],
        'purpose': 'Restore the already-qualified kernel for the unchanged dev3 retest'})
    print('QUALIFIED_IMAGE_VERIFIED; REBOOTING_TO_BOOTLOADER', flush=True)
    command(adb + ['reboot', 'bootloader'])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        listed = command([rig['binaries']['fastboot'], 'devices'], required=False)
        if any(line.split()[:1] == [SERIAL] for line in listed['stdout'].splitlines()):
            break
        time.sleep(2)
    else:
        raise RuntimeError('Exact serial did not enumerate in fastboot; no boot or flash attempted')
    serial_reply = command(fastboot + ['getvar', 'serialno'])
    assert SERIAL in serial_reply['stdout'] + serial_reply['stderr'], 'Fastboot serial differs'
    slot_reply = command(fastboot + ['getvar', 'current-slot'])
    slot = before['ro.boot.slot_suffix'].removeprefix('_')
    assert 'current-slot: ' + slot in slot_reply['stdout'] + slot_reply['stderr'], 'Active slot changed'
    assert hashlib.sha256(image.read_bytes()).hexdigest() == image_sha
    print('EXACT_SERIAL_CONFIRMED; TEMPORARY_BOOT_BEGIN', flush=True)
    command(fastboot + ['boot', str(image)], timeout=60)
    print('TEMPORARY_BOOT_ACCEPTED; WAITING_FOR_ADB', flush=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = command(adb + ['get-state'], required=False, timeout=5)
        if state['returncode'] == 0 and state['stdout'].strip() == 'device':
            kernel = command(adb + ['shell', 'uname', '-r'])['stdout'].strip()
            completed = command(adb + ['shell', 'getprop', 'sys.boot_completed'])['stdout'].strip()
            if completed == '1':
                assert kernel == KERNEL, 'Boot completed with an unexpected kernel'
                break
        time.sleep(3)
    else:
        raise RuntimeError('Boot did not complete within the bounded wait; no flash or wipe attempted')
    after = {'kernel_release': kernel,
             'boot_completed': completed,
             'slot': command(adb + ['shell', 'getprop', 'ro.boot.slot_suffix'])['stdout'].strip(),
             'usb_config': command(adb + ['shell', 'getprop', 'sys.usb.config'])['stdout'].strip(),
             'root_identity': command(adb + ['shell', 'su', '-c', 'id'])['stdout'].strip(),
             'kernel_btf': command(adb + ['shell', "su -c 'sha256sum /sys/kernel/btf/vmlinux'"])['stdout'].strip(),
             'usb_tree': command(['lsusb', '-t'])['stdout']}
    assert after['slot'] == before['ro.boot.slot_suffix']
    assert 'uid=0' in after['root_identity']
    write(ROOT / 'RESTORE_RESULT.json', {'status': 'KERNEL_RESTORED', 'after': after,
                                        'image_sha256': image_sha, 'finished_epoch_ns': time.time_ns(),
                                        'partition_flash_count': 0, 'data_wipe_count': 0,
                                        'method': 'temporary RAM boot'})
    print('KERNEL_RESTORED', json.dumps(after, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            write(ROOT / 'FAILURE.json', {'status': 'RESTORE_FAILED', 'error': repr(error),
                                         'observed_at_epoch_ns': time.time_ns(),
                                         'partition_flash_count': 0, 'data_wipe_count': 0})
        raise
