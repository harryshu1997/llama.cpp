#!/usr/bin/env bash
set -euo pipefail

candidate_root=/home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo
reference_root=/home/myid/zs89458/Documents/s41-op15-kernel-rx-v1
candidate_platform=$candidate_root/source/kernel_platform
candidate_bin=$candidate_platform/out/bazel/output_user_root/d8d2b9f915a3781de209936d292c2394/execroot/_main/bazel-out/k8-fastbuild/bin/common
reference_bin=$reference_root/kernel_platform/out/bazel/output_user_root/818ad93610d1fad156d4ff6675da326c/execroot/_main/bazel-out/k8-fastbuild/bin/common
candidate_kernel=$candidate_bin/kernel_aarch64
reference_kernel=$reference_bin/kernel_aarch64
clang_bin=$candidate_platform/prebuilts/clang/host/linux-x86/clang-r536225/bin
candidate_config=$candidate_bin/kernel_aarch64_config/out_dir/.config
verification=$candidate_root/verification-v2
patch_file=/home/myid/zs89458/Documents/moe-resident-routing-a6000/artifacts/op15_kernel_fix_20260913/fence_lifetime.patch

test ! -e "$candidate_root/VERIFY-v2.log"
test -s "$candidate_bin/kernel_aarch64_gki_artifacts/boot.img"
mkdir "$verification"

verify() {
    set -x
    git -C "$candidate_platform/common" apply --reverse --check "$patch_file"
    cmp "$reference_root/s41_dist/ffs_dmabuf_fixed_v1/kernel.config" "$candidate_config"
    cmp "$reference_kernel/Module.symvers" "$candidate_kernel/Module.symvers"
    cmp "$reference_kernel/include/config/kernel.release" "$candidate_kernel/include/config/kernel.release"
    cmp "$reference_kernel/modules.builtin" "$candidate_kernel/modules.builtin"
    test "$(stat -c %s "$candidate_bin/kernel_aarch64_gki_artifacts/boot.img")" -eq 67108864
    "$clang_bin/llvm-readelf" --notes "$candidate_kernel/vmlinux"
    "$clang_bin/llvm-objcopy" --dump-section .BTF="$verification/reference.btf" --dump-section .notes="$verification/reference.notes" "$reference_kernel/vmlinux" "$verification/reference.elf"
    "$clang_bin/llvm-objcopy" --dump-section .BTF="$verification/candidate.btf" --dump-section .notes="$verification/candidate.notes" "$candidate_kernel/vmlinux" "$verification/candidate.elf"
    test "$(sha256sum "$verification/reference.btf" | cut -d ' ' -f 1)" = f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17
    test "$(sha256sum "$verification/reference.notes" | cut -d ' ' -f 1)" = 6bb2e05d4843db26abb034e1720a4b4b5b7de7cab488e015193e07d91b462bf1
    python3 "$candidate_platform/tools/mkbootimg/unpack_bootimg.py" --boot_img "$reference_root/s41_dist/ffs_dmabuf_fixed_v1/boot.img" --out "$verification/reference-boot" > "$verification/reference-boot.txt"
    python3 "$candidate_platform/tools/mkbootimg/unpack_bootimg.py" --boot_img "$candidate_bin/kernel_aarch64_gki_artifacts/boot.img" --out "$verification/candidate-boot" > "$verification/candidate-boot.txt"
    diff -u <(sed '/^kernel_size: /d' "$verification/reference-boot.txt") <(sed '/^kernel_size: /d' "$verification/candidate-boot.txt")
    diff -u "$verification/reference-boot.txt" "$verification/candidate-boot.txt" || test "$?" -eq 1
    cmp "$verification/reference-boot/ramdisk" "$verification/candidate-boot/ramdisk"
    cmp "$candidate_kernel/Image" "$verification/candidate-boot/kernel"
    sha256sum "$patch_file" "$reference_root/kernel_platform/common/drivers/usb/gadget/function/f_fs.c" "$candidate_platform/common/drivers/usb/gadget/function/f_fs.c" "$reference_root/s41_dist/ffs_dmabuf_fixed_v1/boot.img" "$candidate_bin/kernel_aarch64_gki_artifacts/boot.img" "$candidate_kernel/Image" "$candidate_config" "$candidate_kernel/Module.symvers" "$verification/candidate.btf" "$verification/candidate.notes"
    printf '%s\n' 'BUILD_COMPATIBILITY_CHECKS_PASSED; DEVICE_QUALIFICATION_NOT_PERFORMED'
}

verify 2>&1 | tee "$candidate_root/VERIFY-v2.log"
