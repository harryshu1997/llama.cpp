#!/system/bin/sh
# Recreate the FunctionFS gadget skeleton /config/usb_gadget/g2 after a phone reboot.
# Values are the ones read from the live g2 on 2026-09-25 before any reboot. The session script
# (direct_phone_ffn_session.sh) creates functions/ffs.s41 (+ncm), links them into configs/b.1 and
# binds the UDC itself, so only the device-level skeleton is recreated here. Idempotent.
set -e
g2=/config/usb_gadget/g2
mkdir -p "$g2/strings/0x409" "$g2/configs/b.1/strings/0x409"
printf '0x18d1' > "$g2/idVendor"
printf '0x2d00' > "$g2/idProduct"
printf '0x0320' > "$g2/bcdUSB"
printf '0x0100' > "$g2/bcdDevice"
printf '0x00'   > "$g2/bDeviceClass"
printf 'super-speed-plus' > "$g2/max_speed"
printf 'SCHEDFFN0001' > "$g2/strings/0x409/serialnumber"
printf 'Heterogeneous inference' > "$g2/strings/0x409/manufacturer"
printf 'Direct FFN DMA-BUF' > "$g2/strings/0x409/product"
printf '500'  > "$g2/configs/b.1/MaxPower"
printf '0x80' > "$g2/configs/b.1/bmAttributes"
printf 'ffn_htp_dmabuf' > "$g2/configs/b.1/strings/0x409/configuration"
echo "g2 recreated:"; for f in idVendor idProduct bcdUSB bcdDevice bDeviceClass max_speed strings/0x409/serialnumber strings/0x409/manufacturer strings/0x409/product configs/b.1/MaxPower configs/b.1/bmAttributes configs/b.1/strings/0x409/configuration; do echo "  $f=$(cat $g2/$f)"; done
