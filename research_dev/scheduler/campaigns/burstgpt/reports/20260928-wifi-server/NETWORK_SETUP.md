# Local WiFi 7 network created by the A6000 server (FCHLLX01)

Topology: WiFi 7 router in **access-point / bridge mode** (its own DHCP and NAT off) → one LAN port cabled to
FCHLLX01's free port `enp2s0`. FCHLLX01 hands out addresses on 192.168.77.0/24; the campus link on `enp1s0f0`
(172.19.52.94/26) is untouched. Phones join the router's SSID and get 192.168.77.x.

## 1. On FCHLLX01 (needs sudo; NetworkManager + dnsmasq are installed)
Option A, simplest — NetworkManager "shared" mode runs DHCP (dnsmasq) on enp2s0 and NATs the phones out through the
campus link (phones get internet; drop it with option B if campus policy forbids NAT):
```
sudo nmcli connection add type ethernet ifname enp2s0 con-name wifi7-lan \
     ipv4.method shared ipv4.addresses 192.168.77.1/24 ipv6.method disabled
sudo nmcli connection up wifi7-lan
ip -4 addr show enp2s0          # expect 192.168.77.1/24
```
Option B, isolated LAN, no NAT:
```
sudo nmcli connection add type ethernet ifname enp2s0 con-name wifi7-lan \
     ipv4.method manual ipv4.addresses 192.168.77.1/24 ipv6.method disabled
sudo nmcli connection up wifi7-lan
sudo dnsmasq --interface=enp2s0 --bind-interfaces --dhcp-range=192.168.77.50,192.168.77.150,12h --no-resolv --port=0
```
If a host firewall is active (`sudo ufw status`), allow the worker and benchmark ports on that interface only:
`sudo ufw allow in on enp2s0 to any port 7070:7100 proto tcp`.

## 2. Router
AP/bridge mode, DHCP server off, 6 GHz (or 5 GHz) WiFi 7 SSID, WPA3; MLO on if offered. Client/AP isolation OFF.

## 3. Phones
Turn WiFi on (the OnePlus 15's WiFi is currently off — `wlan0` does not exist; the Pixel's is on but not associated),
join the SSID, keep the USB cables to the desktop for power and adb control. Then report each phone's address:
`adb -s <serial> shell ip -4 -brief addr show wlan0`. For stable latency we will also test Android's low-latency
WiFi mode (`adb shell cmd wifi force-low-latency-mode enabled`) and, rooted, power-save off.

## 4. Verification I will run
- `ping -c 200 -i 0.02 <phone-ip>` from FCHLLX01 (baseline RTT).
- `wifi_rtt.py` 10 KB request/response against `toybox nc -L -p 7070 cat` on each phone: p50/p99/p99.9, idle and
  with both phones at once, with and without low-latency mode. Go if p99 ≤ 5 ms and p99.9 ≤ 15 ms per call.
