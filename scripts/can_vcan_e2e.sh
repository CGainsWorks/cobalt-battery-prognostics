#!/usr/bin/env bash
# The CAN link end to end over a Linux vcan interface, as two processes.
#
#   bash scripts/can_vcan_e2e.sh
#
# The test suite already runs the simulated BMS node and the receiver together
# on python-can's in-process virtual bus. This runs them as separate processes
# through the kernel's SocketCAN stack on a virtual CAN interface, which is the
# same code path a physical SocketCAN adapter would use, minus the wire.
#
# Creating the interface needs root and the vcan kernel module. Where either is
# missing (macOS, Windows, a container without privileges, a runner whose
# kernel ships without the module) this says so and exits 0, unless
# REQUIRE_VCAN=1, because the virtual-bus tests have covered the protocol and
# a missing kernel module is not a defect in this repository.
#
# Both ends are simulated. No hardware is involved.

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/_python.sh
. scripts/_python.sh

IFACE="${IFACE:-vcan0}"
OUT="${OUT:-artifacts/ci/stream_vcan.json}"
EXPECTED="${EXPECTED:-artifacts/default/stream_demo_can.json}"

skip() {
    echo "::notice::vcan end to end skipped: $1. The virtual-bus tests in tests/test_can.py still ran."
    if [ "${REQUIRE_VCAN:-0}" = "1" ]; then
        exit 1
    fi
    exit 0
}

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo > /dev/null || skip "not root and no sudo"
    SUDO="sudo -n"
fi
command -v ip > /dev/null || skip "no ip command (iproute2)"

if ! ip link show "$IFACE" > /dev/null 2>&1; then
    # Some kernels have vcan built in, so try the interface before the module.
    if ! $SUDO ip link add dev "$IFACE" type vcan 2> /dev/null; then
        if ! $SUDO modprobe vcan 2> /dev/null; then
            # Hosted Ubuntu runners ship the module in linux-modules-extra.
            if [ "${CI:-}" = "true" ] && command -v apt-get > /dev/null; then
                echo "==> installing linux-modules-extra-$(uname -r) for the vcan module"
                $SUDO apt-get install -y -qq "linux-modules-extra-$(uname -r)" > /dev/null 2>&1 \
                    || skip "linux-modules-extra-$(uname -r) is not installable here"
                $SUDO modprobe vcan 2> /dev/null || skip "the vcan module will not load"
            else
                skip "the vcan kernel module is not available"
            fi
        fi
        $SUDO ip link add dev "$IFACE" type vcan || skip "cannot create $IFACE"
    fi
    $SUDO ip link set up "$IFACE"
fi
echo "==> $IFACE is up"
ip -details link show "$IFACE"

mkdir -p "$(dirname "$OUT")"

# The node waits so the receiver's socket is bound before the first frame: a
# SocketCAN socket only sees frames sent after it was opened. The frame gap
# keeps the receive buffer from overflowing while the receiver decodes.
"$PYTHON" -m cobalt.cli bms-node --can-interface socketcan --can-channel "$IFACE" \
    --start-delay 3 --frame-gap-ms 0.5 &
NODE=$!
trap 'kill "$NODE" 2> /dev/null || true' EXIT

"$PYTHON" -m cobalt.cli stream --config config/default.yaml --source can \
    --can-interface socketcan --can-channel "$IFACE" --external-node \
    --first-frame-timeout 30 --idle-timeout 3 --output "$OUT"
wait "$NODE"

"$PYTHON" scripts/compare_stream_logs.py "$EXPECTED" "$OUT"
