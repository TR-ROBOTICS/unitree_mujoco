#!/usr/bin/env python3
"""Virtual Gamepad for Unitree G1 Sim2Sim via DDS.

Publishes WirelessController_ messages on 'rt/wirelesscontroller/cmd'
to control the robot FSM transitions and velocity when no physical
joystick is available (e.g., over RDP).

Requires: unitree_sdk2_python, modified unitree_sdk2py_bridge.py

Usage:
    python3 virtual_gamepad.py [--domain 0] [--interface lo]
"""

import argparse
import time
import threading
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

# ── Button bitmask (matches BtnUnion in unitree_joystick.hpp) ──
R1     = 1 << 0
L1     = 1 << 1
START  = 1 << 2
SELECT = 1 << 3
R2     = 1 << 4
L2     = 1 << 5
F1     = 1 << 6
F2     = 1 << 7
A_BTN  = 1 << 8
B_BTN  = 1 << 9
X_BTN  = 1 << 10
Y_BTN  = 1 << 11
UP     = 1 << 12
RIGHT  = 1 << 13
DOWN   = 1 << 14
LEFT   = 1 << 15

# FSM aliases (C++ naming: LT=L2, RT=R2, LB=L1, RB=R1)
LT = L2
RB = R1

TOPIC = "rt/wirelesscontroller/cmd"
PUBLISH_HZ = 50


class VirtualGamepad:
    def __init__(self, domain_id=0, interface="lo"):
        ChannelFactoryInitialize(domain_id, interface)
        self._pub = ChannelPublisher(TOPIC, WirelessController_)
        self._pub.Init()
        self._msg = unitree_go_msg_dds__WirelessController_()

        # Continuously-published state
        self._keys = 0
        self._lx = 0.0
        self._ly = 0.0
        self._rx = 0.0
        self._ry = 0.0
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def _publish_loop(self):
        dt = 1.0 / PUBLISH_HZ
        while self._running:
            with self._lock:
                self._msg.keys = self._keys
                self._msg.lx = self._lx
                self._msg.ly = self._ly
                self._msg.rx = self._rx
                self._msg.ry = self._ry
            self._pub.Write(self._msg)
            time.sleep(dt)

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

    # ── Button sequences for FSM transitions ──

    def send_button_combo(self, hold, trigger, hold_time=0.5, trigger_time=0.3):
        """Hold some keys, then add trigger keys, then release."""
        with self._lock:
            self._keys = hold
        time.sleep(hold_time)

        with self._lock:
            self._keys = hold | trigger
        time.sleep(trigger_time)

        with self._lock:
            self._keys = 0
        time.sleep(0.1)

    def transition_fixstand(self):
        """Passive -> FixStand: LT + up.on_pressed"""
        print("  [LT + Up] Passive -> FixStand")
        self.send_button_combo(hold=LT, trigger=UP)

    def transition_velocity(self):
        """FixStand -> Velocity: RB + X.on_pressed"""
        print("  [RB + X] FixStand -> Velocity")
        self.send_button_combo(hold=RB, trigger=X_BTN)

    def transition_passive(self):
        """Any -> Passive: LT + B.on_pressed"""
        print("  [LT + B] -> Passive")
        self.send_button_combo(hold=LT, trigger=B_BTN)

    # ── Elastic band commands (via F1/F2/SELECT special keys) ──

    def elastic_lower(self):
        """Key 8 equivalent: increase elastic band length (lower robot)."""
        print("  [F1] Elastic band length +0.1 (lower)")
        with self._lock:
            self._keys = F1
        time.sleep(0.15)
        with self._lock:
            self._keys = 0
        time.sleep(0.1)

    def elastic_raise(self):
        """Key 7 equivalent: decrease elastic band length (raise robot)."""
        print("  [F2] Elastic band length -0.1 (raise)")
        with self._lock:
            self._keys = F2
        time.sleep(0.15)
        with self._lock:
            self._keys = 0
        time.sleep(0.1)

    def elastic_toggle(self):
        """Key 9 equivalent: toggle elastic band enable."""
        print("  [SELECT] Toggle elastic band")
        with self._lock:
            self._keys = SELECT
        time.sleep(0.15)
        with self._lock:
            self._keys = 0
        time.sleep(0.1)

    # ── Arm pose commands ──
    # LB = L1 (your naming), RB = R1 (your naming)

    def arm_left_up(self):
        """LB + up → lift left arm to 1.57 rad."""
        print("  [LB+Up] Lift left arm")
        self.send_button_combo(hold=L1, trigger=UP, hold_time=0.1, trigger_time=0.15)

    def arm_left_down(self):
        """LB + down → lower left arm to 0.0 rad."""
        print("  [LB+Down] Lower left arm")
        self.send_button_combo(hold=L1, trigger=DOWN, hold_time=0.1, trigger_time=0.15)

    def arm_right_up(self):
        """RB + up → lift right arm to 1.57 rad."""
        print("  [RB+Up] Lift right arm")
        self.send_button_combo(hold=R1, trigger=UP, hold_time=0.1, trigger_time=0.15)

    def arm_right_down(self):
        """RB + down → lower right arm to 0.0 rad."""
        print("  [RB+Down] Lower right arm")
        self.send_button_combo(hold=R1, trigger=DOWN, hold_time=0.1, trigger_time=0.15)

    # ── Velocity axes ──

    def set_velocity(self, ly=None, lx=None, rx=None):
        """Set joystick axes. ly=forward, -lx=strafe_left, -rx=yaw_left."""
        with self._lock:
            if ly is not None:
                self._ly = ly
            if lx is not None:
                self._lx = lx
            if rx is not None:
                self._rx = rx

    def get_velocity(self):
        with self._lock:
            return self._ly, self._lx, self._rx


def print_help():
    print("""
=== Virtual Gamepad for Unitree G1 ===

FSM transitions:
  1 / fixstand   LT+Up   : Passive -> FixStand
  2 / velocity   RB+X    : FixStand -> Velocity
  3 / passive    LT+B    : Any -> Passive

Elastic band:
  8 / lower      F1      : length +0.1 (lower robot)
  7 / raise      F2      : length -0.1 (raise robot)
  9 / toggle     SELECT  : toggle elastic band on/off

Velocity (while in Velocity mode):
  w / s          forward / backward  (ly +/- 0.2)
  a / d          strafe left / right (lx +/- 0.2)
  q / e          yaw left / right    (rx +/- 0.2)
  0 / stop       zero all axes
  vx <val>       set forward velocity directly
  vy <val>       set lateral velocity directly
  vyaw <val>     set yaw rate directly

Arm pose (toggle, holds until opposite command):
  lu             LB+Up   : lift left arm
  ld             LB+Down : lower left arm
  ru             RB+Up   : lift right arm
  rd             RB+Down : lower right arm

Other:
  help           show this help
  status         show current axes
  quit / exit    exit
""")


def main():
    parser = argparse.ArgumentParser(description="Virtual Gamepad for Unitree G1")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--interface", type=str, default="lo", help="Network interface")
    args = parser.parse_args()

    gp = VirtualGamepad(args.domain, args.interface)
    print_help()

    VEL_STEP = 0.2

    try:
        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                break

            parts = cmd.lower().split()
            if not parts:
                continue
            c = parts[0]

            # ── FSM transitions ──
            if c in ("1", "fixstand"):
                gp.transition_fixstand()

            elif c in ("2", "velocity"):
                gp.transition_velocity()

            elif c in ("3", "passive"):
                gp.transition_passive()

            # ── Elastic band ──
            elif c in ("8", "lower"):
                gp.elastic_lower()

            elif c in ("7", "raise"):
                gp.elastic_raise()

            elif c in ("9", "toggle"):
                gp.elastic_toggle()

            # ── Quick velocity ──
            elif c == "w":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(ly=min(ly + VEL_STEP, 1.0))
                ly, _, _ = gp.get_velocity()
                print(f"  forward={ly:.2f}")

            elif c == "s":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(ly=max(ly - VEL_STEP, -1.0))
                ly, _, _ = gp.get_velocity()
                print(f"  forward={ly:.2f}")

            elif c == "a":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(lx=max(lx - VEL_STEP, -1.0))
                _, lx, _ = gp.get_velocity()
                print(f"  lx={lx:.2f} (strafe_left={-lx:.2f})")

            elif c == "d":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(lx=min(lx + VEL_STEP, 1.0))
                _, lx, _ = gp.get_velocity()
                print(f"  lx={lx:.2f} (strafe_left={-lx:.2f})")

            elif c == "q":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(rx=max(rx - VEL_STEP, -1.0))
                _, _, rx = gp.get_velocity()
                print(f"  rx={rx:.2f} (yaw_left={-rx:.2f})")

            elif c == "e":
                ly, lx, rx = gp.get_velocity()
                gp.set_velocity(rx=min(rx + VEL_STEP, 1.0))
                _, _, rx = gp.get_velocity()
                print(f"  rx={rx:.2f} (yaw_left={-rx:.2f})")

            elif c in ("0", "stop"):
                gp.set_velocity(ly=0.0, lx=0.0, rx=0.0)
                print("  Stopped (all axes = 0)")

            # ── Direct velocity set ──
            elif c == "vx" and len(parts) > 1:
                val = float(parts[1])
                gp.set_velocity(ly=val)
                print(f"  forward={val:.2f}")

            elif c == "vy" and len(parts) > 1:
                val = float(parts[1])
                gp.set_velocity(lx=-val)  # -lx = vy
                print(f"  strafe_left={val:.2f} (lx={-val:.2f})")

            elif c == "vyaw" and len(parts) > 1:
                val = float(parts[1])
                gp.set_velocity(rx=-val)  # -rx = yaw
                print(f"  yaw_left={val:.2f} (rx={-val:.2f})")

            elif c == "status":
                ly, lx, rx = gp.get_velocity()
                print(f"  ly(fwd)={ly:.2f}  lx={lx:.2f}  rx={rx:.2f}")
                print(f"  -> lin_vel_x~{ly:.2f}  lin_vel_y~{-lx:.2f}  ang_vel_z~{-rx:.2f}")

            elif c == "help":
                print_help()

            # ── Arm pose ──
            elif c == "lu":
                gp.arm_left_up()

            elif c == "ld":
                gp.arm_left_down()

            elif c == "ru":
                gp.arm_right_up()

            elif c == "rd":
                gp.arm_right_down()

            elif c in ("quit", "exit"):
                break

            else:
                print(f"  Unknown: {cmd}  (type 'help')")

    except KeyboardInterrupt:
        pass

    gp.set_velocity(ly=0, lx=0, rx=0)
    time.sleep(0.1)
    gp.stop()
    print("\nBye.")


if __name__ == "__main__":
    main()
