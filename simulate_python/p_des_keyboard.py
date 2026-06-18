"""Keyboard controller for target pressure p_des.

W  — raise p_des by STEP PSI
S  — lower p_des by STEP PSI
Q  — quit

Publishes to rt/valve/pressure_des_cmd (Point_.x = PSI).
valve_pressure_node.py relays it to rt/valve/pressure_des consumed by g1_ctrl.

Run in env_isaaclab:
  conda run -n env_isaaclab python p_des_keyboard.py [--iface lo] [--p_des 100.0] [--step 5.0]
"""

import argparse
import sys
import termios
import tty

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point_

PRESSURE_DES_CMD_TOPIC = "rt/valve/pressure_des_cmd"
DOMAIN_ID = 0
P_MIN, P_MAX = 15.0, 200.0


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--p_des", type=float, default=100.0, help="initial p_des PSI")
    ap.add_argument("--step", type=float, default=5.0, help="PSI per keypress")
    args = ap.parse_args()

    ChannelFactoryInitialize(DOMAIN_ID, args.iface)
    pub = ChannelPublisher(PRESSURE_DES_CMD_TOPIC, Point_)
    pub.Init()

    msg = geometry_msgs_msg_dds__Point_()
    p_des = args.p_des

    print(f"[p_des_keyboard] p_des={p_des:.1f} PSI  step={args.step} PSI")
    print("W=raise  S=lower  Q=quit")

    while True:
        ch = getch().lower()
        if ch == 'q':
            print("\n[p_des_keyboard] quit")
            break
        elif ch == 'w':
            p_des = min(P_MAX, p_des + args.step)
        elif ch == 's':
            p_des = max(P_MIN, p_des - args.step)
        else:
            continue

        msg.x = p_des
        msg.y = 0.0
        msg.z = 0.0
        pub.Write(msg)
        print(f"\r[p_des_keyboard] p_des={p_des:.1f} PSI    ", end="", flush=True)


if __name__ == "__main__":
    main()
