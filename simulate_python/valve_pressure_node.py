"""Valve pressure republisher.

Subscribes the valve hinge angle theta (rad) published by the sim2sim bridge on
`rt/valve/angle` (Point_.x()), maps it to gauge pressure via g(theta), and
publishes pressure (PSI) on `rt/valve/pressure` (Point_.x()).

Also publishes the current target pressure on `rt/valve/pressure_des` (Point_.x()).
Initially set via --p_des arg; future: overwritten by a subscriber on
`rt/valve/pressure_des_cmd` so the target can be changed at runtime.

This is the central pressure source: any consumer (live plot, logger, vision
surrogate) subscribes `rt/valve/pressure` instead of recomputing g(theta).

g(theta):  p = a*theta + b, clamped to [P_MIN, P_MAX]
  a = 4.527 PSI/rad, b = -27.66 PSI   (CONTEXT.md §g(theta))

Run in the env_isaaclab conda env (has cyclonedds + unitree_sdk2py):
  conda run -n env_isaaclab python valve_pressure_node.py [--iface lo] [--p_des 100.0]
Default interface "lo", domain 0 — matches simulate/config.yaml.
"""

import argparse
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point_

# g(theta) coefficients — CONTEXT.md §g(theta)
A = 4.527    # PSI/rad
B = -27.66   # PSI
P_MIN = 15.0
P_MAX = 200.0

ANGLE_TOPIC = "rt/valve/angle"
PRESSURE_TOPIC = "rt/valve/pressure"
PRESSURE_DES_TOPIC = "rt/valve/pressure_des"
PRESSURE_DES_CMD_TOPIC = "rt/valve/pressure_des_cmd"
DOMAIN_ID = 0


def g(theta: float) -> float:
    return max(P_MIN, min(P_MAX, A * theta + B))


class ValvePressureNode:
    def __init__(self, p_des: float):
        self._p_des = p_des

        self._msg = geometry_msgs_msg_dds__Point_()
        self._pub = ChannelPublisher(PRESSURE_TOPIC, Point_)
        self._pub.Init()

        self._des_msg = geometry_msgs_msg_dds__Point_()
        self._des_pub = ChannelPublisher(PRESSURE_DES_TOPIC, Point_)
        self._des_pub.Init()

        self._sub = ChannelSubscriber(ANGLE_TOPIC, Point_)
        self._sub.Init(self._on_angle, 10)

        # allow runtime updates to p_des
        self._des_cmd_sub = ChannelSubscriber(PRESSURE_DES_CMD_TOPIC, Point_)
        self._des_cmd_sub.Init(self._on_des_cmd, 10)

        self.last_p = None

    def _on_angle(self, msg: Point_):
        p = g(msg.x)
        self._msg.x = p
        self._msg.y = 0.0
        self._msg.z = 0.0
        self._pub.Write(self._msg)
        self.last_p = p

        self._des_msg.x = self._p_des
        self._des_msg.y = 0.0
        self._des_msg.z = 0.0
        self._des_pub.Write(self._des_msg)

    def _on_des_cmd(self, msg: Point_):
        self._p_des = max(P_MIN, min(P_MAX, msg.x))
        print(f"\n[valve_pressure] p_des updated -> {self._p_des:.1f} PSI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo", help="DDS network interface")
    ap.add_argument("--p_des", type=float, default=100.0, help="initial target pressure PSI")
    # legacy positional arg support
    ap.add_argument("iface_pos", nargs="?", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    iface = args.iface_pos if args.iface_pos is not None else args.iface
    ChannelFactoryInitialize(DOMAIN_ID, iface)
    node = ValvePressureNode(args.p_des)
    print(f"[valve_pressure] {ANGLE_TOPIC} -> g(theta) -> {PRESSURE_TOPIC} on {iface}")
    print(f"[valve_pressure] p_des = {args.p_des:.1f} PSI -> {PRESSURE_DES_TOPIC}")
    print(f"[valve_pressure] runtime p_des updates via {PRESSURE_DES_CMD_TOPIC}")
    try:
        while True:
            time.sleep(1.0)
            if node.last_p is not None:
                print(f"[valve_pressure] p = {node.last_p:6.2f} PSI  p_des = {node._p_des:6.2f} PSI", end="\r")
    except KeyboardInterrupt:
        print("\n[valve_pressure] stop")


if __name__ == "__main__":
    main()
