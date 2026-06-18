"""Live valve pressure plot (+ optional real-time CSV log).

Subscribes `rt/valve/pressure` (Point_.x(), PSI) and `rt/valve/pressure_des`
and draws both as scrolling time-series lines. p_des steps are visible as
a staircase when W/S are pressed in g1_ctrl.

With --csv, every p_now sample is also appended to a CSV file, flushed
per-row so the log is durable in real time.

CSV columns: wall_iso, t_s, pressure_psi

Run in env_isaaclab (cyclonedds + unitree_sdk2py + matplotlib):
  conda run -n env_isaaclab python plot_pressure.py [--iface lo] [--window 30] [--csv pressure.csv]
"""

import argparse
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_

PRESSURE_TOPIC = "rt/valve/pressure"
PRESSURE_DES_TOPIC = "rt/valve/pressure_des"
DOMAIN_ID = 0
P_MIN, P_MAX = 15.0, 200.0


class PressurePlot:
    def __init__(self, window_s: float, csv_path: str | None = None):
        self.window_s = window_s
        self.t0 = time.monotonic()

        self.t = deque()
        self.p = deque()

        self.t_des = deque()
        self.p_des = deque()

        self._csv = None
        if csv_path:
            # buffering=1 -> line-buffered; flush() per row makes it real-time durable
            self._csv = open(csv_path, "w", buffering=1)
            self._csv.write("wall_iso,t_s,pressure_psi\n")
            print(f"[plot_pressure] logging CSV -> {csv_path}")

        self._sub = ChannelSubscriber(PRESSURE_TOPIC, Point_)
        self._sub.Init(self._on_pressure, 10)

        self._des_sub = ChannelSubscriber(PRESSURE_DES_TOPIC, Point_)
        self._des_sub.Init(self._on_pressure_des, 10)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], lw=2, label="p_now")
        (self.line_des,) = self.ax.plot([], [], lw=1.5, color="r", linestyle="--", label="p_des")
        self.ax.legend(loc="upper left")
        self.ax.set_xlabel("t [s]")
        self.ax.set_ylabel("pressure [PSI]")
        self.ax.set_ylim(P_MIN - 5, P_MAX + 5)
        self.ax.grid(True)

    def _on_pressure(self, msg: Point_):
        now = time.monotonic() - self.t0
        p = msg.x
        self.t.append(now)
        self.p.append(p)
        while self.t and self.t[0] < now - self.window_s:
            self.t.popleft()
            self.p.popleft()
        if self._csv is not None:
            self._csv.write(f"{datetime.now().isoformat()},{now:.4f},{p:.4f}\n")
            self._csv.flush()

    def _on_pressure_des(self, msg: Point_):
        now = time.monotonic() - self.t0
        p = msg.x
        self.t_des.append(now)
        self.p_des.append(p)
        while self.t_des and self.t_des[0] < now - self.window_s:
            self.t_des.popleft()
            self.p_des.popleft()

    def spin(self):
        while plt.fignum_exists(self.fig.number):
            if self.t:
                self.line.set_data(self.t, self.p)
                self.ax.set_xlim(max(0, self.t[-1] - self.window_s), self.t[-1] + 0.5)
                title = f"p_now={self.p[-1]:.1f} PSI"
                if self.t_des:
                    self.line_des.set_data(self.t_des, self.p_des)
                    title += f"  p_des={self.p_des[-1]:.1f} PSI"
                self.ax.set_title(title)
            self.fig.canvas.draw_idle()
            plt.pause(0.05)

    def close(self):
        if self._csv is not None:
            self._csv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo", help="DDS network interface")
    ap.add_argument("--window", type=float, default=30.0, help="plot window [s]")
    ap.add_argument("--csv", default=None, help="CSV log path (optional)")
    args = ap.parse_args()

    ChannelFactoryInitialize(DOMAIN_ID, args.iface)
    plot = PressurePlot(args.window, args.csv)
    print(f"[plot_pressure] subscribing {PRESSURE_TOPIC} + {PRESSURE_DES_TOPIC} on {args.iface}")
    try:
        plot.spin()
    except KeyboardInterrupt:
        print("\n[plot_pressure] stop")
    finally:
        plot.close()


if __name__ == "__main__":
    main()
