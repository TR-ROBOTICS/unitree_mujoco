"""Sim2sim pressure-tracking RMSE bench.

Drives p_des (rt/valve/pressure_des_cmd) with step/sine/cosine/jitter profiles,
logs realized p_now (rt/valve/pressure) vs applied p_des (rt/valve/pressure_des),
and reports tracking RMSE + settling time.

NOTE: RMSE here is a COMBINED controller+physics tracking error, NOT a clean
sim2sim-gap isolator (g(theta) is deterministic; deadband HOLD lives in the C++
controller; overshoot also depends on MuJoCo contact tuning). Read-only metric:
never used to tune MJCF.

Usage:
  # offline self-check (no DDS / no sim)
  conda run -n env_isaaclab python sim2sim_tracking_bench.py --selftest
  # against a live sim (valve_pressure_node.py + controller running)
  conda run -n env_isaaclab python sim2sim_tracking_bench.py \
      --signal step --delta 50 --duration 20 --csv step50.csv
"""
import argparse
import csv
import math
import random
import time

A, B = 4.527, -27.66          # g(theta) = A*theta + B  (CONTEXT.md §g(theta))
P_MIN, P_MAX = 15.0, 200.0

PRESSURE_TOPIC = "rt/valve/pressure"
PRESSURE_DES_TOPIC = "rt/valve/pressure_des"
PRESSURE_DES_CMD_TOPIC = "rt/valve/pressure_des_cmd"
DOMAIN_ID = 0


def p_des_at(t, signal, *, base, delta, t_step, amp, freq, jitter_period, seed):
    if signal == "step":
        return base + (delta if t >= t_step else 0.0)
    if signal == "sine":
        return base + amp * math.sin(2 * math.pi * freq * t)
    if signal == "cosine":
        return base + amp * math.cos(2 * math.pi * freq * t)
    if signal == "jitter":
        # piecewise-constant: same value within each jitter_period window,
        # deterministic given seed so runs/tests reproduce
        bucket = int(t // jitter_period)
        rng = random.Random(seed * 1_000_003 + bucket)
        return base + rng.uniform(-amp, amp)
    raise ValueError(f"unknown signal {signal!r}")


def rmse(p_des, p_now):
    n = min(len(p_des), len(p_now))
    if n == 0:
        return float("nan")
    return math.sqrt(sum((d - p) ** 2 for d, p in zip(p_des[:n], p_now[:n])) / n)


def settling_time(t, p_now, target, t_step, band):
    # earliest sample (>= t_step) from which all later samples stay within band
    enter = None
    for ti, pi in zip(t, p_now):
        if ti < t_step:
            continue
        if abs(pi - target) <= band:
            if enter is None:
                enter = ti
        else:
            enter = None  # left the band, reset
    return enter


def run(args):
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_
    from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point_

    ChannelFactoryInitialize(DOMAIN_ID, args.iface)
    cmd_pub = ChannelPublisher(PRESSURE_DES_CMD_TOPIC, Point_)
    cmd_pub.Init()

    state = {"p_now": None, "p_des": args.base}
    rows = []  # (t_s, p_des_applied, p_now)

    def on_pressure(msg):
        state["p_now"] = msg.x

    def on_des(msg):
        state["p_des"] = msg.x

    p_sub = ChannelSubscriber(PRESSURE_TOPIC, Point_)
    p_sub.Init(on_pressure, 10)
    d_sub = ChannelSubscriber(PRESSURE_DES_TOPIC, Point_)
    d_sub.Init(on_des, 10)

    msg = geometry_msgs_msg_dds__Point_()
    dt = 1.0 / args.rate
    t0 = time.monotonic()
    print(f"[bench] signal={args.signal} duration={args.duration}s rate={args.rate}Hz "
          f"-> {PRESSURE_DES_CMD_TOPIC}")
    try:
        while True:
            t = time.monotonic() - t0
            if t >= args.duration:
                break
            msg.x = p_des_at(t, args.signal, base=args.base, delta=args.delta,
                             t_step=args.t_step, amp=args.amp, freq=args.freq,
                             jitter_period=args.jitter_period, seed=args.seed)
            cmd_pub.Write(msg)
            if state["p_now"] is not None:
                rows.append((t, state["p_des"], state["p_now"]))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[bench] interrupted")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "p_des", "p_now"])
            for r in rows:
                w.writerow([f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}"])
        print(f"[bench] wrote {len(rows)} rows -> {args.csv}")

    ts = [r[0] for r in rows]
    des = [r[1] for r in rows]
    now = [r[2] for r in rows]
    print(f"[bench] RMSE={rmse(des, now):.3f} PSI over {len(rows)} samples")
    if args.signal == "step":
        st = settling_time(ts, now, target=args.base + args.delta,
                           t_step=args.t_step, band=args.band)
        print(f"[bench] settling={st if st is None else round(st, 3)} s "
              f"(band ±{args.band} PSI)")


def _selftest():
    # step: before t_step -> base; after -> base+delta
    assert p_des_at(0.0, "step", base=100, delta=50, t_step=5, amp=0, freq=0,
                    jitter_period=1, seed=0) == 100
    assert p_des_at(6.0, "step", base=100, delta=50, t_step=5, amp=0, freq=0,
                    jitter_period=1, seed=0) == 150
    # sine at t=0 -> base; cosine at t=0 -> base+amp
    assert abs(p_des_at(0.0, "sine", base=100, delta=0, t_step=0, amp=20, freq=1,
                        jitter_period=1, seed=0) - 100) < 1e-9
    assert abs(p_des_at(0.0, "cosine", base=100, delta=0, t_step=0, amp=20, freq=1,
                        jitter_period=1, seed=0) - 120) < 1e-9
    # jitter: deterministic given seed, within band
    v = p_des_at(0.3, "jitter", base=100, delta=0, t_step=0, amp=5, freq=0,
                 jitter_period=1, seed=42)
    assert 95 <= v <= 105
    # rmse: identical -> 0; off-by-2 -> 2
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert abs(rmse([0, 0, 0], [2, 2, 2]) - 2.0) < 1e-9
    # settling: enters band at t=2 and stays
    t = [0, 1, 2, 3, 4]
    p = [100, 130, 149, 150, 150]
    assert settling_time(t, p, target=150, t_step=0, band=2) == 2
    # never settles -> None
    assert settling_time([0, 1], [100, 100], target=150, t_step=0, band=2) is None
    print("selftest OK")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run pure-fn asserts and exit")
    ap.add_argument("--iface", default="lo", help="DDS interface")
    ap.add_argument("--signal", default="step",
                    choices=["step", "sine", "cosine", "jitter"])
    ap.add_argument("--base", type=float, default=100.0, help="baseline p_des [PSI]")
    ap.add_argument("--delta", type=float, default=50.0, help="step size [PSI] (step)")
    ap.add_argument("--t-step", dest="t_step", type=float, default=5.0,
                    help="when the step happens [s]")
    ap.add_argument("--amp", type=float, default=20.0,
                    help="amplitude [PSI] (sine/cosine/jitter)")
    ap.add_argument("--freq", type=float, default=0.2, help="frequency [Hz] (sine/cosine)")
    ap.add_argument("--jitter-period", dest="jitter_period", type=float, default=0.5,
                    help="resample period [s] (jitter)")
    ap.add_argument("--seed", type=int, default=0, help="jitter RNG seed")
    ap.add_argument("--rate", type=float, default=50.0, help="publish/sample rate [Hz]")
    ap.add_argument("--duration", type=float, default=20.0, help="run length [s]")
    ap.add_argument("--band", type=float, default=4.0,
                    help="settling band ±PSI (step); matches deadband ε_exit=4")
    ap.add_argument("--csv", default=None, help="output CSV path (t_s,p_des,p_now)")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.selftest:
        _selftest()
    else:
        run(args)
