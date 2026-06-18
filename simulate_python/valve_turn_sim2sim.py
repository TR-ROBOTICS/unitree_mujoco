"""
valve_turn_sim2sim.py — Sim2sim runner for chained reach→turn valve task.

Decision D2 (fixed): reach phase = 15 s, then switch to turn policy.
Decision D3 (resolved 2026-06-02): reach obs (45d) implemented:
  [0:14] joint_pos_rel = qpos[arm] (reach q_default = zeros)
  [14:28] joint_vel_rel = qvel[arm]
  [28:31] valve_pos = (0.60, 0.0, 0.107) constant (DR off, pelvis welded)
  [31:45] last_action = previous step raw net output, init zeros

Action offset semantics (confirmed from IsaacLab source joint_actions.py):
  JointPositionAction.__init__: if use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, joint_ids].clone()
  This is the STATIC init pose cloned ONCE at env init — NOT accumulated per step.
  → reach q_target = ZEROS + 0.1 * net_out
  → turn  q_target = Q_DEFAULT_pregrip + 0.1 * net_out

Actuator mode: g1_29dof.xml uses <position kp> actuators for 14 arm joints.
  ACTUATOR_MODE = "position" → write q_target directly to data.ctrl.
  Do NOT also apply _pd_torque() — would double-control.

Joint-index map (B5): built at load time from mujoco.MjModel.joint_name2id().
  Asserts all 14 Isaac arm joint names are present and in the expected order.
  Valve joint also asserted. Fail loud = immediate AssertionError on mismatch.

Finger rest: MJCF finger joints (14 joints, damping=25, no actuator) set to
  qpos=0 at reset (MJCF default, within all joint ranges). No keyframe.

CLI:
  python valve_turn_sim2sim.py [--episodes N] [--p_des PSI] [--render]
                               [--seed S] [--max_steps M]
                               [--theta_init RAD] [--success_hold K]
                               [--mode {chained,turn_only}]

  --mode chained   (default): t=0 arms=zeros, run reach 15s then turn.
  --mode turn_only: t=0 arms sampled from reach_arm_positions.npy, skip reach.
                    Isolates the turn policy evaluation (95.1% Isaac claim).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# NOTE: src/unitree_mujoco and src/unitree_rl_lab are symlinks pointing OUTSIDE
# TFG (to ~/unitree_mujoco, ~/unitree_rl_lab).  Path(__file__).resolve() follows
# them and breaks parents[] arithmetic, so anchor on the fixed TFG root via env
# override or the known absolute path.
_REPO_ROOT = Path(os.environ.get("TFG_ROOT", "/home/tr-robotics/TFG"))
_MJCF_DIR = _REPO_ROOT / "src/unitree_mujoco/unitree_robots/g1"
SCENE_XML = str(_MJCF_DIR / "scene_29dof.xml")

_RL_ROOT = _REPO_ROOT / "src/unitree_rl_lab"
TURN_POLICY_PT = str(
    _RL_ROOT
    / "logs/rsl_rl/valve_turn_g1_29dof/2026-05-27_17-08-10/exported/policy.pt"
)
REACH_POLICY_PT = str(
    _RL_ROOT
    / "logs/rsl_rl/valve_reach_g1_29dof/2026-05-27_10-40-12/exported/policy.pt"
)
REACH_DATASET_NPY = str(
    _RL_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/datasets/reach_arm_positions.npy"
)

# ---------------------------------------------------------------------------
# Constants — verified against CONTEXT.md and docs/plan.md §Sim2sim
# ---------------------------------------------------------------------------

# Physics / policy rates
SIM_DT: float = 0.005          # 200 Hz
POLICY_HZ: float = 50.0
DECIMATION: int = 4             # policy_dt = SIM_DT * DECIMATION = 0.02 s

# g(θ) sensor  — CONTEXT.md §g(θ)
G_A: float = 4.527              # PSI/rad
G_B: float = -27.66             # PSI
P_MIN_PSI: float = 15.0
P_MAX_PSI: float = 200.0
P_SPAN: float = P_MAX_PSI - P_MIN_PSI  # 185 PSI

# Valve init
THETA_INIT_RAD: float = 9.42   # 1.5 revolutions (θ_min of operating envelope)
THETA_MIN: float = 9.42
THETA_MAX: float = 50.27

# Phase cutover — D2
REACH_DURATION_S: float = 15.0

# Isaac arm joint order (14d, L then R).  This is the canonical ordering for
# obs[0:14] and action[0:14].  The joint-index map assert checks MuJoCo names
# against this list in this exact sequence.
ISAAC_ARM_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# q_default (rad) in Isaac arm order above
Q_DEFAULT: np.ndarray = np.array([
    # left
    -0.7610, +0.1937, -0.1239, +0.4869, +0.3787, 0.0, 0.0,
    # right
    -0.7610, -0.1937, +0.1257, +0.5236, -0.4712, 0.0, 0.0,
], dtype=np.float32)

# PD gains per joint group — docs/plan.md §Sim2sim "PD gains k/d"
# shoulder_pitch + shoulder_roll : kp=100, kd=2
# shoulder_yaw + elbow           : kp=50,  kd=2
# all 6 wrist joints             : kp=40,  kd=2
_KP_PER_JOINT: np.ndarray = np.array([
    100.0, 100.0, 50.0, 50.0, 40.0, 40.0, 40.0,   # left
    100.0, 100.0, 50.0, 50.0, 40.0, 40.0, 40.0,   # right
], dtype=np.float32)
_KD_PER_JOINT: np.ndarray = np.full(14, 2.0, dtype=np.float32)

# Actuator mode flag.  Wave A2 wrote 14 <position> actuators for the arm joints
# (kp = Isaac stiffness; kd realized via joint damping=2).  Position mode => write
# q_target directly to ctrl; do NOT also apply _pd_torque() (would double-control).
ACTUATOR_MODE: str = "position"   # "torque" | "position"

# Valve joint name — TODO: confirm once Wave A1 rewrites valve.xml
VALVE_JOINT_NAME: str = "valve_joint"

# Success criterion: |p_now - p_des| < tolerance held for hold_steps consecutive
# policy steps.  Default K=10 @ 50 Hz = 0.2 s.
# CONTEXT.md specifies K=50 (1 s hold at 50 Hz) for the "sustained" criterion.
# Using K=10 here as a fast-convergence metric; override via --success_hold.
SUCCESS_HOLD_DEFAULT: int = 10
SUCCESS_TOL_PSI: float = 2.0    # ~1% of span, ε_sim from CONTEXT.md

# Isaac obs scale for action
ACTION_SCALE: float = 0.1       # q_target = q_default + ACTION_SCALE * net_out


# ---------------------------------------------------------------------------
# Episode phase FSM
# ---------------------------------------------------------------------------

class Phase(Enum):
    REACH = auto()
    TURN = auto()


# ---------------------------------------------------------------------------
# Joint-index map (B5) — built once at model load, asserted loudly
# ---------------------------------------------------------------------------

@dataclass
class JointIndexMap:
    """MuJoCo qpos/qvel indices for the 14 arm joints (Isaac order) + valve."""
    arm_qpos_idx: np.ndarray    # shape (14,), int — indices into data.qpos
    arm_qvel_idx: np.ndarray    # shape (14,), int — indices into data.qvel
    arm_ctrl_idx: np.ndarray    # shape (14,), int — indices into data.ctrl (actuator order)
    valve_qpos_idx: int
    valve_qvel_idx: int


def build_joint_index_map(model: mujoco.MjModel) -> JointIndexMap:
    """
    Resolve MuJoCo qpos/qvel/ctrl indices for all 14 Isaac arm joints and the
    valve joint.  Asserts that every expected joint name exists in the model and
    that the actuator names are consistent with the joint order.

    Fails loudly (AssertionError) on any mismatch — this is the #1 correctness
    risk (B5) for sim2sim.

    qpos offset: each 1-DoF hinge joint occupies 1 element in qpos, starting
    at model.jnt_qposadr[joint_id].  Likewise qvel uses model.jnt_dofadr[joint_id].
    """

    print("\n[joint-map] Resolving MuJoCo joint indices ...")

    # --- Arm joints ---
    arm_qpos_idx = np.zeros(14, dtype=int)
    arm_qvel_idx = np.zeros(14, dtype=int)
    arm_ctrl_idx = np.zeros(14, dtype=int)

    # Build actuator name → ctrl index map
    act_name_to_ctrl: dict[str, int] = {}
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name is not None:
            act_name_to_ctrl[act_name] = i

    for i, jname in enumerate(ISAAC_ARM_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, (
            f"[joint-map] FATAL: joint '{jname}' not found in MuJoCo model. "
            f"Check g1_29dof.xml / scene_29dof.xml parity."
        )
        arm_qpos_idx[i] = model.jnt_qposadr[jid]
        arm_qvel_idx[i] = model.jnt_dofadr[jid]

        # Actuator name convention: strip '_joint' suffix
        act_name_candidate = jname.replace("_joint", "")
        assert act_name_candidate in act_name_to_ctrl, (
            f"[joint-map] FATAL: expected actuator '{act_name_candidate}' for "
            f"joint '{jname}' not found.  Actuators present: "
            f"{sorted(act_name_to_ctrl.keys())}"
        )
        arm_ctrl_idx[i] = act_name_to_ctrl[act_name_candidate]

    # --- Valve joint ---
    valve_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, VALVE_JOINT_NAME)
    assert valve_jid >= 0, (
        f"[joint-map] FATAL: valve joint '{VALVE_JOINT_NAME}' not found in model. "
        f"Wave A1 (valve.xml rewrite) must use joint name '{VALVE_JOINT_NAME}'."
    )
    valve_qpos_idx = int(model.jnt_qposadr[valve_jid])
    valve_qvel_idx = int(model.jnt_dofadr[valve_jid])

    # --- Print resolved map ---
    print(f"{'Isaac order':>4}  {'joint name':<35}  qpos  qvel  ctrl")
    print("-" * 68)
    for i, jname in enumerate(ISAAC_ARM_JOINT_NAMES):
        print(
            f"  {i:>2}   {jname:<35}  "
            f"{arm_qpos_idx[i]:>4}  {arm_qvel_idx[i]:>4}  {arm_ctrl_idx[i]:>4}"
        )
    print(
        f"{'valve':>6}  {VALVE_JOINT_NAME:<35}  "
        f"{valve_qpos_idx:>4}  {valve_qvel_idx:>4}  (passive)"
    )
    print()

    return JointIndexMap(
        arm_qpos_idx=arm_qpos_idx,
        arm_qvel_idx=arm_qvel_idx,
        arm_ctrl_idx=arm_ctrl_idx,
        valve_qpos_idx=valve_qpos_idx,
        valve_qvel_idx=valve_qvel_idx,
    )


# ---------------------------------------------------------------------------
# g(θ) sensor wrapper
# ---------------------------------------------------------------------------

def compute_p_now(theta: float) -> float:
    """
    p_now (PSI) = clamp(G_A * theta + G_B, P_MIN_PSI, P_MAX_PSI).
    Mirrors g(θ) in CONTEXT.md §g(θ) and docs/agents/hardware-rig.md.
    """
    return float(np.clip(G_A * theta + G_B, P_MIN_PSI, P_MAX_PSI))


def normalize_p(p_psi: float) -> float:
    """Map PSI ∈ [15, 200] → [0, 1] and clip."""
    return float(np.clip((p_psi - P_MIN_PSI) / P_SPAN, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------

def build_turn_obs(
    data: mujoco.MjData,
    jmap: JointIndexMap,
    p_des_norm: float,
) -> torch.Tensor:
    """
    Turn obs (30d) — fully implemented.

    Layout (Isaac canonical, CONTEXT.md §Policy: Turn):
      [0:14]  joint_pos_rel  = qpos[arm] − q_default
      [14:28] joint_vel_rel  = qvel[arm]   (raw; Isaac uses qd directly)
      [28]    p_now_norm     = clamp(g(θ),15,200)/185, ∈[0,1]
      [29]    p_des_norm     = p_des_buf / 185, ∈[0,1]
    """
    q = data.qpos[jmap.arm_qpos_idx].astype(np.float32)
    qd = data.qvel[jmap.arm_qvel_idx].astype(np.float32)

    joint_pos_rel = q - Q_DEFAULT
    joint_vel_rel = qd

    theta = float(data.qpos[jmap.valve_qpos_idx])
    p_now_psi = compute_p_now(theta)
    p_now_norm = normalize_p(p_now_psi)

    obs = np.concatenate([
        joint_pos_rel,          # 14
        joint_vel_rel,          # 14
        [p_now_norm],           # 1
        [p_des_norm],           # 1
    ]).astype(np.float32)       # total: 30

    assert obs.shape == (30,), f"build_turn_obs: expected 30d, got {obs.shape}"
    return torch.from_numpy(obs).unsqueeze(0)  # (1, 30)


# Reach action offset = ZEROS (reach env zeroed all 14 arm init pos; D3).
# Turn uses Q_DEFAULT (pre-grip).  Both are STATIC (IsaacLab joint_actions.py:
# offset cloned once at env init, NOT accumulated per step — D3 accumulation
# claim corrected against source).
Q_DEFAULT_REACH: np.ndarray = np.zeros(14, dtype=np.float32)

# valve_pos obs term [28:31] = valve_root_pos_w − robot_root_pos_w (world frame).
# Valve DR is OFF and pelvis welded → constant.
#   valve root (0.60, 0.0, 0.90) − robot root (0.0, 0.0, 0.793) = (0.60, 0.0, 0.107)
_VALVE_POS_ROBOT_FRAME: np.ndarray = np.array([0.60, 0.0, 0.107], dtype=np.float32)


def build_reach_obs(
    data: mujoco.MjData,
    jmap: JointIndexMap,
    last_action: np.ndarray,
) -> torch.Tensor:
    """
    Reach obs (45d) — implemented per D3 (sim-isaaclab report 2026-06-02).

    Layout (reach env_cfg, valve_reach_g1_29dof):
      [0:14]  joint_pos_rel = qpos[arm] − q_default_REACH (= zeros) → raw qpos. clip(±10)
      [14:28] joint_vel_rel = qvel[arm].                              clip(±20)
      [28:31] valve_pos     = valve_root − robot_root (world).        clip(±2)
              DR off + pelvis welded → constant (0.60, 0.0, 0.107).
      [31:45] last_action   = previous step's raw net output ∈[−1,1]. clip(±1)
              Zeros on first step.
    """
    q = data.qpos[jmap.arm_qpos_idx].astype(np.float32)
    qd = data.qvel[jmap.arm_qvel_idx].astype(np.float32)

    joint_pos_rel = np.clip(q - Q_DEFAULT_REACH, -10.0, 10.0)
    joint_vel_rel = np.clip(qd, -20.0, 20.0)
    valve_pos = np.clip(_VALVE_POS_ROBOT_FRAME, -2.0, 2.0)
    last_act = np.clip(last_action.astype(np.float32), -1.0, 1.0)

    obs = np.concatenate([
        joint_pos_rel,          # 14
        joint_vel_rel,          # 14
        valve_pos,              # 3
        last_act,               # 14
    ]).astype(np.float32)       # total: 45

    assert obs.shape == (45,), f"build_reach_obs: expected 45d, got {obs.shape}"
    return torch.from_numpy(obs).unsqueeze(0)  # (1, 45)


# ---------------------------------------------------------------------------
# PD torque controller (Python-side, for torque-mode actuators)
# ---------------------------------------------------------------------------

def _pd_torque(
    q_target: np.ndarray,
    data: mujoco.MjData,
    jmap: JointIndexMap,
) -> np.ndarray:
    """
    Compute PD torques for the 14 arm joints.
    τ = kp * (q_target - q) - kd * qd
    Returns torques in Isaac arm order (same as arm_ctrl_idx).
    """
    q = data.qpos[jmap.arm_qpos_idx].astype(np.float32)
    qd = data.qvel[jmap.arm_qvel_idx].astype(np.float32)
    tau = _KP_PER_JOINT * (q_target - q) - _KD_PER_JOINT * qd
    return tau


def _apply_action(
    net_out: np.ndarray,
    data: mujoco.MjData,
    jmap: JointIndexMap,
    offset: np.ndarray,
) -> None:
    """
    Convert policy output → ctrl signal and write to data.ctrl.

    net_out: (14,) float32 in [-1, 1] (Isaac network raw output)
    offset:  (14,) static action offset — Q_DEFAULT (turn) or Q_DEFAULT_REACH=0 (reach)
    q_target = offset + ACTION_SCALE * net_out

    ACTUATOR_MODE == "position" (current): arms are <position kp> actuators →
      write q_target directly to ctrl; kp/kd handled by MJCF.
    ACTUATOR_MODE == "torque": compute PD torque in Python, clip to ctrlrange.
    """
    q_target = offset + ACTION_SCALE * net_out

    if ACTUATOR_MODE == "torque":
        tau = _pd_torque(q_target, data, jmap)
        for i, ctrl_idx in enumerate(jmap.arm_ctrl_idx):
            lo = model_ctrl_range[ctrl_idx, 0]
            hi = model_ctrl_range[ctrl_idx, 1]
            data.ctrl[ctrl_idx] = float(np.clip(tau[i], lo, hi))
    elif ACTUATOR_MODE == "position":
        for i, ctrl_idx in enumerate(jmap.arm_ctrl_idx):
            data.ctrl[ctrl_idx] = float(q_target[i])
    else:
        raise ValueError(f"Unknown ACTUATOR_MODE: {ACTUATOR_MODE}")


# Module-level ctrl range cache — populated in load_model()
model_ctrl_range: np.ndarray = np.zeros((0, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Model + policy loading
# ---------------------------------------------------------------------------

def load_model() -> tuple[mujoco.MjModel, mujoco.MjData, JointIndexMap]:
    """Load MuJoCo model from scene_29dof.xml and build joint-index map."""
    print(f"[load] scene XML: {SCENE_XML}")
    if not os.path.exists(SCENE_XML):
        sys.exit(f"[load] ERROR: scene XML not found: {SCENE_XML}")

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    global model_ctrl_range
    model_ctrl_range = model.actuator_ctrlrange.copy()

    jmap = build_joint_index_map(model)
    return model, data, jmap


def load_policies(device: str = "cpu") -> tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    """
    Load turn JIT (30d obs, 14d act) and reach JIT (45d obs, 14d act).
    Returns (reach_policy, turn_policy).

    Both are torch.jit.ScriptModule — no IsaacLab dependency at inference.
    """
    print(f"[policy] Loading turn policy: {TURN_POLICY_PT}")
    if not os.path.exists(TURN_POLICY_PT):
        sys.exit(f"[policy] ERROR: turn policy not found: {TURN_POLICY_PT}")
    turn_policy = torch.jit.load(TURN_POLICY_PT, map_location=device)
    turn_policy.eval()

    print(f"[policy] Loading reach policy: {REACH_POLICY_PT}")
    if not os.path.exists(REACH_POLICY_PT):
        sys.exit(f"[policy] ERROR: reach policy not found: {REACH_POLICY_PT}")
    reach_policy = torch.jit.load(REACH_POLICY_PT, map_location=device)
    reach_policy.eval()

    print(f"[policy] Both policies loaded on device={device}")
    return reach_policy, turn_policy


def load_reach_dataset() -> np.ndarray:
    """Load reach terminal arm states dataset — (10000, 14) float32."""
    print(f"[dataset] Loading reach arm positions: {REACH_DATASET_NPY}")
    if not os.path.exists(REACH_DATASET_NPY):
        sys.exit(f"[dataset] ERROR: dataset not found: {REACH_DATASET_NPY}")
    ds = np.load(REACH_DATASET_NPY).astype(np.float32)
    assert ds.ndim == 2 and ds.shape[1] == 14, (
        f"[dataset] Expected shape (N, 14), got {ds.shape}"
    )
    print(f"[dataset] Loaded {ds.shape[0]} samples, shape={ds.shape}")
    return ds


# ---------------------------------------------------------------------------
# Episode reset
# ---------------------------------------------------------------------------

def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jmap: JointIndexMap,
    reach_dataset: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    theta_init: float = THETA_INIT_RAD,
) -> None:
    """
    Reset simulation state for one episode.

    - Zero all velocities and forces (mj_resetData → MJCF default qpos).
    - Arm init depends on mode:
        chained   : arms = all ZEROS (reach env t=0 init); reach policy runs first.
        turn_only : arms = sampled row from reach_dataset (skip reach, isolate turn).
    - Set valve qpos = theta_init.
    - Fingers left at MJCF default qpos=0 (frozen by damping=25, no actuator).
    - Pelvis is welded in MJCF (no free joint); legacy pin guard kept harmless.
    """
    mujoco.mj_resetData(model, data)

    # Legacy guard: pelvis is now welded (no floating_base_joint).  If an old MJCF
    # is loaded this pins it; with the welded MJCF mj_name2id returns -1 → no-op.
    pelvis_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if pelvis_jid >= 0:
        pelvis_qpos_start = model.jnt_qposadr[pelvis_jid]
        data.qpos[pelvis_qpos_start + 0] = 0.0
        data.qpos[pelvis_qpos_start + 1] = 0.0
        data.qpos[pelvis_qpos_start + 2] = 0.793
        data.qpos[pelvis_qpos_start + 3] = 1.0  # qw
        data.qpos[pelvis_qpos_start + 4] = 0.0
        data.qpos[pelvis_qpos_start + 5] = 0.0
        data.qpos[pelvis_qpos_start + 6] = 0.0

    # Arm init per mode
    if mode == "turn_only":
        row_idx = int(rng.integers(0, len(reach_dataset)))
        data.qpos[jmap.arm_qpos_idx] = reach_dataset[row_idx]  # (14,) Isaac order
    elif mode == "chained":
        data.qpos[jmap.arm_qpos_idx] = 0.0  # reach env t=0 = arms at zero
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Set valve angle
    data.qpos[jmap.valve_qpos_idx] = theta_init

    # Forward to propagate geom positions
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    t: float
    phase: str
    theta: float
    p_now: float
    p_des: float
    action_norm: float
    hold_count: int
    success: bool


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jmap: JointIndexMap,
    reach_policy: torch.jit.ScriptModule,
    turn_policy: torch.jit.ScriptModule,
    reach_dataset: np.ndarray,
    rng: np.random.Generator,
    p_des_psi: float,
    mode: str = "chained",
    theta_init: float = THETA_INIT_RAD,
    max_steps: int = 3000,
    success_hold: int = SUCCESS_HOLD_DEFAULT,
    viewer: Optional[mujoco.viewer.Handle] = None,
) -> tuple[bool, list[StepLog], np.ndarray]:
    """
    Run one episode.

    mode == "chained":   reach policy 15 s → cutover → turn policy.
    mode == "turn_only": start directly in TURN (arms init from dataset).

    Args:
        max_steps:    maximum policy steps (at 50 Hz; 3000 = 60 s)
        success_hold: consecutive policy steps within SUCCESS_TOL_PSI to call success
        viewer:       optional mujoco viewer handle (passive)

    Returns:
        (success, logs, theta_trajectory)
        theta_trajectory: np.ndarray of shape (policy_steps,) — valve angle per step
    """
    reset_episode(model, data, jmap, reach_dataset, rng, mode, theta_init)

    p_des_norm = normalize_p(p_des_psi)
    # turn_only skips reach entirely; chained starts in REACH
    phase = Phase.TURN if mode == "turn_only" else Phase.REACH
    hold_count = 0
    success = False

    logs: list[StepLog] = []
    theta_traj: list[float] = []

    last_action = np.zeros(14, dtype=np.float32)  # reach obs last_action slot, init zeros

    for step in range(max_steps):
        t_sim = step * DECIMATION * SIM_DT  # elapsed sim time (s)

        # --- Phase transition (chained only) ---
        if phase == Phase.REACH and t_sim >= REACH_DURATION_S:
            phase = Phase.TURN
            hold_count = 0
            print(f"[episode] t={t_sim:.2f}s — switching REACH→TURN")

        # --- Build obs, infer, pick action offset ---
        with torch.no_grad():
            if phase == Phase.REACH:
                obs = build_reach_obs(data, jmap, last_action)
                net_out = reach_policy(obs).squeeze(0).numpy()
                offset = Q_DEFAULT_REACH
            else:
                obs = build_turn_obs(data, jmap, p_des_norm)
                net_out = turn_policy(obs).squeeze(0).numpy()
                offset = Q_DEFAULT

        last_action = net_out.copy()

        # --- Apply action (position target or PD torque) ---
        _apply_action(net_out, data, jmap, offset)

        # --- Step physics (decimation) ---
        for _ in range(DECIMATION):
            # Pin pelvis if floating base still present (pre-Wave-A2)
            pelvis_jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
            )
            if pelvis_jid >= 0:
                ps = model.jnt_qposadr[pelvis_jid]
                data.qvel[model.jnt_dofadr[pelvis_jid] : model.jnt_dofadr[pelvis_jid] + 6] = 0.0
                data.qacc[model.jnt_dofadr[pelvis_jid] : model.jnt_dofadr[pelvis_jid] + 6] = 0.0

            mujoco.mj_step(model, data)

            # Re-pin pelvis position after step
            if pelvis_jid >= 0:
                ps = model.jnt_qposadr[pelvis_jid]
                data.qpos[ps + 0] = 0.0
                data.qpos[ps + 1] = 0.0
                data.qpos[ps + 2] = 0.793
                data.qpos[ps + 3] = 1.0
                data.qpos[ps + 4:ps + 7] = 0.0

        if viewer is not None:
            viewer.sync()

        # --- Read state ---
        theta = float(data.qpos[jmap.valve_qpos_idx])
        p_now_psi = compute_p_now(theta)
        action_norm = float(np.linalg.norm(net_out))
        theta_traj.append(theta)

        # --- Success check (turn phase only) ---
        if phase == Phase.TURN:
            if abs(p_now_psi - p_des_psi) < SUCCESS_TOL_PSI:
                hold_count += 1
            else:
                hold_count = 0
            if hold_count >= success_hold:
                success = True

        logs.append(StepLog(
            t=t_sim,
            phase=phase.name,
            theta=theta,
            p_now=p_now_psi,
            p_des=p_des_psi,
            action_norm=action_norm,
            hold_count=hold_count,
            success=success,
        ))

        if success:
            break

    return success, logs, np.array(theta_traj, dtype=np.float32)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def write_logs_csv(logs: list[StepLog], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "theta", "p_now", "p_des", "action_norm", "hold_count", "success"])
        for l in logs:
            w.writerow([
                f"{l.t:.4f}", l.phase, f"{l.theta:.4f}", f"{l.p_now:.2f}",
                f"{l.p_des:.2f}", f"{l.action_norm:.4f}", l.hold_count, int(l.success)
            ])


# ---------------------------------------------------------------------------
# Batch runner (Wave C / T9 entry point)
# ---------------------------------------------------------------------------

def batch_eval(
    n_episodes: int,
    p_des_psi: float,
    theta_init: float,
    max_steps: int,
    success_hold: int,
    render: bool,
    seed: int,
    mode: str = "chained",
    log_dir: str = "sim2sim_logs",
) -> float:
    """
    Run n_episodes, return success rate.  This is what Wave C/T9 calls.

    Logs per-episode CSV to log_dir/episode_NNNN.csv.
    Prints per-episode result and final success rate.
    """
    os.makedirs(log_dir, exist_ok=True)

    model, data, jmap = load_model()
    reach_policy, turn_policy = load_policies()
    reach_dataset = load_reach_dataset()
    rng = np.random.default_rng(seed)

    viewer_handle = None
    if render:
        # Passive viewer — must be created before the loop.
        # Only works on a display; headless runs should set render=False.
        viewer_handle = mujoco.viewer.launch_passive(model, data)

    successes = 0
    for ep in range(n_episodes):
        # Randomize p_des and theta_init per episode for proper eval
        ep_p_des = p_des_psi if p_des_psi > 0 else float(
            rng.uniform(P_MIN_PSI, P_MAX_PSI)
        )
        ep_theta_init = theta_init if theta_init > 0 else float(
            rng.uniform(THETA_MIN, THETA_MAX)
        )

        success, logs, _ = run_episode(
            model, data, jmap,
            reach_policy, turn_policy,
            reach_dataset, rng,
            p_des_psi=ep_p_des,
            mode=mode,
            theta_init=ep_theta_init,
            max_steps=max_steps,
            success_hold=success_hold,
            viewer=viewer_handle,
        )

        if success:
            successes += 1

        # Write episode log
        csv_path = os.path.join(log_dir, f"episode_{ep:04d}.csv")
        write_logs_csv(logs, csv_path)

        sr_so_far = successes / (ep + 1) * 100
        final_p = logs[-1].p_now if logs else float("nan")
        print(
            f"[ep {ep+1:>4}/{n_episodes}]  "
            f"success={int(success)}  "
            f"p_des={ep_p_des:.1f}  p_final={final_p:.1f}  "
            f"steps={len(logs)}  "
            f"SR_so_far={sr_so_far:.1f}%"
        )

    if viewer_handle is not None:
        viewer_handle.close()

    success_rate = successes / n_episodes
    print(f"\n[result] Episodes={n_episodes}  Successes={successes}  "
          f"SuccessRate={success_rate*100:.1f}%")
    print(f"[result] Gate: {'PASS' if success_rate >= 0.80 else 'FAIL'} "
          f"(threshold 80%)")
    return success_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo sim2sim runner: chained reach→turn")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to run (default 1 for smoke test)")
    p.add_argument("--p_des", type=float, default=107.5,
                   help="Target pressure PSI.  "
                        "Default 107.5 = mid-range.  "
                        "Set ≤0 for random p_des per episode.")
    p.add_argument("--theta_init", type=float, default=THETA_INIT_RAD,
                   help=f"Initial valve angle rad (default {THETA_INIT_RAD}).  "
                        "Set ≤0 for random θ per episode.")
    p.add_argument("--render", action="store_true",
                   help="Open MuJoCo passive viewer (requires display)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--max_steps", type=int, default=3000,
                   help="Max policy steps per episode (default 3000 = 60 s @ 50 Hz)")
    p.add_argument("--success_hold", type=int, default=SUCCESS_HOLD_DEFAULT,
                   help=f"Consecutive steps within tolerance for success "
                        f"(default {SUCCESS_HOLD_DEFAULT}; CONTEXT.md criterion = 50)")
    p.add_argument("--log_dir", type=str, default="sim2sim_logs",
                   help="Directory for per-episode CSV logs")
    p.add_argument("--mode", type=str, default="chained",
                   choices=["chained", "turn_only"],
                   help="chained: reach 15s then turn (arms init zeros).  "
                        "turn_only: skip reach, arms init from dataset (isolate turn).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("valve_turn_sim2sim.py — MuJoCo sim2sim runner")
    print(f"  scene:         {SCENE_XML}")
    print(f"  turn policy:   {TURN_POLICY_PT}")
    print(f"  reach policy:  {REACH_POLICY_PT}")
    print(f"  mode:          {args.mode}")
    print(f"  episodes:      {args.episodes}")
    print(f"  p_des:         {args.p_des} PSI {'(fixed)' if args.p_des > 0 else '(random)'}")
    print(f"  theta_init:    {args.theta_init} rad {'(fixed)' if args.theta_init > 0 else '(random)'}")
    print(f"  max_steps:     {args.max_steps}  ({args.max_steps * DECIMATION * SIM_DT:.1f} s sim)")
    print(f"  success_hold:  {args.success_hold} steps  ({args.success_hold / POLICY_HZ:.2f} s)")
    print(f"  actuator_mode: {ACTUATOR_MODE}")
    print(f"  render:        {args.render}")
    print(f"  seed:          {args.seed}")
    print("=" * 70)

    batch_eval(
        n_episodes=args.episodes,
        p_des_psi=args.p_des,
        theta_init=args.theta_init,
        max_steps=args.max_steps,
        success_hold=args.success_hold,
        render=args.render,
        seed=args.seed,
        mode=args.mode,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
