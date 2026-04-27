"""
Phase 4: Inverse Kinematics demo for industrial G1 humanoid.

Right arm tool tip tracks a small target circle anchored at the arm's
starting position. Conservative IK gains keep arm motion gentle enough
to not disturb the robot's balance. A safety cutoff freezes the IK if
the pelvis deviates significantly from nominal.

IK method: Damped Least Squares (DLS) Jacobian.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ============================================================
# Setup
# ============================================================
model = mujoco.MjModel.from_xml_path("robot/scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
standing_ctrl = data.ctrl.copy()

tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip_site")
torso_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
pelvis_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

RIGHT_ARM_ACTUATOR_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
right_arm_actuator_ids = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    for n in RIGHT_ARM_ACTUATOR_NAMES
])

right_arm_dof_ids = []
right_arm_qpos_ids = []
right_arm_jnt_ranges = []
for jname in RIGHT_ARM_ACTUATOR_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    right_arm_dof_ids.append(model.jnt_dofadr[jid])
    right_arm_qpos_ids.append(model.jnt_qposadr[jid])
    right_arm_jnt_ranges.append(model.jnt_range[jid])
right_arm_dof_ids   = np.array(right_arm_dof_ids)
right_arm_qpos_ids  = np.array(right_arm_qpos_ids)
right_arm_jnt_ranges = np.array(right_arm_jnt_ranges)

arm_q_target = data.qpos[right_arm_qpos_ids].copy()

# Anchor target circle at starting tool tip
mujoco.mj_forward(model, data)
START_TIP = data.site_xpos[tool_site_id].copy()
NOMINAL_PELVIS = data.xpos[pelvis_id].copy()

print(f"Initial tool tip:  {START_TIP}")
print(f"Initial pelvis:    {NOMINAL_PELVIS}\n")

# ============================================================
# Target: SMALL vertical circle, slow
# ============================================================
CIRCLE_CENTER = START_TIP
CIRCLE_RADIUS = 0.04    # 4cm — small enough to keep arm motion gentle
CIRCLE_PERIOD = 8.0     # slow

def target_position(t):
    omega = 2 * np.pi / CIRCLE_PERIOD
    x_off = CIRCLE_RADIUS * np.sin(omega * t)
    z_off = CIRCLE_RADIUS * (1 - np.cos(omega * t))
    return CIRCLE_CENTER + np.array([x_off, 0.0, z_off])

# ============================================================
# DLS Jacobian IK — very conservative
# ============================================================
DAMPING        = 0.15
IK_GAIN        = 0.3      # gentle
MAX_JOINT_STEP = 0.005    # very small per-step cap (rad)
PELVIS_DRIFT_CUTOFF = 0.05  # m — if pelvis drifts more than this, freeze IK

def solve_ik_step(model, data, target_pos):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
    J = jacp[:, right_arm_dof_ids]

    current_tip = data.site_xpos[tool_site_id]
    error = target_pos - current_tip

    JJt = J @ J.T
    damped_inv = np.linalg.inv(JJt + (DAMPING ** 2) * np.eye(3))
    dq = J.T @ damped_inv @ (IK_GAIN * error)
    dq = np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP)
    return dq, np.linalg.norm(error)

# ============================================================
# Main loop
# ============================================================
print(f"Vertical circle:  r={CIRCLE_RADIUS}m, period={CIRCLE_PERIOD}s")
print(f"IK: damping={DAMPING}, gain={IK_GAIN}, step cap={MAX_JOINT_STEP} rad")
print(f"Safety: freeze IK if pelvis drifts > {PELVIS_DRIFT_CUTOFF}m\n")
print(f"{'time':<8}{'tip_err':<12}{'pelvis_drift':<15}{'status'}")
print("-" * 55)

last_log = 0.0
ik_frozen = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        sim_t = data.time

        # Always hold standing pose for legs/waist/left arm
        data.ctrl[:] = standing_ctrl

        # Check pelvis drift — safety cutoff
        pelvis_drift = np.linalg.norm(
            data.xpos[pelvis_id, :2] - NOMINAL_PELVIS[:2]
        )
        if pelvis_drift > PELVIS_DRIFT_CUTOFF:
            ik_frozen = True

        # Run IK only if not frozen
        target = target_position(sim_t)
        if not ik_frozen:
            dq, err = solve_ik_step(model, data, target)
            arm_q_target += dq
            for i in range(len(arm_q_target)):
                lo, hi = right_arm_jnt_ranges[i]
                arm_q_target[i] = np.clip(arm_q_target[i], lo, hi)
            data.ctrl[right_arm_actuator_ids] = arm_q_target
        else:
            err = np.linalg.norm(target - data.site_xpos[tool_site_id])
            # Don't update arm_q_target — keep last commanded position
            data.ctrl[right_arm_actuator_ids] = arm_q_target

        # Visualize target
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=target,
            mat=np.eye(3).flatten(),
            rgba=[0.2, 1.0, 0.2, 0.9],
        )
        viewer.user_scn.ngeom = 1

        if sim_t - last_log > 0.5:
            status = "FROZEN" if ik_frozen else "tracking"
            print(f"t={sim_t:5.2f}s  err={err*100:5.1f}cm   "
                  f"drift={pelvis_drift*100:4.1f}cm   {status}")
            last_log = sim_t

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)