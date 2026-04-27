"""
Phase 3: Balance controller for industrial G1 humanoid.

Holds the stock standing keyframe via joint-space PD (built into MuJoCo's
<position> actuators). Periodic torso pushes characterize the passive
stability region.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("robot/scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)  # "stand" keyframe
standing_ctrl = data.ctrl.copy()

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
torso_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

mujoco.mj_forward(model, data)
nominal_pelvis = data.xpos[pelvis_id].copy()
print(f"Total mass: {sum(model.body_mass):.1f} kg")
print(f"Nominal pelvis: ({nominal_pelvis[0]:+.3f}, {nominal_pelvis[1]:+.3f}, {nominal_pelvis[2]:.3f})\n")

# ---------------- Push settings ----------------
PUSH_INTERVAL    = 6.0
PUSH_DURATION    = 0.15
PUSH_FORCE_MAG   = 30.0
FALL_HEIGHT      = 0.55
RECOVERY_THRESH  = 0.03

next_push_time   = PUSH_INTERVAL
push_end_time    = 0.0
push_force       = np.zeros(3)
push_count       = 0
fallen           = False
in_recovery      = False
recovery_start_t = 0.0
peak_deviation   = 0.0

print(f"Pushes: {PUSH_FORCE_MAG:.0f}N for {PUSH_DURATION}s, every {PUSH_INTERVAL}s")
print("-" * 65)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        sim_t = data.time

        data.ctrl[:] = standing_ctrl

        if not fallen and sim_t >= next_push_time:
            angle = np.random.uniform(0, 2 * np.pi)
            push_force = np.array([
                PUSH_FORCE_MAG * np.cos(angle),
                PUSH_FORCE_MAG * np.sin(angle),
                0.0
            ])
            push_end_time    = sim_t + PUSH_DURATION
            next_push_time   = sim_t + PUSH_INTERVAL
            push_count      += 1
            in_recovery      = True
            recovery_start_t = sim_t
            peak_deviation   = 0.0
            print(f"  t={sim_t:5.1f}s | PUSH #{push_count}  ({push_force[0]:+4.0f}, {push_force[1]:+4.0f})N",
                  end="", flush=True)

        if sim_t < push_end_time:
            data.xfrc_applied[torso_id, :3] = push_force
        else:
            data.xfrc_applied[torso_id, :3] = 0.0

        pelvis_pos = data.xpos[pelvis_id]
        horiz_dev = np.linalg.norm(pelvis_pos[:2] - nominal_pelvis[:2])

        if in_recovery:
            peak_deviation = max(peak_deviation, horiz_dev)
            if sim_t > push_end_time + 0.3 and horiz_dev < RECOVERY_THRESH:
                rec_time = sim_t - recovery_start_t
                print(f"   peak={peak_deviation*100:5.1f}cm  rec={rec_time:.2f}s", flush=True)
                in_recovery = False

        if not fallen and pelvis_pos[2] < FALL_HEIGHT:
            fallen = True
            print(f"   *** FELL at t={sim_t:.1f}s ***", flush=True)
            data.xfrc_applied[torso_id, :3] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)