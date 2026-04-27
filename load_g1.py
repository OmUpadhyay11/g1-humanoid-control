import mujoco
import mujoco.viewer
import time

# Load the model
model = mujoco.MjModel.from_xml_path("robot/scene.xml")
data = mujoco.MjData(model)


print("\nBodies:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {i}: {name}")


print(f"Model loaded: {model.nq} position DOFs, {model.nv} velocity DOFs")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.nu}")
print(f"Number of bodies: {model.nbody}")

# List all joints
print("\nJoints:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  {i}: {name}")

# List all actuators (these are what you'll command later)
print("\nActuators:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {i}: {name}")

# Launch interactive viewer with simulation running
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        # Real-time pacing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)