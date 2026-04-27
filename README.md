# G1 Humanoid Control

**29-DOF Unitree G1 humanoid simulation in MuJoCo** with joint-space PD balance control and Damped Least Squares Jacobian inverse kinematics. Configured for an industrial inspection application with a sensor pod and payload pack.

> Built as a controls/sim portfolio project to demonstrate working knowledge of humanoid platform configuration, classical balance control, and redundant-arm IK — the foundational toolkit underneath modern RL-based humanoid pipelines.

---

## Demos

### Balance under disturbance
The robot holds a hand-tuned standing pose via joint-space PD on MuJoCo's `<position>` actuators, while randomized 30 N impulse forces are applied at the torso every 6 seconds.

| Metric | Value |
|---|---|
| Total mass | 33.3 kg |
| DOFs (actuated) | 29 |
| Push magnitude | 30 N for 0.15 s |
| Pushes survived (continuous run) | **36 / 36** over 3.5 min |
| Lateral / forward push: peak deviation | 0.9–3.2 cm |
| Lateral / forward push: settling time | 0.45 s |
| Backward push: peak deviation | 5.6–6.0 cm |
| Backward push: settling time | 1.6–2.3 s |

The anisotropic recovery (clean lateral, sluggish backward) is consistent with the ankle pitch joint's asymmetric range (-0.87 / +0.52 rad), which limits backward rotation of the foot and shrinks the rearward stability margin.

### Inverse kinematics
The right arm's tool-tip end-effector tracks a 3D Cartesian target (vertical circle) using Damped Least Squares (DLS) Jacobian IK on the 7 redundant arm joints. The rest of the body (legs, waist, left arm) continues balancing under PD.

| Metric | Value |
|---|---|
| Arm DOFs | 7 (redundant) |
| IK method | DLS Jacobian, λ = 0.15 |
| Position tracking error (steady state) | 0.4–2.8 cm |
| Pelvis drift induced by arm motion | < 1 cm |
| Continuous tracking duration tested | 13+ s |

A pelvis-drift safety cutoff freezes the IK if arm motion begins to destabilize the base — a defensive layer learned from observing arm/base coupling failures during tuning.

---

## What this project demonstrates

- **Robot description authoring** — modified MJCF (MuJoCo XML) to add an industrial payload pack, head-mounted sensor pod, and tool-tip end-effector site to a Unitree G1 base model
- **Joint-space PD control** — used MuJoCo's built-in PD on a hand-tuned standing keyframe; characterized the passive recovery region under randomized disturbances
- **Damped Least Squares IK** — implemented Jacobian-based IK from scratch using `mj_jacSite`, including damping for singularity robustness, integration on the actual joint state to avoid command-vs-actual lag, and clipping to joint limits
- **System debugging** — diagnosed and fixed cascading failure modes including arm/base dynamic coupling, payload-induced CoM shifts, and Jacobian linearization errors at far targets

---

## Repository layout

```
humanoid_sim/
├── load_g1.py        # Phase 1: load + visualize the model, print structure
├── balance.py        # Phase 3: PD balance + disturbance characterization
├── ik_demo.py        # Phase 4: DLS Jacobian IK on the right arm
├── robot/
│   ├── g1.xml        # Modified G1 model (payload, sensor pod, tool tip)
│   ├── scene.xml     # Wraps g1.xml with floor + lighting
│   └── assets/       # STL meshes
└── README.md
```

---

## Running it

### Setup
```bash
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate # macOS/Linux
pip install mujoco numpy
```

You also need the [Unitree G1 mesh assets from MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) copied into `robot/assets/`. The `g1.xml` and `scene.xml` in this repo are modified versions of those files.

### Demos
```bash
python load_g1.py    # visualize the robot, print joint/actuator structure
python balance.py    # PD balance with periodic torso disturbances
python ik_demo.py    # right arm IK tracking a moving target
```

---

## Technical notes

**Why hand-tuned PD instead of RL?** Modern humanoid pipelines (Figure, Apptronik, etc.) train walking policies via RL in Isaac Lab and deploy via sim-to-real transfer. RL is the right choice for dynamic locomotion. For *static stabilization* and *manipulation IK*, classical control is faster, easier to debug, and is the layer that actually runs underneath learned policies on real robots.

**Why DLS instead of analytical IK?** The G1 arm is 7-DOF (redundant — more joints than the 3 task-space dimensions), so analytical IK requires picking a parameterization for the null space (e.g., elbow elevation). DLS sidesteps this by solving numerically and naturally exploits redundancy via the pseudo-inverse. It's also the same family of methods used in industrial-arm controllers including the ABB IRC5.

**Coupling between arm and base.** A floating-base humanoid couples arm and base dynamics — fast arm motions push the base around (Newton's third law on a free-floating system). Tuning the IK gains required balancing target tracking against base disturbance. The pelvis-drift cutoff in `ik_demo.py` is a defensive backstop discovered through this tuning.

---

## Future work

- **Active ankle strategy** — add a torso-tilt feedback term to the ankle pitch/roll commands to extend the backward stability margin
- **Whole-body coordination** — let the waist and pelvis contribute to IK targets instead of holding them at the standing setpoint
- **RL walking policy** — port the model to Isaac Lab and train a PPO walking controller using domain randomization, then validate sim-to-sim back in MuJoCo

---

## Acknowledgments

Base humanoid model from [DeepMind's MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Unitree G1 (29-DOF revision).
