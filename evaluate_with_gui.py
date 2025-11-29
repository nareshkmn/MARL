import time
import traci
from multiagent_sumo_env import MultiAgentSUMOEnv

sumo_cfg = "config.sumocfg"

# run with GUI in slow mode
env = MultiAgentSUMOEnv(
    sumo_cfg_path=sumo_cfg,
    use_gui=True,
    max_steps=300
)

obs = env.reset()

for step in range(300):

    # keep all lights fixed (or random) just for visualization
    actions = {}
    for aid in env.agent_ids:
        actions[aid] = 0   # always phase 0

    obs, rewards, done, info = env.step(actions)

    time.sleep(0.2)      # <<< slow down GUI

    if done:
        break

env.close()
