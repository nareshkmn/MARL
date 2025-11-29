import numpy as np
import traci
from multiagent_sumo_env import MultiAgentSUMOEnv

env = MultiAgentSUMOEnv("config.sumocfg", use_gui=False, max_steps=300)
obs = env.reset()

queue_log = []

for step in range(300):

    # fixed action baseline
    actions = {aid: 0 for aid in env.agent_ids}

    # step environment
    obs, rewards, done, info = env.step(actions)

    # ---- GET QUEUE BEFORE TRACI CLOSES ----
    if traci.isLoaded():
        lanes = traci.lane.getIDList()
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    else:
        total_queue = 0

    queue_log.append(total_queue)

    if done:
        break

# now safe to close
env.close()

np.save("baseline_queue.npy", np.array(queue_log))
print("✅ Baseline evaluation complete.")
