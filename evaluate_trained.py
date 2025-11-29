import numpy as np
import torch
import traci

from train_centralized_critic import ActorNet   # ✅ correct import
from multiagent_sumo_env import MultiAgentSUMOEnv

# ---- Setup environment ----
env = MultiAgentSUMOEnv("config.sumocfg", use_gui=False, max_steps=300)
obs = env.reset()

agent_ids = list(obs.keys())
agent_ids.sort()

# ---- Load trained actors ----
actors = {}
for aid in agent_ids:
    obs_dim = obs[aid].shape[0]

    # determine action dimension via SUMO
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(aid)[0]
    act_dim = len(logic.phases)

    model = ActorNet(obs_dim, act_dim)
    model.load_state_dict(torch.load(f"actor_{aid}.pth"))
    model.eval()
    actors[aid] = model

queue_log = []

# ---- Run evaluation episode ----
for step in range(300):

    actions = {}
    for aid in agent_ids:
        obs_tensor = torch.FloatTensor(obs[aid]).unsqueeze(0)
        with torch.no_grad():
            logits = actors[aid](obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
        actions[aid] = int(action)

    obs, rewards, done, info = env.step(actions)

    # ---- Compute total queue count BEFORE TraCI closes ----
    if traci.isLoaded():
        lanes = traci.lane.getIDList()
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    else:
        total_queue = 0

    queue_log.append(total_queue)

    if done:
        break

env.close()

np.save("trained_queue.npy", np.array(queue_log))
print("✅ Trained evaluation complete.")
