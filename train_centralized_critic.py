import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from multiagent_sumo_env import MultiAgentSUMOEnv
import traci

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Networks ----------

class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # logits


class CriticNet(nn.Module):
    def __init__(self, joint_obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(joint_obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # state value V(s)


def compute_returns(rewards, gamma=0.97):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def main():
    sumo_cfg = "config.sumocfg"
    env = MultiAgentSUMOEnv(sumo_cfg_path=sumo_cfg, use_gui=True,
                            max_steps=300, min_phase_duration=5)

    # initial reset to infer dimensions
    obs = env.reset()
    agent_ids = sorted(list(obs.keys()))
    print("Agents (TLS IDs):", agent_ids)

    obs_dims = {aid: obs[aid].shape[0] for aid in agent_ids}

    # get action dims from current SUMO connection (env already started it)
    act_dims = {}
    for aid in agent_ids:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(aid)[0]
        act_dims[aid] = len(logic.phases)

    joint_obs_dim = sum(obs_dims[aid] for aid in agent_ids)

    actors = {
        aid: ActorNet(obs_dims[aid], act_dims[aid]).to(device)
        for aid in agent_ids
    }
    critic = CriticNet(joint_obs_dim).to(device)

    actor_optims = {
        aid: optim.Adam(actors[aid].parameters(), lr=1e-3)
        for aid in agent_ids
    }
    critic_optim = optim.Adam(critic.parameters(), lr=1e-4)

    num_episodes = 500
    gamma = 0.97
    entropy_coef = 0.05

    for ep in range(num_episodes):
        obs = env.reset()

        joint_obs_list = []
        global_reward_list = []
        log_probs = {aid: [] for aid in agent_ids}

        done = False
        ep_len = 0

        while not done:
            joint_obs = np.concatenate([obs[aid] for aid in agent_ids], axis=0)
            joint_obs_list.append(joint_obs)

            actions = {}
            for aid in agent_ids:
                o = torch.tensor(obs[aid], dtype=torch.float32,
                                 device=device).unsqueeze(0)
                logits = actors[aid](o)
                dist = Categorical(logits=logits)
                a = dist.sample()
                actions[aid] = int(a.item())
                log_probs[aid].append((dist.log_prob(a), dist.entropy()))

            next_obs, rewards, done, info = env.step(actions)
            global_reward = info["global_reward"]
            global_reward_list.append(global_reward)

            obs = next_obs
            ep_len += 1

        joint_obs_tensor = torch.tensor(
            np.array(joint_obs_list),
            dtype=torch.float32,
            device=device
        )

        # normalize joint obs
        joint_obs_tensor = (joint_obs_tensor - joint_obs_tensor.mean(dim=0)) / \
                           (joint_obs_tensor.std(dim=0) + 1e-8)

        returns = compute_returns(global_reward_list, gamma=gamma)
        values = critic(joint_obs_tensor).squeeze(-1)

        # critic loss
        critic_loss = nn.functional.mse_loss(values, returns)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()

        with torch.no_grad():
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        for aid in agent_ids:
            lp_list, ent_list = zip(*log_probs[aid])
            lp = torch.stack(lp_list)
            ent = torch.stack(ent_list)

            actor_loss = -(lp * advantages).mean() - entropy_coef * ent.mean()

            actor_optims[aid].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actors[aid].parameters(), 1.0)
            actor_optims[aid].step()
            total_actor_loss += actor_loss.item()

        avg_return = float(np.mean(global_reward_list))
        print(f"Episode {ep+1}/{num_episodes} | "
              f"len={ep_len} | "
              f"avg global reward={avg_return:.2f} | "
              f"critic_loss={critic_loss.item():.3f} | "
              f"actor_loss_sum={total_actor_loss:.3f}")

    # save models
    for aid in agent_ids:
        torch.save(actors[aid].state_dict(), f"actor_{aid}.pth")
    torch.save(critic.state_dict(), "critic.pth")
    print("✅ Saved trained actor models and critic.")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
