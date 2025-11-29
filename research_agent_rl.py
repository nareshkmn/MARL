# research_agent_rl.py
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
import random

from research_agents import ResearchAgentSystem


class MultiAgentReinforcementLearning:
    """
    Simple multi-agent Q-learning over collaboration "styles".
    Each agent chooses an action a_i in {0, 1}:
      0 = conservative style
      1 = exploratory style
    Currently we only use actions for learning; later you can
    feed them back into prompts/temperatures per agent.
    """

    def __init__(self, num_agents: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        self.num_agents = num_agents
        self.q_table = defaultdict(lambda: np.zeros((self.num_agents, 2), dtype=float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def select_action(self, state: str, agent_id: int) -> int:
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)
        q_values = self.q_table[state][agent_id]
        return int(np.argmax(q_values))

    def update_q_value(self, state: str, action: int, reward: float,
                       next_state: str, agent_id: int):
        q_values_next = self.q_table[next_state][agent_id]
        best_next_action = int(np.argmax(q_values_next))
        td_target = reward + self.discount_factor * q_values_next[best_next_action]
        td_error = td_target - self.q_table[state][agent_id][action]
        self.q_table[state][agent_id][action] += self.learning_rate * td_error

    # ---------- Reward shaping ----------

    def calculate_collaboration_reward(
        self, agent_interactions: List[Dict[str, Any]]
    ) -> float:
        if not agent_interactions:
            return 0.0

        diversity_score = self._calculate_diversity(agent_interactions)
        coherence_score = self._calculate_coherence(agent_interactions)
        novelty_score = self._calculate_novelty(agent_interactions)

        total_reward = 0.4 * diversity_score + 0.4 * coherence_score + 0.2 * novelty_score
        return float(total_reward)

    def _calculate_diversity(self, interactions: List[Dict[str, Any]]) -> float:
        approaches = [i.get("approach", "") for i in interactions]
        unique_approaches = len(set(approaches))
        return min(unique_approaches / max(len(approaches), 1), 1.0)

    def _calculate_coherence(self, interactions: List[Dict[str, Any]]) -> float:
        types = {i.get("task_type", "") for i in interactions}
        needed = {"literature", "critique", "synthesis", "coordination"}
        overlap = len(types.intersection(needed))
        return overlap / len(needed)

    def _calculate_novelty(self, interactions: List[Dict[str, Any]]) -> float:
        count_syn = sum(i.get("task_type") == "synthesis" for i in interactions)
        count_coord = sum(i.get("task_type") == "coordination" for i in interactions)
        total = len(interactions)
        if total == 0:
            return 0.0
        return min((count_syn + count_coord) / total, 1.0)


class EnhancedResearchAgentSystem(ResearchAgentSystem):
    """
    Wraps ResearchAgentSystem with a MARL layer:
    - Runs several episodes
    - Learns a collaboration policy
    - Returns the best episode by collaboration score
    """

    def __init__(self, topic: str, num_agents: int = 4):
        super().__init__(topic)
        self.rl_system = MultiAgentReinforcementLearning(num_agents=num_agents)

    def run_enhanced_research(self, num_episodes: int = 3) -> Dict[str, Any]:
        best_result: Dict[str, Any] | None = None
        best_score = -float("inf")

        state = "start"

        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

            actions = [
                self.rl_system.select_action(state, agent_id=i)
                for i in range(self.rl_system.num_agents)
            ]
            print(f"Chosen actions (per agent): {actions}")

            results = super().run_research()

            reward = self.rl_system.calculate_collaboration_reward(
                results.get("agent_interactions", [])
            )
            print(f"[Episode {episode + 1}] Collaboration Reward = {reward:.3f}")

            next_state = "start"
            for i, a in enumerate(actions):
                self.rl_system.update_q_value(
                    state=state,
                    action=a,
                    reward=reward,
                    next_state=next_state,
                    agent_id=i,
                )

            if reward > best_score:
                best_score = reward
                best_result = results

        if best_result is None:
            raise RuntimeError("No successful episodes")

        best_result["best_collaboration_score"] = best_score
        return best_result
