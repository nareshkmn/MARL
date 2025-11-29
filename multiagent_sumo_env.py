import os
import numpy as np
import traci


class MultiAgentSUMOEnv:
    """
    Multi-agent SUMO environment with one agent per traffic light.

    - Agents: all TLS IDs reported by SUMO (A1, B0, B1, B2, C1 for you)
    - Observation per agent: [q_in, q_out, pressure, current_phase]
        q_in      = sum of halting vehicles on incoming lanes
        q_out     = sum of halting vehicles on outgoing lanes
        pressure  = q_in - q_out
    - Action per agent: choose next phase index (0 .. num_phases-1),
      with a minimum phase duration to avoid flickering.
    - Reward per agent: local max-pressure reward:  r_i = -(q_in - q_out)
      Global reward (for critic): sum_i r_i
    """

    def __init__(self, sumo_cfg_path: str,
                 use_gui: bool = False,
                 max_steps: int = 300,
                 min_phase_duration: int = 5):
        self.sumo_cfg_path = os.path.abspath(sumo_cfg_path)
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.min_phase_duration = min_phase_duration

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"

        self.tls_ids = []
        self.agent_ids = []
        self.lanes_by_tls = {}
        self.phase_maps = {}

        self.in_lanes = {}
        self.out_lanes = {}

        self.sim_step = 0
        self.last_phase_change_step = {}

    # ---------- SUMO control ----------

    def _start_sumo(self):
        if traci.isLoaded():
            traci.close()
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg_path,
            "--step-length", "1.0",
            "--no-warnings", "true",
        ]
        traci.start(sumo_cmd)

    def _discover_tls_and_lanes(self):
        self.tls_ids = list(traci.trafficlight.getIDList())
        self.agent_ids = list(self.tls_ids)

        self.lanes_by_tls = {}
        self.phase_maps = {}
        self.in_lanes = {}
        self.out_lanes = {}

        for tls in self.tls_ids:
            # Number of phases (compatible with your SUMO version)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
            num_phases = len(logic.phases)
            self.phase_maps[tls] = list(range(num_phases))

            # Controlled lanes
            lanes = traci.trafficlight.getControlledLanes(tls)
            self.lanes_by_tls[tls] = sorted(list(set(lanes)))

            # Incoming / outgoing lanes via controlled links
            in_set = set()
            out_set = set()
            links = traci.trafficlight.getControlledLinks(tls)
            # links is a list of groups of (incoming, outgoing, via)
            for group in links:
                for inc, out, via in group:
                    if inc:
                        in_set.add(inc)
                    if out:
                        out_set.add(out)
            self.in_lanes[tls] = sorted(list(in_set))
            self.out_lanes[tls] = sorted(list(out_set))

    # ---------- Observation / reward helpers ----------

    def _get_state_for_tls(self, tls_id: str) -> np.ndarray:
        in_lanes = self.in_lanes[tls_id]
        out_lanes = self.out_lanes[tls_id]

        q_in = sum(traci.lane.getLastStepHaltingNumber(l) for l in in_lanes)
        q_out = sum(traci.lane.getLastStepHaltingNumber(l) for l in out_lanes)
        pressure = q_in - q_out
        current_phase = traci.trafficlight.getPhase(tls_id)

        state = np.array([q_in, q_out, pressure, current_phase],
                         dtype=np.float32)
        return state

    def _get_all_obs(self):
        obs = {}
        for tls in self.agent_ids:
            obs[tls] = self._get_state_for_tls(tls)
        return obs

    def _compute_rewards(self):
        """
        Local max-pressure:
            r_i = -(q_in - q_out)
        Global reward is sum of local rewards (used by centralized critic).
        """
        rewards = {}
        for tls in self.agent_ids:
            in_lanes = self.in_lanes[tls]
            out_lanes = self.out_lanes[tls]

            q_in = sum(traci.lane.getLastStepHaltingNumber(l) for l in in_lanes)
            q_out = sum(traci.lane.getLastStepHaltingNumber(l) for l in out_lanes)
            pressure = q_in - q_out

            rewards[tls] = -float(pressure)
        return rewards

    # ---------- Gym-like API ----------

    def reset(self):
        """Start a new SUMO episode and return initial multi-agent observations."""
        self._start_sumo()
        self._discover_tls_and_lanes()
        self.sim_step = 0
        self.last_phase_change_step = {tls: 0 for tls in self.tls_ids}
        return self._get_all_obs()

    def step(self, actions: dict):
        """
        actions: dict {tls_id: action_index}
        returns: obs_dict, rewards_dict, done, info
        """
        # apply actions with minimum phase duration
        for tls, act in actions.items():
            if tls not in self.phase_maps:
                continue
            desired_phase = int(self.phase_maps[tls][act])
            current_phase = traci.trafficlight.getPhase(tls)

            if desired_phase != current_phase:
                if self.sim_step - self.last_phase_change_step[tls] >= self.min_phase_duration:
                    traci.trafficlight.setPhase(tls, desired_phase)
                    self.last_phase_change_step[tls] = self.sim_step

        traci.simulationStep()
        self.sim_step += 1

        obs = self._get_all_obs()
        rewards = self._compute_rewards()

        global_reward = sum(rewards.values())

        done = False
        if traci.simulation.getMinExpectedNumber() <= 0:
            done = True
        if self.sim_step >= self.max_steps:
            done = True

        info = {
            "global_reward": global_reward,
            "sim_step": self.sim_step,
        }

        if done:
            traci.close()

        return obs, rewards, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()
