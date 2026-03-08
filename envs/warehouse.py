"""
Warehouse Environment Wrapper for RWARE.

Adds navigation observations, reward shaping, and battery management
with charging station support.

Mission flow:
  SEEK_SHELF -> (pickup) -> DELIVER -> (drop at goal) -> SEEK_SHELF (repeat)
                 | battery low and not carrying?
              CHARGING -> (at charger, battery >= resume) -> SEEK_SHELF

Battery:
  - Drains each step when not at charger
  - When battery < threshold and agent is NOT carrying: enter CHARGING state
  - Agent navigates to charger, charges until battery >= resume level
  - Then resumes mission (SEEK_SHELF)
  - If battery hits 0: episode ends with penalty

Based on the main branch's working reward system, extended with charging.
"""

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box, Tuple as TupleSpace
import numpy as np
import rware
from rware.warehouse import Warehouse, RewardType

# Mission states
SEEK_SHELF = 0
DELIVER = 1
CHARGING = 2

# Extra observation dimensions
# pos(2) + target(2) + dir_to_target(2) + facing_onehot(4) + carrying(1) + battery(1) + agent_id(1)
EXTRA_OBS_DIMS = 13


class WarehouseWrapper(Wrapper):
    """
    Wraps RWARE with navigation observations and reward shaping.
    Uses the main branch's proven reward structure plus battery/charging.

    Extended observations (12 dims appended):
      - Agent position (x, y) normalized
      - Target position (x, y) -- shelf, goal, or charger depending on state
      - Direction to target (dx, dy) normalized
      - Agent facing direction one-hot [UP, DOWN, LEFT, RIGHT]
      - Carrying flag (0/1)
      - Battery level (0-1 normalized)

    Reward shaping (from main branch, proven to work + battery management):
      - Step penalty: -0.5
      - Potential-based distance shaping toward correct target (scale 3.0)
      - Wall-bump penalty: -1.0
      - Toggle at correct location: +3.0
      - Pointless toggle: -0.5
      - Pickup milestone: +20.0
      - Delivery milestone: +50.0
      - Mission completion: +100.0 + 50.0 * (battery_remaining / max_battery)
      - Battery death: -50.0
      - Charging at station: +5.0 per step
      - Recharge complete: +10.0 bonus
    """

    def __init__(self, env, battery_config=None, max_deliveries=1):
        super().__init__(env)
        self.max_deliveries = max_deliveries

        # Battery config
        bc = battery_config or {}
        self.max_battery = bc.get("max_battery", 100.0)
        self.battery_drain = bc.get("battery_drain", 0.3)
        self.charge_rate = bc.get("charge_rate", 25.0)
        self.battery_threshold = bc.get("battery_threshold", 25.0)
        self.battery_resume = bc.get("battery_resume", 70.0)
        # Support multiple charger locations (one per agent)
        charger_locs = bc.get("charger_locations", None)
        if charger_locs is None:
            single = bc.get("charger_location", (0, 0))
            charger_locs = [single, single]
        self.charger_locations = [tuple(loc) for loc in charger_locs]
        self._auto_charger_positions = bc.get("auto_charger_positions", True)
        # Assign charger number to each agent: agent 0 -> charger 0, agent 1 -> charger 1
        self.agent_charger_numbers = [i % len(self.charger_locations) for i in range(self.n_agents)]

        self._setup_observation_space()

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _setup_observation_space(self):
        orig = self.env.observation_space
        if isinstance(orig, TupleSpace):
            new_spaces = []
            for sp in orig.spaces:
                lo = np.concatenate([sp.low, [-1.0] * 2, [0.0] * (EXTRA_OBS_DIMS - 2)])
                hi = np.concatenate([sp.high, [1.0] * 6, [1.0] * 4, [1.0], [1.0], [1.0]])
                new_spaces.append(Box(low=lo.astype(np.float32), high=hi.astype(np.float32)))
            self.observation_space = TupleSpace(new_spaces)
        else:
            lo = np.concatenate([orig.low, [-1.0] * 2, [0.0] * (EXTRA_OBS_DIMS - 2)])
            hi = np.concatenate([orig.high, [1.0] * 6, [1.0] * 4, [1.0], [1.0], [1.0]])
            self.observation_space = Box(low=lo.astype(np.float32), high=hi.astype(np.float32))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def n_agents(self):
        return getattr(self.env.unwrapped, "n_agents", 1)

    def _agent_positions(self):
        try:
            return [(a.x, a.y) for a in self.env.unwrapped.agents]
        except Exception:
            return [(0, 0)] * self.n_agents

    def _requested_shelves(self):
        try:
            return [(s.x, s.y) for s in self.env.unwrapped.request_queue]
        except Exception:
            return []

    def _all_shelf_positions(self):
        try:
            return [(s.x, s.y) for s in self.env.unwrapped.shelfs]
        except Exception:
            return []

    def _non_requested_shelves(self):
        requested = set(self._requested_shelves())
        return [s for s in self._all_shelf_positions() if s not in requested]

    def _goal_positions(self):
        try:
            return list(self.env.unwrapped.goals)
        except Exception:
            return []

    def _facing_onehot(self, idx):
        try:
            d = self.env.unwrapped.agents[idx].dir.value
            oh = [0.0, 0.0, 0.0, 0.0]
            if 0 <= d <= 3:
                oh[d] = 1.0
            return oh
        except Exception:
            return [1.0, 0.0, 0.0, 0.0]

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _min_dist(self, pos, targets):
        if not targets:
            return 0
        return min(self._manhattan(pos, t) for t in targets)

    def _is_at_charger(self, pos, agent_idx=0):
        charger_idx = self.agent_charger_numbers[agent_idx % len(self.agent_charger_numbers)]
        return pos == self.charger_locations[charger_idx]

    def _compute_optimal_steps(self):
        positions = self._agent_positions()
        shelves = self._requested_shelves()
        goals = self._goal_positions()
        if not positions or not shelves or not goals:
            return 20
        p = positions[0]
        s = min(shelves, key=lambda s: self._manhattan(p, s))
        g = min(goals, key=lambda g: self._manhattan(s, g))
        return self._manhattan(p, s) + self._manhattan(s, g) + 4

    # ------------------------------------------------------------------
    # Build extended observation
    # ------------------------------------------------------------------

    def _extend_obs(self, obs):
        gh, gw = self.env.unwrapped.grid_size
        positions = self._agent_positions()
        goals = self._goal_positions()
        shelves = self._requested_shelves()

        def _build_extra(i, single_obs):
            pos = positions[i] if i < len(positions) else (0, 0)
            carrying = float(self.env.unwrapped.agents[i].carrying_shelf is not None)
            battery = self.battery_levels[i] if i < len(self.battery_levels) else self.max_battery

            goal = min(goals, key=lambda g: self._manhattan(pos, g)) if goals else (gw // 2, gh)
            shelf = min(shelves, key=lambda s: self._manhattan(pos, s)) if shelves else (gw // 2, gh // 2)

            # Target depends on mission state
            mission = self.mission_state[i] if i < len(self.mission_state) else SEEK_SHELF
            if mission == CHARGING:
                charger_idx = self.agent_charger_numbers[i] if i < len(self.agent_charger_numbers) else 0
                target = self.charger_locations[charger_idx]
            elif carrying > 0.5:
                target = goal
            else:
                target = shelf

            dx = (target[0] - pos[0]) / max(gw, 1)
            dy = (target[1] - pos[1]) / max(gh, 1)
            facing = self._facing_onehot(i)

            extra = np.array([
                pos[0] / max(gw, 1), pos[1] / max(gh, 1),
                target[0] / max(gw, 1), target[1] / max(gh, 1),
                dx, dy,
                *facing,
                carrying, battery / self.max_battery,
                i / max(self.n_agents - 1, 1),  # agent id normalised 0..1
            ], dtype=np.float32)
            return np.concatenate([single_obs.astype(np.float32), extra])

        if isinstance(obs, tuple):
            return tuple(_build_extra(i, o) for i, o in enumerate(obs))
        return _build_extra(0, obs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Auto-compute charger positions from grid corners (one per agent)
        if self._auto_charger_positions:
            gh, gw = self.env.unwrapped.grid_size
            self.charger_locations = [
                (0, 0),        # Charger 0: top-left corner  (agent 0)
                (gw - 1, 0),   # Charger 1: top-right corner (agent 1)
            ]
            self.agent_charger_numbers = [i % len(self.charger_locations) for i in range(self.n_agents)]

        # Place each agent one step inside from its assigned charger at episode start
        for i, agent in enumerate(self.env.unwrapped.agents):
            charger = self.charger_locations[self.agent_charger_numbers[i]]
            cx, cy = charger
            agent.x = cx + 1 if cx == 0 else cx - 1
            agent.y = cy
        self.env.unwrapped._recalc_grid()

        self.battery_levels = [self.max_battery] * self.n_agents
        self.deliveries_count = 0
        self.agent_deliveries = [0] * self.n_agents
        self.charging_events = 0
        self.total_steps = 0
        self.segment_steps = 0
        self.mission_state = [SEEK_SHELF] * self.n_agents
        self.prev_potential = [None] * self.n_agents
        self.delivery_segment_steps = []
        self.agent_shelf_claim = [None] * self.n_agents  # exclusive shelf target per agent

        # Store original shelf positions so we can return shelves after delivery
        self._shelf_home = {}
        try:
            for s in self.env.unwrapped.shelfs:
                self._shelf_home[s.id] = (s.x, s.y)
        except Exception:
            pass

        obs = self._extend_obs(obs)
        info["battery_levels"] = self.battery_levels.copy()
        info["deliveries"] = 0
        return obs, info

    # ------------------------------------------------------------------
    # Target and potential (same as main branch)
    # ------------------------------------------------------------------

    def _assign_shelf(self, i, pos, shelves):
        """Assign the nearest unclaimed shelf to agent i.
        If all shelves are claimed, fall back to nearest overall."""
        claimed = {self.agent_shelf_claim[j] for j in range(self.n_agents) if j != i and self.agent_shelf_claim[j] is not None}
        free = [s for s in shelves if s not in claimed]
        pool = free if free else shelves
        best = min(pool, key=lambda s: self._manhattan(pos, s))
        self.agent_shelf_claim[i] = best
        return best

    def _get_target(self, i, pos, carrying, battery):
        """Determine what the agent should be navigating toward."""
        goals = self._goal_positions()
        shelves = self._requested_shelves()

        if self.mission_state[i] == CHARGING:
            charger_idx = self.agent_charger_numbers[i]
            return self.charger_locations[charger_idx]
        elif carrying:
            if goals:
                return min(goals, key=lambda g: self._manhattan(pos, g))
            return pos
        else:
            if shelves:
                # Use existing claim if still valid, else assign a new one
                claim = self.agent_shelf_claim[i]
                if claim is None or claim not in shelves:
                    claim = self._assign_shelf(i, pos, shelves)
                return claim
            return pos

    def _potential(self, pos, target):
        """Potential = negative Manhattan distance to target."""
        return -(abs(pos[0] - target[0]) + abs(pos[1] - target[1]))

    # ------------------------------------------------------------------
    # Step -- main branch reward logic + battery charging
    # ------------------------------------------------------------------

    def step(self, action):
        was_carrying = [a.carrying_shelf is not None for a in self.env.unwrapped.agents]
        was_carrying_shelf = [a.carrying_shelf for a in self.env.unwrapped.agents]
        old_pos = self._agent_positions()

        # --- Action masking ---
        requested = set(self._requested_shelves())
        goals = set(self._goal_positions())
        action = list(action) if hasattr(action, "__iter__") else [action]
        for i in range(len(action)):
            if action[i] == 4:
                agent_pos = old_pos[i] if i < len(old_pos) else (0, 0)
                if not was_carrying[i] and agent_pos not in requested:
                    action[i] = 0  # prevent picking up non-requested shelf
                elif was_carrying[i] and agent_pos not in goals:
                    action[i] = 0  # prevent dropping shelf at non-goal
        action = tuple(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.segment_steps += 1

        now_carrying = [a.carrying_shelf is not None for a in self.env.unwrapped.agents]
        pos = self._agent_positions()
        shelves = self._requested_shelves()
        goals = self._goal_positions()

        # ---- Detect RWARE delivery (env_r > 0) ----
        env_r = sum(reward) if isinstance(reward, (list, tuple)) else reward

        # After RWARE delivery: return ONLY the delivered agent's shelf to its home
        # position. Use the actual post-step carrying state (RWARE already cleared
        # carrying_shelf for the delivering agent) so other agents keep their shelves.
        if env_r > 0:
            for i in range(self.n_agents):
                # Agent actually delivered: was carrying, now not carrying (RWARE cleared it)
                if was_carrying[i] and not now_carrying[i]:
                    shelf = was_carrying_shelf[i]
                    if shelf is not None:
                        home = self._shelf_home.get(shelf.id, None)
                        if home is not None:
                            shelf.x, shelf.y = home
            self.env.unwrapped._recalc_grid()
            # now_carrying reflects RWARE's real state — do NOT override it

        # ---- Reset potential when carrying state changes (target switches) ----
        for i in range(self.n_agents):
            if was_carrying[i] != now_carrying[i]:
                self.prev_potential[i] = None

        # ---- Reward (main branch logic) ----
        r = 0.0

        for i in range(self.n_agents):
            act = action[i] if hasattr(action, "__iter__") else action
            p = pos[i]
            battery = self.battery_levels[i]
            carrying = now_carrying[i]

            r -= 0.5  # per-agent step penalty (each agent contributes independently)

            # 1) Potential-based distance shaping (from main branch)
            target = self._get_target(i, p, carrying, battery)
            phi_now = self._potential(p, target)

            if self.prev_potential[i] is not None:
                shaping = 0.99 * phi_now - self.prev_potential[i]
                r += 3.0 * shaping

            self.prev_potential[i] = phi_now

            # 2) Wall-bump penalty (only FORWARD=1, not NOOP=0)
            if p == old_pos[i] and act == 1:
                r -= 1.0

            # 3) Toggle logic (wrong-shelf pickup already blocked by action masking)
            if act == 4:
                if not was_carrying[i] and p in set(shelves):
                    r += 3.0   # toggle at REQUESTED shelf -- good
                elif was_carrying[i] and p in set(goals):
                    r += 3.0   # toggle at goal to deliver -- good
                else:
                    r -= 0.5   # pointless toggle

        # ---- Battery drain/charge ----
        for i in range(self.n_agents):
            if self._is_at_charger(pos[i], i):
                self.battery_levels[i] = min(
                    self.max_battery,
                    self.battery_levels[i] + self.charge_rate
                )
            else:
                self.battery_levels[i] = max(
                    0, self.battery_levels[i] - self.battery_drain
                )

        # ---- CHARGING state transitions ----
        for i in range(self.n_agents):
            battery = self.battery_levels[i]

            # Enter CHARGING: battery low and not carrying
            if (self.mission_state[i] != CHARGING
                    and battery < self.battery_threshold
                    and not now_carrying[i]):
                self.mission_state[i] = CHARGING
                self.prev_potential[i] = None  # reset shaping toward charger
                self.agent_shelf_claim[i] = None  # release claim while charging

            # At charger while CHARGING
            if self.mission_state[i] == CHARGING:
                if self._is_at_charger(pos[i], i):
                    r += 5.0  # reward for being at charger when needed
                    if battery >= self.battery_resume:
                        self.mission_state[i] = SEEK_SHELF
                        self.prev_potential[i] = None  # reset toward next shelf
                        r += 10.0  # bonus for successful recharge
                        self.charging_events += 1

        # ---- Pickup milestone ----
        for i in range(self.n_agents):
            if (self.mission_state[i] == SEEK_SHELF
                    and not was_carrying[i] and now_carrying[i]):
                r += 20.0
                self.mission_state[i] = DELIVER
                self.agent_shelf_claim[i] = None  # release claim on pickup

            # Wrong drop (dropped shelf off-goal)
            if self.mission_state[i] == DELIVER and was_carrying[i] and not now_carrying[i]:
                check_r = sum(reward) if isinstance(reward, (list, tuple)) else reward
                if check_r <= 0:
                    r -= 30.0
                    self.mission_state[i] = SEEK_SHELF

        # ---- Delivery milestone ----
        if env_r > 0:
            # Track which agent(s) actually delivered this step
            for i in range(self.n_agents):
                if was_carrying[i] and not now_carrying[i]:
                    self.agent_deliveries[i] += 1
                    self.deliveries_count += 1
                    self.agent_shelf_claim[i] = None  # ensure claim cleared after delivery
                    self.mission_state[i] = SEEK_SHELF
                    self.prev_potential[i] = None
            r += 50.0

            # Track segment steps
            self.delivery_segment_steps.append(self.segment_steps)
            self.segment_steps = 0

        # ---- Termination ----
        battery_dead = any(b <= 0 for b in self.battery_levels)
        # Episode ends when every agent has completed its own max_deliveries quota
        mission_done = all(self.agent_deliveries[i] >= self.max_deliveries for i in range(self.n_agents))

        if battery_dead:
            terminated = True
            r -= 50.0  # battery death penalty

        if mission_done:
            terminated = True
            # Reward completing all deliveries + bonus for battery efficiency
            battery_ratio = min(self.battery_levels) / self.max_battery
            r += 100.0 + 50.0 * battery_ratio

        obs = self._extend_obs(obs)
        info.update({
            "battery_levels": self.battery_levels.copy(),
            "deliveries": self.deliveries_count,
            "agent_deliveries": self.agent_deliveries.copy(),
            "max_deliveries": self.max_deliveries,
            "mission_complete": mission_done,
            "battery_dead": battery_dead,
            "battery_remaining": min(self.battery_levels),
            "charging_events": self.charging_events,
            "delivery_segment_steps": self.delivery_segment_steps.copy(),
            "agent_stuck": False,
        })
        return obs, float(r), terminated, truncated, info


def make_env(env_config, battery_config=None):
    """Create RWARE environment with wrapper."""
    n_agents = env_config.get("n_agents", 1)
    env = Warehouse(
        shelf_columns=env_config.get("shelf_columns", 3),
        column_height=env_config.get("column_height", 1),
        shelf_rows=env_config.get("shelf_rows", 1),
        n_agents=n_agents,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=max(n_agents, 2),
        max_inactivity_steps=None,
        max_steps=env_config.get("max_steps", 100),
        reward_type=RewardType.INDIVIDUAL,
    )
    return WarehouseWrapper(env, battery_config=battery_config,
                            max_deliveries=env_config.get("max_deliveries", 1))
