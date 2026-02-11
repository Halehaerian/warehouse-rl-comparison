"""
Warehouse Environment Wrapper for RWARE.

Adds navigation observations, reward shaping, and optional battery management.

Mission flow: SEEK_SHELF → PICKUP → DELIVER → DROP → repeat
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

# Extra observation dimensions
# pos(2) + target(2) + dir_to_target(2) + facing_onehot(4) + carrying(1) + battery(1)
EXTRA_OBS_DIMS = 12


class WarehouseWrapper(Wrapper):
    """
    Wraps RWARE with navigation observations and reward shaping.

    Extended observations (12 dims appended):
      - Agent position (x, y) normalized
      - Target position (x, y) — shelf or goal depending on carrying state
      - Direction to target (dx, dy) normalized
      - Agent facing direction one-hot [UP, DOWN, LEFT, RIGHT]
      - Carrying flag (0/1)
      - Battery level (0–100)

    Reward shaping:
      - Growing step penalty
      - Distance-based shaping toward correct target
      - Arrival bonuses, toggle hints, hesitation penalties
      - Pickup (+150), delivery (500 * efficiency), completion bonus
    """

    def __init__(self, env, battery_config=None, max_deliveries=1):
        super().__init__(env)
        self.max_deliveries = max_deliveries

        # Battery config (optional — disable by setting drain to 0)
        bc = battery_config or {}
        self.max_battery = bc.get("max_battery", 100.0)
        self.battery_drain = bc.get("battery_drain", 0.1)
        self.charge_rate = bc.get("charge_rate", 50.0)
        self.battery_threshold = bc.get("battery_threshold", 5.0)
        self.charger_location = tuple(bc.get("charger_location", (0, 0)))

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
                hi = np.concatenate([sp.high, [1.0] * 6, [1.0] * 4, [1.0], [1.0]])
                new_spaces.append(Box(low=lo.astype(np.float32), high=hi.astype(np.float32)))
            self.observation_space = TupleSpace(new_spaces)
        else:
            lo = np.concatenate([orig.low, [-1.0] * 2, [0.0] * (EXTRA_OBS_DIMS - 2)])
            hi = np.concatenate([orig.high, [1.0] * 6, [1.0] * 4, [1.0], [1.0]])
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
        """Return positions of ALL shelves (requested + non-requested)."""
        try:
            return [(s.x, s.y) for s in self.env.unwrapped.shelfs]
        except Exception:
            return []

    def _non_requested_shelves(self):
        """Return positions of non-requested shelves."""
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

    def _min_dist(self, pos, targets):
        if not targets:
            return 0
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _is_at_charger(self, pos):
        return abs(pos[0] - self.charger_location[0]) + abs(pos[1] - self.charger_location[1]) <= 1

    def _compute_optimal_steps(self):
        positions = self._agent_positions()
        shelves = self._requested_shelves()
        goals = self._goal_positions()
        if not positions or not shelves or not goals:
            return 20
        p = positions[0]
        s = min(shelves, key=lambda s: abs(s[0] - p[0]) + abs(s[1] - p[1]))
        g = min(goals, key=lambda g: abs(g[0] - s[0]) + abs(g[1] - s[1]))
        return abs(p[0] - s[0]) + abs(p[1] - s[1]) + abs(s[0] - g[0]) + abs(s[1] - g[1]) + 4

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

            goal = min(goals, key=lambda g: abs(g[0] - pos[0]) + abs(g[1] - pos[1])) if goals else (gw // 2, gh)
            shelf = min(shelves, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1])) if shelves else (gw // 2, gh // 2)

            if battery < self.battery_threshold:
                target = self.charger_location
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
        self.battery_levels = [self.max_battery] * self.n_agents
        self.deliveries_count = 0
        self.total_steps = 0
        self.mission_state = [SEEK_SHELF] * self.n_agents
        self.prev_potential = [None] * self.n_agents

        obs = self._extend_obs(obs)
        info["battery_levels"] = self.battery_levels.copy()
        info["deliveries"] = 0
        return obs, info

    # ------------------------------------------------------------------
    # Step — clean potential-based reward shaping
    # ------------------------------------------------------------------

    def _get_target(self, i, pos, carrying, battery):
        """Determine what the agent should be navigating toward."""
        goals = self._goal_positions()
        shelves = self._requested_shelves()

        if battery < self.battery_threshold and not carrying:
            return self.charger_location
        elif carrying:
            if goals:
                return min(goals, key=lambda g: abs(g[0] - pos[0]) + abs(g[1] - pos[1]))
            return pos
        else:
            if shelves:
                return min(shelves, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1]))
            return pos

    def _potential(self, pos, target):
        """Potential = negative Manhattan distance to target."""
        return -(abs(pos[0] - target[0]) + abs(pos[1] - target[1]))

    def step(self, action):
        was_carrying = [a.carrying_shelf is not None for a in self.env.unwrapped.agents]
        old_pos = self._agent_positions()

        # --- Action masking: prevent picking up non-requested shelves ---
        requested = set(self._requested_shelves())
        action = list(action) if hasattr(action, "__iter__") else [action]
        for i in range(len(action)):
            if action[i] == 4 and not was_carrying[i]:
                # Agent wants to toggle but isn't carrying — only allow at requested shelf
                agent_pos = old_pos[i] if i < len(old_pos) else (0, 0)
                if agent_pos not in requested:
                    action[i] = 0  # convert to NOOP
        action = tuple(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        now_carrying = [a.carrying_shelf is not None for a in self.env.unwrapped.agents]
        pos = self._agent_positions()
        shelves = self._requested_shelves()
        goals = self._goal_positions()

        # ---- Reward ----
        r = -0.5  # small constant step penalty

        for i in range(self.n_agents):
            act = action[i] if hasattr(action, "__iter__") else action
            p = pos[i]
            battery = self.battery_levels[i]
            carrying = now_carrying[i]

            # 1) Potential-based distance shaping
            target = self._get_target(i, p, carrying, battery)
            phi_now = self._potential(p, target)

            if self.prev_potential[i] is not None:
                # Potential-based shaping: r += gamma * phi(s') - phi(s)
                shaping = 0.99 * phi_now - self.prev_potential[i]
                r += 3.0 * shaping  # scale factor

            self.prev_potential[i] = phi_now

            # 2) Wall-bump / NOOP penalty
            if p == old_pos[i] and act in (0, 1):
                r -= 1.0

            # 3) Toggle logic (wrong-shelf pickup is already blocked by action masking)
            if act == 4:
                if not was_carrying[i] and p in shelves:
                    r += 3.0   # toggle at REQUESTED shelf — good
                elif was_carrying[i] and p in goals:
                    r += 3.0   # toggle at goal to deliver — good
                else:
                    r -= 0.5   # pointless toggle

        # ---- Battery ----
        for i in range(self.n_agents):
            if self._is_at_charger(pos[i]):
                self.battery_levels[i] = min(self.max_battery,
                                             self.battery_levels[i] + self.charge_rate)
            else:
                self.battery_levels[i] = max(0,
                                             self.battery_levels[i] - self.battery_drain)

        # ---- Pickup milestone (wrong shelf pickup is blocked by action masking) ----
        for i in range(self.n_agents):
            if (self.mission_state[i] == SEEK_SHELF
                    and not was_carrying[i] and now_carrying[i]):
                r += 20.0
                self.mission_state[i] = DELIVER

            # Wrong drop
            if self.mission_state[i] == DELIVER and was_carrying[i] and not now_carrying[i]:
                env_r = sum(reward) if isinstance(reward, (list, tuple)) else reward
                if env_r <= 0:
                    r -= 30.0
                    self.mission_state[i] = SEEK_SHELF

        # ---- Delivery milestone ----
        env_r = sum(reward) if isinstance(reward, (list, tuple)) else reward
        if env_r > 0:
            self.deliveries_count += 1
            r += 50.0
            for i in range(self.n_agents):
                if not now_carrying[i]:
                    self.mission_state[i] = SEEK_SHELF

        # ---- Termination ----
        battery_dead = any(b <= 0 for b in self.battery_levels)
        mission_done = self.deliveries_count >= self.max_deliveries

        if battery_dead:
            terminated = True
            r -= 20.0
        if mission_done:
            terminated = True
            r += 50.0

        obs = self._extend_obs(obs)
        info.update({
            "battery_levels": self.battery_levels.copy(),
            "deliveries": self.deliveries_count,
            "mission_complete": mission_done,
            "battery_dead": battery_dead,
            "agent_stuck": False,
        })
        return obs, float(r), terminated, truncated, info


def make_env(env_config, battery_config=None):
    """Create RWARE environment with wrapper.

    Args:
        env_config: dict with max_steps, max_deliveries, and optionally
                    shelf_columns, column_height, shelf_rows, n_agents.
        battery_config: dict passed to WarehouseWrapper (optional).
    """
    env = Warehouse(
        shelf_columns=env_config.get("shelf_columns", 3),
        column_height=env_config.get("column_height", 1),
        shelf_rows=env_config.get("shelf_rows", 1),
        n_agents=env_config.get("n_agents", 1),
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        max_inactivity_steps=None,
        max_steps=env_config.get("max_steps", 100),
        reward_type=RewardType.INDIVIDUAL,
    )
    return WarehouseWrapper(env, battery_config=battery_config,
                            max_deliveries=env_config.get("max_deliveries", 1))
