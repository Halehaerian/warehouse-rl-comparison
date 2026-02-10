"""
Battery Wrapper for RWARE Environment
Adds battery management and mission-state tracking to RWARE.

Mission flow per agent:
  1. SEEK_SHELF: Go to pickup point (requested shelf)
  2. PICKUP: Toggle to grab item (ht: 0 -> 1)
  3. DELIVER: Go to destination (goal zone)
  4. DROP: Toggle to deliver (ht: 1 -> 0)
  5. Repeat from step 1
  
  * If battery low (bt < threshold): Go to charger to recharge

Observation extended with:
  - Agent position (x, y) normalized
  - Target position (x, y) - shelf if not carrying, goal if carrying
  - Charger position (x, y) normalized
  - ht ∈ {0, 1}: Item held status
  - bt ∈ [0, 100]: Battery level

Reward function (Eq. 2):
    +100 * 1[delivery]  +10 * 1[pickup]
     -50 * 1[battery_death]  -1 * 1[step]
      +5 * 1[good_charge]   -2 * 1[wasteful_charge]
"""

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box, Tuple as TupleSpace
import numpy as np
import rware

# Mission states
SEEK_SHELF = 0   # Go to pickup point
DELIVER    = 1   # Go to destination (goal)


class BatteryWrapper(Wrapper):
    """
    Wraps RWARE with battery management and proper mission-state tracking.

    RWARE mechanics (unchanged):
      - Agents navigate, pick up shelves, deliver to goal zones.
      - Actions: 0=NOOP, 1=FORWARD, 2=LEFT, 3=RIGHT, 4=TOGGLE_LOAD

    Added mechanics:
      - Battery drains every step; charger restores it.
      - Mission state machine prevents reward-hacking (toggle exploit).
      - 30-second wall-clock time limit per episode.
    """

    def __init__(self, env, max_battery=100.0, battery_drain=0.2,
                 charge_rate=5.0, battery_threshold=20.0,
                 charger_location=(0, 0), max_deliveries=5):
        super().__init__(env)

        # Battery
        self.max_battery = max_battery
        self.battery_drain = battery_drain
        self.charge_rate = charge_rate
        self.battery_threshold = battery_threshold
        self.charger_location = charger_location

        # Mission
        self.max_deliveries = max_deliveries

        # Will be set in reset()
        self.battery_levels = None
        self.deliveries_count = 0
        self.total_steps = 0
        self.mission_state = None             # per-agent mission state
        self.pickup_cooldown = None           # ANTI-EXPLOIT: prevent pickup spam
        self.steps_at_goal_carrying = None    # Track how long agent camps at goal

        self.trip_start_step = None           # Track when pickup happened for speed bonus

        self._setup_observation_space()

        print(f"\n{'='*60}")
        print("RWARE + BATTERY WRAPPER INITIALIZED")
        print(f"{'='*60}")
        print(f"  Max Battery     : {max_battery}%")
        print(f"  Drain / step    : {battery_drain}%")
        print(f"  Charge / step   : {charge_rate}%")
        print(f"  Critical        : {battery_threshold}%")
        print(f"  Charger at      : {charger_location}")
        print(f"  Deliveries goal : {max_deliveries}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _setup_observation_space(self):
        """
        Extend observation with:
        - agent position normalized (2)
        - target position normalized (2) - goal if carrying, shelf otherwise
        - direction to target dx, dy normalized (2) - tells agent which way to go
        - agent facing direction one-hot (4) - UP/DOWN/LEFT/RIGHT
        - ht ∈ {0, 1}: Item held status (1)
        - bt ∈ [0, 100]: Battery level (1)
        Total: 12 extra dimensions
        """
        orig = self.env.observation_space
        extra_dims = 12  # pos(2) + target(2) + dir_to_target(2) + facing(4) + ht(1) + bt(1)
        
        if isinstance(orig, TupleSpace):
            new_spaces = []
            for sp in orig.spaces:
                lo = np.append(sp.low, [-1.0] * 2 + [0.0] * (extra_dims - 2))
                hi = np.append(sp.high, [1.0] * 6 + [1.0] * 4 + [1.0] + [100.0])
                new_spaces.append(Box(low=lo, high=hi, dtype=np.float32))
            self.observation_space = TupleSpace(new_spaces)
        else:
            lo = np.append(orig.low, [-1.0] * 2 + [0.0] * (extra_dims - 2))
            hi = np.append(orig.high, [1.0] * 6 + [1.0] * 4 + [1.0] + [100.0])
            self.observation_space = Box(low=lo, high=hi, dtype=np.float32)

    def _get_agent_direction_onehot(self, agent_idx):
        """Get agent facing direction as one-hot [UP, DOWN, LEFT, RIGHT]."""
        try:
            d = self.env.unwrapped.agents[agent_idx].dir.value
            onehot = [0.0, 0.0, 0.0, 0.0]
            if 0 <= d <= 3:
                onehot[d] = 1.0
            return onehot
        except Exception:
            return [1.0, 0.0, 0.0, 0.0]

    def _add_navigation_info_to_obs(self, obs):
        """
        Add global navigation info so agent knows where to go.
        
        Target priority:
        1. If battery < threshold: target = charger (need to recharge!)
        2. If carrying item (ht=1): target = goal (go to destination)
        3. If not carrying (ht=0): target = requested shelf (go to pickup)
        """
        grid_h, grid_w = self.env.unwrapped.grid_size
        
        # Get positions
        positions = self._get_agent_positions()
        goals = self._get_goal_positions()
        shelves = self._get_requested_shelf_positions()
        
        if isinstance(obs, tuple):
            new_obs = []
            for i, o in enumerate(obs):
                pos = positions[i] if i < len(positions) else (0, 0)
                carrying = 1.0 if self.env.unwrapped.agents[i].carrying_shelf else 0.0
                battery = self.battery_levels[i] if i < len(self.battery_levels) else self.max_battery
                
                # Find nearest goal
                if goals:
                    goal = min(goals, key=lambda g: abs(g[0]-pos[0]) + abs(g[1]-pos[1]))
                else:
                    goal = (grid_w//2, grid_h)  # default to center-bottom
                
                # Find nearest requested shelf
                if shelves:
                    shelf = min(shelves, key=lambda s: abs(s[0]-pos[0]) + abs(s[1]-pos[1]))
                else:
                    shelf = (grid_w//2, grid_h//2)  # default to center
                
                # Determine target based on priority:
                # 1. Battery critical -> go to charger
                # 2. Carrying -> go to goal (destination)
                # 3. Not carrying -> go to shelf (pickup point)
                if battery < self.battery_threshold:
                    target = self.charger_location  # Priority: recharge!
                elif carrying > 0.5:
                    target = goal  # Go to destination when carrying
                else:
                    target = shelf  # Go to pickup point
                
                # Normalize positions to [0, 1], add direction info
                # Direction to target (normalized delta)
                dx = (target[0] - pos[0]) / max(grid_w, 1)
                dy = (target[1] - pos[1]) / max(grid_h, 1)
                facing = self._get_agent_direction_onehot(i)
                
                extra = np.array([
                    pos[0] / max(grid_w, 1),      # agent x [0,1]
                    pos[1] / max(grid_h, 1),      # agent y [0,1]
                    target[0] / max(grid_w, 1),   # TARGET x (dynamic based on state)
                    target[1] / max(grid_h, 1),   # TARGET y
                    dx,                            # direction to target x [-1,1]
                    dy,                            # direction to target y [-1,1]
                    facing[0], facing[1], facing[2], facing[3],  # agent facing one-hot
                    float(carrying),               # ht ∈ {0, 1}: Item held status
                    battery,                       # bt ∈ [0, 100]: Battery level
                ], dtype=np.float32)
                
                new_obs.append(np.append(o.astype(np.float32), extra))
            return tuple(new_obs)
        else:
            pos = positions[0] if positions else (0, 0)
            carrying = 1.0 if self.env.unwrapped.agents[0].carrying_shelf else 0.0
            battery = self.battery_levels[0]
            goal = goals[0] if goals else (grid_w//2, grid_h)
            shelf = shelves[0] if shelves else (grid_w//2, grid_h//2)
            
            # Determine target based on priority
            if battery < self.battery_threshold:
                target = self.charger_location
            elif carrying > 0.5:
                target = goal
            else:
                target = shelf
            
            # Direction to target + facing direction
            dx = (target[0] - pos[0]) / max(grid_w, 1)
            dy = (target[1] - pos[1]) / max(grid_h, 1)
            facing = self._get_agent_direction_onehot(0)
            
            extra = np.array([
                pos[0] / max(grid_w, 1),
                pos[1] / max(grid_h, 1),
                target[0] / max(grid_w, 1),   # TARGET (dynamic)
                target[1] / max(grid_h, 1),
                dx,                            # direction to target x [-1,1]
                dy,                            # direction to target y [-1,1]
                facing[0], facing[1], facing[2], facing[3],  # agent facing one-hot
                float(carrying),               # ht ∈ {0, 1}: Item held status
                battery,                       # bt ∈ [0, 100]: Battery level
            ], dtype=np.float32)
            return np.append(obs.astype(np.float32), extra)

    def _add_battery_to_obs(self, obs):
        """Deprecated - use _add_navigation_info_to_obs instead."""
        return self._add_navigation_info_to_obs(obs)

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    def _get_agent_positions(self):
        try:
            return [(a.x, a.y) for a in self.env.unwrapped.agents]
        except Exception:
            return [(0, 0)] * self.n_agents

    def _is_at_charger(self, pos):
        return (abs(pos[0] - self.charger_location[0])
                + abs(pos[1] - self.charger_location[1])) <= 1.0

    def _get_requested_shelf_positions(self):
        """Get positions of shelves in the request queue."""
        try:
            return [(s.x, s.y) for s in self.env.unwrapped.request_queue]
        except Exception:
            return []

    def _get_goal_positions(self):
        """Get goal zone positions."""
        try:
            return list(self.env.unwrapped.goals)
        except Exception:
            return []

    def _min_distance_to_targets(self, pos, targets):
        """Manhattan distance to nearest target."""
        if not targets:
            return 0
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    @property
    def n_agents(self):
        return getattr(self.env.unwrapped, 'n_agents', 1)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.battery_levels = [self.max_battery] * self.n_agents
        self.deliveries_count = 0
        self.total_steps = 0
        self.mission_state = [SEEK_SHELF] * self.n_agents
        self.pickup_cooldown = [0] * self.n_agents  # ANTI-EXPLOIT: cooldown tracking
        self.steps_at_goal_carrying = [0] * self.n_agents  # Track goal camping
        self.steps_at_pickup = [0] * self.n_agents  # Track pickup point camping
        self.trip_start_step = [0] * self.n_agents  # Track when pickup happened
        self.pickup_rewarded = [False] * self.n_agents  # ONLY reward pickup ONCE!

        # Compute OPTIMAL route length at reset (agent → shelf → goal)
        self.optimal_steps = self._compute_optimal_steps()

        obs = self._add_battery_to_obs(obs)
        info['battery_levels'] = self.battery_levels.copy()
        info['deliveries'] = 0
        
        # Print starting mission
        print(f"  >>> Moving to PICKUP point... (optimal route: {self.optimal_steps} steps)")
        
        return obs, info

    def _compute_optimal_steps(self):
        """Compute minimum steps for a perfect run: agent→shelf→goal + 2 toggles."""
        positions = self._get_agent_positions()
        shelves = self._get_requested_shelf_positions()
        goals = self._get_goal_positions()
        if not positions or not shelves or not goals:
            return 20  # fallback
        agent_pos = positions[0]
        shelf_pos = min(shelves, key=lambda s: abs(s[0]-agent_pos[0]) + abs(s[1]-agent_pos[1]))
        dist_to_shelf = abs(agent_pos[0]-shelf_pos[0]) + abs(agent_pos[1]-shelf_pos[1])
        goal_pos = min(goals, key=lambda g: abs(g[0]-shelf_pos[0]) + abs(g[1]-shelf_pos[1]))
        dist_to_goal = abs(shelf_pos[0]-goal_pos[0]) + abs(shelf_pos[1]-goal_pos[1])
        # +2 for the two TOGGLE actions (pickup + drop)
        # +turns (rough estimate: ~2 turns needed on average)
        return dist_to_shelf + dist_to_goal + 2 + 2

    def step(self, action):
        # Snapshot state BEFORE the step
        was_carrying = [
            a.carrying_shelf is not None for a in self.env.unwrapped.agents
        ]
        old_positions = self._get_agent_positions()

        # Get targets for distance shaping
        shelf_targets = self._get_requested_shelf_positions()
        goal_targets = self._get_goal_positions()

        # Compute old distances for shaping
        old_distances = []
        for i, pos in enumerate(old_positions):
            if was_carrying[i]:
                old_distances.append(self._min_distance_to_targets(pos, goal_targets))
            else:
                old_distances.append(self._min_distance_to_targets(pos, shelf_targets))

        # --- base env step ---
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # Snapshot state AFTER the step
        now_carrying = [
            a.carrying_shelf is not None for a in self.env.unwrapped.agents
        ]
        positions = self._get_agent_positions()

        # ======== SIMPLIFIED REWARD STRUCTURE ========
        # 
        # Agent should ONLY care about:
        # 1. Going to REQUESTED shelf (pickup point in request_queue)
        # 2. Picking up that shelf
        # 3. Going to goal
        # 4. Dropping at goal
        #
        # Ignore all other shelves!
        
        # Step penalty GROWS over time: early steps cheap, late steps expensive
        # Optimal ~12 steps. Step 1=-1, step 12=-2, step 25=-5, step 50=-10
        step_penalty = -1.0 - (self.total_steps / self.optimal_steps)
        r = step_penalty
        
        # --- DISTANCE SHAPING + ARRIVAL BONUS + TOGGLE HINT ---
        for i, pos in enumerate(positions):
            # Determine current action
            if hasattr(action, '__iter__'):
                act = action[i]
            else:
                act = action

            if pos == old_positions[i] and act != 4:
                # Agent didn't move AND didn't toggle! Escalating penalty
                if now_carrying[i]:
                    self.steps_at_goal_carrying[i] += 1
                    r -= 15.0 * self.steps_at_goal_carrying[i]
                else:
                    self.steps_at_pickup[i] += 1
                    r -= 15.0 * self.steps_at_pickup[i]
            else:
                # Agent moved or toggled - reset camping counters
                self.steps_at_goal_carrying[i] = 0
                self.steps_at_pickup[i] = 0
                
                # Distance shaping - ONLY toward correct targets
                if now_carrying[i] and not was_carrying[i]:
                    pass  # Just picked up - skip distance check this step
                elif now_carrying[i]:
                    # Carrying: go to GOAL only
                    if goal_targets:
                        old_dist = self._min_distance_to_targets(old_positions[i], goal_targets)
                        new_dist = self._min_distance_to_targets(pos, goal_targets)
                        if new_dist < old_dist:
                            r += 8.0  # Bonus for getting closer to goal
                        elif new_dist > old_dist:
                            r -= 5.0  # Penalty for moving away from goal
                elif not now_carrying[i]:
                    # Not carrying: go to REQUESTED SHELF only
                    if shelf_targets:
                        old_dist = self._min_distance_to_targets(old_positions[i], shelf_targets)
                        new_dist = self._min_distance_to_targets(pos, shelf_targets)
                        if new_dist < old_dist:
                            r += 8.0  # Bonus for getting closer to pickup point
                        elif new_dist > old_dist:
                            r -= 5.0  # Penalty for moving away from pickup point

            # --- ARRIVAL BONUS: reward reaching the right spot ---
            if not was_carrying[i] and pos in shelf_targets and old_positions[i] not in shelf_targets:
                r += 20.0  # "You're here! Now toggle!"

            if now_carrying[i] and pos in goal_targets and old_positions[i] not in goal_targets:
                r += 20.0  # "You're at goal! Now drop!"

            # --- AT TARGET BUT NOT TOGGLING: penalize hesitation ---
            if act != 4:
                if not was_carrying[i] and pos in shelf_targets:
                    r -= 10.0  # You're AT the shelf, just toggle!
                elif was_carrying[i] and pos in goal_targets:
                    r -= 10.0  # You're AT the goal, just drop!

            # --- TOGGLE HINT: reward toggling at the right place ---
            if act == 4:
                if not was_carrying[i] and pos in shelf_targets:
                    r += 15.0  # Good: toggling at correct shelf
                elif was_carrying[i] and pos in goal_targets:
                    r += 15.0  # Good: toggling to drop at goal

        # --- battery / charger (simple - no camping rewards) ---
        for i in range(self.n_agents):
            if self._is_at_charger(positions[i]):
                self.battery_levels[i] = min(
                    self.max_battery,
                    self.battery_levels[i] + self.charge_rate)
            else:
                self.battery_levels[i] = max(
                    0, self.battery_levels[i] - self.battery_drain)

        # --- pickup: BIG reward only for ACTUAL successful pickup (ONCE per delivery!) ---
        for i in range(self.n_agents):
            # Decrease cooldown timer
            if self.pickup_cooldown[i] > 0:
                self.pickup_cooldown[i] -= 1

            # Check for actual pickup (state change: not carrying -> carrying)
            if (self.mission_state[i] == SEEK_SHELF
                    and not was_carrying[i]
                    and now_carrying[i]
                    and self.pickup_cooldown[i] <= 0
                    and not self.pickup_rewarded[i]):  # Only reward ONCE per delivery cycle!
                
                pos = positions[i]
                old_pos = old_positions[i]
                
                # Check if agent picked up a REQUESTED shelf
                picked_requested = old_pos in shelf_targets
                
                if picked_requested:
                    r += 150.0  # BIG reward for successful pickup at correct location!
                    self.trip_start_step[i] = self.total_steps
                    self.steps_at_pickup[i] = 0
                    self.pickup_rewarded[i] = True  # Mark as rewarded - no more pickup bonus!
                    if goal_targets:
                        nearest_goal = min(goal_targets, key=lambda g: abs(g[0]-pos[0]) + abs(g[1]-pos[1]))
                        dist_to_goal = abs(nearest_goal[0]-pos[0]) + abs(nearest_goal[1]-pos[1])
                        print(f"  [PICKUP +150] Step {self.total_steps}, go to goal (dist: {dist_to_goal})")
                    self.mission_state[i] = DELIVER
                # Wrong shelf pickup ignored - agent just doesn't get reward

            # Penalty for dropping at wrong place
            if (self.mission_state[i] == DELIVER
                    and was_carrying[i]
                    and not now_carrying[i]):
                env_r = sum(reward) if isinstance(reward, (list, tuple)) else reward
                if env_r <= 0:
                    r -= 50.0  # INCREASED penalty for dropping at wrong place
                    self.mission_state[i] = SEEK_SHELF
                    self.pickup_cooldown[i] = 5  # Longer cooldown
                    # NOTE: pickup_rewarded stays True - NO more pickup bonuses this episode!

        # --- delivery: BIG reward — scales with how close to OPTIMAL route ---
        env_r = sum(reward) if isinstance(reward, (list, tuple)) else reward
        if env_r > 0:
            self.deliveries_count += 1
            
            # How efficient was the agent?
            optimal = self.optimal_steps
            actual = self.total_steps
            efficiency = optimal / max(actual, 1)  # 1.0 = perfect, lower = worse
            
            # Scale reward: perfect route = 500, double optimal = 250, worse = less
            delivery_reward = 500.0 * min(efficiency, 1.0)  # Cap at 500
            r += delivery_reward
            
            print(f"  [DELIVERY +{delivery_reward:.0f}] Steps: {actual} (optimal: {optimal}, efficiency: {efficiency:.1%})")

            # Reset state for next delivery (if max_deliveries > 1)
            for i in range(self.n_agents):
                if not now_carrying[i]:
                    self.mission_state[i] = SEEK_SHELF
                    self.trip_start_step[i] = self.total_steps
                    self.pickup_rewarded[i] = False

        # ======== TERMINATION ========
        battery_dead = any(b <= 0 for b in self.battery_levels)
        mission_complete = self.deliveries_count >= self.max_deliveries

        if battery_dead:
            terminated = True
            r -= 100.0
            print(f"  [BATTERY DEAD] Step {self.total_steps}")

        if mission_complete:
            terminated = True
            # Bonus scales with efficiency — perfect route gets max bonus
            optimal = self.optimal_steps
            actual = self.total_steps
            efficiency = optimal / max(actual, 1)
            completion_bonus = 300.0 * min(efficiency, 1.0)
            r += completion_bonus
            print(f"  [WIN +{completion_bonus:.0f}] Done in {actual} steps (optimal: {optimal}, eff: {efficiency:.1%})")

        # ======== PACKAGE OUTPUT ========
        obs = self._add_battery_to_obs(obs)
        info.update({
            'battery_levels': self.battery_levels.copy(),
            'deliveries': self.deliveries_count,
            'mission_complete': mission_complete,
            'battery_dead': battery_dead,
        })
        return obs, float(r), terminated, truncated, info


def make_battery_warehouse(
    env_id=None,
    shelf_columns=3,  # ODD number required by RWARE 
    column_height=1,  # MINIMAL: 1 shelf per column = 6 total shelves
    shelf_rows=1,     # Keep simple with 1 row
    n_agents=1,
    max_steps=50,
    **battery_kwargs
):
    """
    Create RWARE environment with battery wrapper.
    
    Args:
        env_id: Ignored - uses direct Warehouse creation (more reliable)
        shelf_columns: Number of shelf columns (MUST be ODD: 1, 3, 5, 7...)
        column_height: Height of each shelf column
        shelf_rows: Number of shelf rows
        n_agents: Number of agents
        max_steps: Max steps per episode
        **battery_kwargs: Passed to BatteryWrapper
    """
    from rware.warehouse import Warehouse, RewardType
    env = Warehouse(
        shelf_columns=shelf_columns,
        column_height=column_height,
        shelf_rows=shelf_rows,
        n_agents=n_agents,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,  # ONLY ONE pickup target at a time!
        max_inactivity_steps=None,
        max_steps=max_steps,
        reward_type=RewardType.INDIVIDUAL,
    )
    return BatteryWrapper(env, **battery_kwargs)
