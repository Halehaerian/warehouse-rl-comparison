"""
Visualize trained agent in RWARE environment.

Shows the agent in a graphical window with battery level bar,
charger station, and real-time stats.

Usage:
    python visualize.py --algo dqn
    python visualize.py --algo ppo --model models/ppo_best.pt
"""

import argparse
import numpy as np
import torch
import time
from pathlib import Path
import gymnasium as gym
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

from envs.warehouse import make_env
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
from configs.config import ENV_CONFIG, ENV_PRESETS, BATTERY_CONFIG, ALGO_CONFIGS

# Import pyglet for graphical rendering
try:
    import pyglet
    from pyglet.gl import *
    PYGLET_AVAILABLE = True
except ImportError:
    PYGLET_AVAILABLE = False
    print("WARNING: pyglet not available. Install with: pip install pyglet")


class BatteryRenderer:
    """
    Custom RWARE renderer with battery visualization.
    
    Shows:
    - Battery bar at top with percentage
    - Yellow charger station with lightning bolt
    - Agent (orange/red based on carrying status)
    - Shelves (dark blue = regular, teal = requested)
    - Goals (dark gray)
    """
    
    def __init__(self, env, charger_locations=None):
        self.env = env.unwrapped
        self.charger_locations = charger_locations or [(0, 0)]
        
        self.rows, self.cols = self.env.grid_size
        self.grid_size = 30
        self.icon_size = 20
        self.grid_offset = 80  # pixels at bottom for per-agent battery bars
        
        # Window dimensions
        self.width = max(300, 1 + self.cols * (self.grid_size + 1))
        self.height = 2 + self.rows * (self.grid_size + 1) + self.grid_offset
        
        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            caption="RWARE + Battery Visualization"
        )
        self.window.on_close = self._on_close
        
        # State
        self.battery_levels = [100.0, 100.0]
        self.mission_states = [0, 0]
        self.step_count = 0
        self.deliveries = 0
        self.pickups = 0
        self.closed = False
        
        # Colors
        self._BACKGROUND = (255, 255, 255)
        self._GRID = (200, 200, 200)
        self._SHELF = (72, 61, 139)
        self._SHELF_REQ = (0, 128, 128)
        # Per-agent: agent 0 = orange, agent 1 = blue
        self._AGENT_COLORS = [(255, 140, 0), (30, 120, 255)]
        self._AGENT_LOADED_COLORS = [(220, 50, 50), (30, 50, 220)]
        self._GOAL = (60, 60, 60)
        self._CHARGER = (255, 215, 0)
        self._BATTERY_BG = (100, 100, 100)
        self._BATTERY_GOOD = (0, 200, 0)
        self._BATTERY_MED = (255, 200, 0)
        self._BATTERY_LOW = (255, 0, 0)
    
    def _on_close(self):
        self.closed = True
    
    def _draw_grid(self):
        glColor3ub(*self._GRID)
        glBegin(GL_LINES)
        for r in range(self.rows + 1):
            y = r * (self.grid_size + 1) + self.grid_offset
            glVertex2f(0, y)
            glVertex2f(self.width, y)
        for c in range(self.cols + 1):
            x = c * (self.grid_size + 1)
            glVertex2f(x, self.grid_offset)
            glVertex2f(x, self.height)
        glEnd()
    
    def _draw_rect(self, x, y, color, padding=2):
        px = x * (self.grid_size + 1) + padding
        py = y * (self.grid_size + 1) + padding + self.grid_offset
        size = self.grid_size - 2 * padding
        
        glColor3ub(*color)
        glBegin(GL_QUADS)
        glVertex2f(px, py)
        glVertex2f(px + size, py)
        glVertex2f(px + size, py + size)
        glVertex2f(px, py + size)
        glEnd()
    
    def _draw_charger(self):
        for idx, (x, y) in enumerate(self.charger_locations):
            px = x * (self.grid_size + 1)
            py = y * (self.grid_size + 1) + self.grid_offset
            size = self.grid_size

            # Yellow background
            glColor3ub(*self._CHARGER)
            glBegin(GL_QUADS)
            glVertex2f(px + 2, py + 2)
            glVertex2f(px + size - 2, py + 2)
            glVertex2f(px + size - 2, py + size - 2)
            glVertex2f(px + 2, py + size - 2)
            glEnd()

            # Lightning bolt
            glColor3ub(0, 0, 0)
            glLineWidth(2.0)
            cx = px + size // 2
            cy = py + size // 2
            glBegin(GL_LINE_STRIP)
            glVertex2f(cx + 2, cy + 10)
            glVertex2f(cx - 4, cy + 2)
            glVertex2f(cx + 1, cy + 2)
            glVertex2f(cx - 2, cy - 10)
            glVertex2f(cx + 4, cy - 2)
            glVertex2f(cx - 1, cy - 2)
            glVertex2f(cx + 2, cy + 10)
            glEnd()
            glLineWidth(1.0)
    
    def _draw_border(self, x, y, color, padding=1):
        """Draw a colored outline around a grid cell (for claimed shelf highlight)."""
        px = x * (self.grid_size + 1) + padding
        py = y * (self.grid_size + 1) + padding + self.grid_offset
        size = self.grid_size - 2 * padding
        glColor3ub(*color)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(px, py)
        glVertex2f(px + size, py)
        glVertex2f(px + size, py + size)
        glVertex2f(px, py + size)
        glEnd()
        glLineWidth(1.0)

    def _draw_shelves(self, shelf_claims=None):
        requested = set()
        if hasattr(self.env, 'request_queue'):
            requested = {(s.x, s.y) for s in self.env.request_queue}

        # Map claimed position -> agent index for border highlight
        claim_map = {}
        if shelf_claims:
            for i, claim in enumerate(shelf_claims):
                if claim is not None:
                    claim_map[claim] = i

        for shelf in self.env.shelfs:
            pos = (shelf.x, shelf.y)
            if pos in requested:
                self._draw_rect(shelf.x, shelf.y, self._SHELF_REQ, padding=3)
            else:
                self._draw_rect(shelf.x, shelf.y, self._SHELF, padding=4)
            # Colored border = this shelf is claimed by that agent
            if pos in claim_map:
                agent_i = claim_map[pos]
                border_color = self._AGENT_COLORS[agent_i % len(self._AGENT_COLORS)]
                self._draw_border(shelf.x, shelf.y, border_color)
    
    def _draw_goals(self):
        for gx, gy in self.env.goals:
            self._draw_rect(gx, gy, self._GOAL, padding=1)
    
    def _draw_agents(self):
        for idx, agent in enumerate(self.env.agents):
            colors = self._AGENT_LOADED_COLORS if agent.carrying_shelf else self._AGENT_COLORS
            color = colors[idx % len(colors)]
            
            px = agent.x * (self.grid_size + 1)
            py = agent.y * (self.grid_size + 1) + self.grid_offset
            size = self.grid_size
            
            # Draw agent as circle
            glColor3ub(*color)
            cx = px + size // 2
            cy = py + size // 2
            radius = size // 2 - 3
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(21):
                angle = 2 * 3.14159 * i / 20
                glVertex2f(cx + radius * np.cos(angle), cy + radius * np.sin(angle))
            glEnd()
            
            # Direction indicator
            glColor3ub(0, 0, 0)
            glLineWidth(2.0)
            dir_offsets = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
            dx, dy = dir_offsets.get(agent.dir.value, (0, 0))
            glBegin(GL_LINES)
            glVertex2f(cx, cy)
            glVertex2f(cx + dx * radius * 0.8, cy + dy * radius * 0.8)
            glEnd()
            glLineWidth(1.0)

            # Agent number label
            pyglet.text.Label(
                str(idx), font_name='Arial', font_size=8, bold=True,
                x=cx, y=cy, anchor_x='center', anchor_y='center',
                color=(255, 255, 255, 255)
            ).draw()
    
    def _draw_battery_bar(self):
        n = max(len(self.battery_levels), 1)
        bar_height = 22
        total_w = self.width - 20
        bar_width = (total_w - (n - 1) * 5) // n

        for i in range(n):
            bar_x = 10 + i * (bar_width + 5)
            bar_y = 10
            bl = self.battery_levels[i] if i < len(self.battery_levels) else 100.0

            # Background
            glColor3ub(*self._BATTERY_BG)
            glBegin(GL_QUADS)
            glVertex2f(bar_x, bar_y)
            glVertex2f(bar_x + bar_width, bar_y)
            glVertex2f(bar_x + bar_width, bar_y + bar_height)
            glVertex2f(bar_x, bar_y + bar_height)
            glEnd()

            # Battery fill
            fill_width = (bl / 100.0) * (bar_width - 4)
            color = self._BATTERY_GOOD if bl > 50 else (self._BATTERY_MED if bl > 20 else self._BATTERY_LOW)
            glColor3ub(*color)
            glBegin(GL_QUADS)
            glVertex2f(bar_x + 2, bar_y + 2)
            glVertex2f(bar_x + 2 + fill_width, bar_y + 2)
            glVertex2f(bar_x + 2 + fill_width, bar_y + bar_height - 2)
            glVertex2f(bar_x + 2, bar_y + bar_height - 2)
            glEnd()

            # Label inside bar
            ms = self.mission_states[i] if i < len(self.mission_states) else 0
            state_str = "CHARGING" if ms == 2 else ("CARRYING" if ms == 1 else "SEEKING")
            pyglet.text.Label(
                f"A{i}: {bl:.0f}%  {state_str}",
                font_name='Arial', font_size=9,
                x=bar_x + bar_width // 2, y=bar_y + bar_height // 2,
                anchor_x='center', anchor_y='center',
                color=(255, 255, 255, 255)
            ).draw()

        # Step / delivery summary
        pyglet.text.Label(
            f"Step: {self.step_count}  |  Deliveries: {self.deliveries}  |  Pickups: {self.pickups}",
            font_name='Arial', font_size=9,
            x=self.width // 2, y=42,
            anchor_x='center', anchor_y='center',
            color=(30, 30, 30, 255)
        ).draw()
    
    def render(self, battery_levels, step_count, deliveries, pickups, mission_states, shelf_claims=None):
        if self.closed:
            return False
        
        self.battery_levels = list(battery_levels)
        self.mission_states = list(mission_states)
        self.step_count = step_count
        self.deliveries = deliveries
        self.pickups = pickups
        
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        
        self._draw_grid()
        self._draw_charger()
        self._draw_goals()
        self._draw_shelves(shelf_claims)
        self._draw_agents()
        self._draw_battery_bar()
        
        self.window.flip()
        self.window.dispatch_events()
        
        return not self.closed
    
    def close(self):
        if not self.closed:
            self.window.close()
            self.closed = True


def find_latest_model(algo="dqn"):
    """Find the most recent model file for the given algorithm."""
    models_dir = Path("models")
    models = list(models_dir.glob(f"{algo}*.pt"))
    if not models:
        # Fallback: any .pt file
        models = list(models_dir.glob("*.pt"))
    if not models:
        return None
    best = [m for m in models if "best" in m.name]
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return max(models, key=lambda p: p.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained agent")
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ppo", "sac"],
                        help="Algorithm to visualize (default: dqn)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: latest)")
    parser.add_argument("--env", type=str, default="default",
                        choices=list(ENV_PRESETS.keys()),
                        help="Warehouse map preset (default: default)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.08,
                        help="Delay between frames (seconds)")
    return parser.parse_args()


AGENT_CLASSES = {"dqn": DQNAgent, "ppo": PPOAgent, "sac": SACAgent}


def visualize(args):
    """Run visualization."""
    if not PYGLET_AVAILABLE:
        print("ERROR: pyglet is required. Install with: pip install pyglet")
        return

    algo = args.algo
    model_path = args.model or find_latest_model(algo)
    if model_path is None:
        print(f"No trained model found for {algo}. Run train.py --algo {algo} first!")
        return

    env_config = ENV_PRESETS[args.env]
    battery_config = BATTERY_CONFIG
    print(f"{'='*60}")
    print(f"Visualizing {algo.upper()} | Model: {model_path}")
    print(f"{'='*60}")

    # Create environment
    env = make_env(env_config, battery_config)
    # Trigger reset to auto-compute charger positions
    env.reset()
    charger_locs = getattr(env, "charger_locations", [(0, 0)])
    renderer = BatteryRenderer(env, charger_locations=charger_locs)

    # Create and load agent
    device = torch.device("cpu")
    obs, _ = env.reset()
    state = np.array(obs[0]) if isinstance(obs, tuple) else np.array(obs)
    obs_size = state.shape[0]
    n_actions = env.action_space.spaces[0].n if hasattr(env.action_space, "spaces") else env.action_space.n

    agent = AGENT_CLASSES[algo](obs_size, n_actions, device, ALGO_CONFIGS[algo])
    agent.load(str(model_path))
    print("Model loaded\n")

    total_deliveries = 0
    total_pickups = 0

    for episode in range(args.episodes):
        obs, info = env.reset()
        if isinstance(obs, tuple):
            states = [np.array(o) for o in obs]
        else:
            states = [np.array(obs)]
        steps = 0
        pickups = 0
        deliveries = 0
        print(f"Episode {episode + 1}/{args.episodes}")

        while True:
            # Get action for each agent using the shared policy
            actions = [agent.select_action(s, training=False) for s in states]
            carrying_before = [a.carrying_shelf is not None for a in env.unwrapped.agents]

            if hasattr(env.action_space, "spaces"):
                n = len(env.action_space.spaces)
                wrapped_actions = tuple(actions[:n] + [0] * max(0, n - len(actions)))
            else:
                wrapped_actions = actions[0]

            next_obs, reward, terminated, truncated, info = env.step(wrapped_actions)
            done = terminated or truncated
            if isinstance(next_obs, tuple):
                states = [np.array(o) for o in next_obs]
            else:
                states = [np.array(next_obs)]
            steps += 1

            carrying_after = [a.carrying_shelf is not None for a in env.unwrapped.agents]
            for before, after in zip(carrying_before, carrying_after):
                if not before and after:
                    pickups += 1
                    total_pickups += 1

            new_del = info.get("deliveries", 0)
            if new_del > deliveries:
                total_deliveries += new_del - deliveries
                deliveries = new_del

            # Per-agent data for renderer
            battery_levels = info.get("battery_levels", [100, 100])
            mission_states = getattr(env, "mission_state", [0] * len(battery_levels))
            shelf_claims = getattr(env, "agent_shelf_claim", None)

            if not renderer.render(battery_levels, steps, deliveries, pickups, mission_states, shelf_claims):
                renderer.close()
                env.close()
                return

            time.sleep(args.delay)
            if done:
                break

        mc = info.get("mission_complete", False)
        bd = info.get("battery_dead", False)
        stuck = info.get("agent_stuck", False)
        status = "SUCCESS" if mc else ("BATTERY DEAD" if bd else ("STUCK" if stuck else "TIMEOUT"))
        print(f"  {status} | Steps: {steps} | Pickups: {pickups} | Deliveries: {deliveries}")

    print(f"\nTotal: {total_pickups} pickups, {total_deliveries} deliveries")
    time.sleep(3)
    try:
        renderer.close()
    except Exception:
        pass
    env.close()


if __name__ == "__main__":
    args = parse_args()
    visualize(args)
