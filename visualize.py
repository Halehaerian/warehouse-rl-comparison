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
from configs.config import ENV_CONFIG, BATTERY_CONFIG, ALGO_CONFIGS

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
    
    def __init__(self, env, charger_location=(0, 0)):
        self.env = env.unwrapped
        self.charger_location = charger_location
        
        self.rows, self.cols = self.env.grid_size
        self.grid_size = 30
        self.icon_size = 20
        
        # Window dimensions with extra space for battery bar
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 2 + self.rows * (self.grid_size + 1) + 60
        
        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            caption="RWARE + Battery Visualization"
        )
        self.window.on_close = self._on_close
        
        # State
        self.battery_level = 100.0
        self.step_count = 0
        self.deliveries = 0
        self.pickups = 0
        self.is_carrying = False
        self.closed = False
        
        # Colors
        self._BACKGROUND = (255, 255, 255)
        self._GRID = (200, 200, 200)
        self._SHELF = (72, 61, 139)
        self._SHELF_REQ = (0, 128, 128)
        self._AGENT = (255, 140, 0)
        self._AGENT_LOADED = (255, 0, 0)
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
            y = r * (self.grid_size + 1) + 60
            glVertex2f(0, y)
            glVertex2f(self.width, y)
        for c in range(self.cols + 1):
            x = c * (self.grid_size + 1)
            glVertex2f(x, 60)
            glVertex2f(x, self.height)
        glEnd()
    
    def _draw_rect(self, x, y, color, padding=2):
        px = x * (self.grid_size + 1) + padding
        py = y * (self.grid_size + 1) + padding + 60
        size = self.grid_size - 2 * padding
        
        glColor3ub(*color)
        glBegin(GL_QUADS)
        glVertex2f(px, py)
        glVertex2f(px + size, py)
        glVertex2f(px + size, py + size)
        glVertex2f(px, py + size)
        glEnd()
    
    def _draw_charger(self):
        x, y = self.charger_location
        px = x * (self.grid_size + 1)
        py = y * (self.grid_size + 1) + 60
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
    
    def _draw_shelves(self):
        requested = set()
        if hasattr(self.env, 'request_queue'):
            requested = {(s.x, s.y) for s in self.env.request_queue}
        
        for shelf in self.env.shelfs:
            if (shelf.x, shelf.y) in requested:
                self._draw_rect(shelf.x, shelf.y, self._SHELF_REQ, padding=3)
            else:
                self._draw_rect(shelf.x, shelf.y, self._SHELF, padding=4)
    
    def _draw_goals(self):
        for gx, gy in self.env.goals:
            self._draw_rect(gx, gy, self._GOAL, padding=1)
    
    def _draw_agents(self):
        for agent in self.env.agents:
            color = self._AGENT_LOADED if agent.carrying_shelf else self._AGENT
            
            px = agent.x * (self.grid_size + 1)
            py = agent.y * (self.grid_size + 1) + 60
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
    
    def _draw_battery_bar(self):
        bar_x = 10
        bar_y = 15
        bar_width = self.width - 20
        bar_height = 25
        
        # Background
        glColor3ub(*self._BATTERY_BG)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Battery fill
        fill_width = (self.battery_level / 100.0) * (bar_width - 4)
        if self.battery_level > 50:
            color = self._BATTERY_GOOD
        elif self.battery_level > 20:
            color = self._BATTERY_MED
        else:
            color = self._BATTERY_LOW
        
        glColor3ub(*color)
        glBegin(GL_QUADS)
        glVertex2f(bar_x + 2, bar_y + 2)
        glVertex2f(bar_x + 2 + fill_width, bar_y + 2)
        glVertex2f(bar_x + 2 + fill_width, bar_y + bar_height - 2)
        glVertex2f(bar_x + 2, bar_y + bar_height - 2)
        glEnd()
        
        # Labels
        label = pyglet.text.Label(
            f"Battery: {self.battery_level:.1f}%  |  Step: {self.step_count}  |  "
            f"Pickups: {self.pickups}  |  Deliveries: {self.deliveries}",
            font_name='Arial', font_size=10,
            x=self.width // 2, y=bar_y + bar_height // 2,
            anchor_x='center', anchor_y='center',
            color=(255, 255, 255, 255)
        )
        label.draw()
        
        status = "CARRYING SHELF →" if self.is_carrying else "Seeking shelf..."
        status_color = (255, 100, 100, 255) if self.is_carrying else (150, 150, 150, 255)
        status_label = pyglet.text.Label(
            status, font_name='Arial', font_size=9,
            x=self.width // 2, y=bar_y + bar_height + 8,
            anchor_x='center', anchor_y='center',
            color=status_color
        )
        status_label.draw()
    
    def render(self, battery_level, step_count, deliveries, pickups, is_carrying):
        if self.closed:
            return False
        
        self.battery_level = battery_level
        self.step_count = step_count
        self.deliveries = deliveries
        self.pickups = pickups
        self.is_carrying = is_carrying
        
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        
        self._draw_grid()
        self._draw_charger()
        self._draw_goals()
        self._draw_shelves()
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

    print(f"{'='*60}")
    print(f"Visualizing {algo.upper()} | Model: {model_path}")
    print(f"{'='*60}")

    # Create environment
    env = make_env(ENV_CONFIG, BATTERY_CONFIG)
    charger = tuple(BATTERY_CONFIG.get("charger_location", (0, 0)))
    renderer = BatteryRenderer(env, charger_location=charger)

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
        state = np.array(obs[0]) if isinstance(obs, tuple) else np.array(obs)
        steps = 0
        pickups = 0
        deliveries = 0
        print(f"Episode {episode + 1}/{args.episodes}")

        while True:
            action = agent.select_action(state, training=False)
            carrying_before = env.unwrapped.agents[0].carrying_shelf is not None

            if hasattr(env.action_space, "spaces"):
                actions = tuple([action] + [0] * (len(env.action_space.spaces) - 1))
            else:
                actions = action

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            state = np.array(next_obs[0]) if isinstance(next_obs, tuple) else np.array(next_obs)
            steps += 1

            carrying_after = env.unwrapped.agents[0].carrying_shelf is not None
            if not carrying_before and carrying_after:
                pickups += 1
                total_pickups += 1

            new_del = info.get("deliveries", 0)
            if new_del > deliveries:
                total_deliveries += new_del - deliveries
                deliveries = new_del

            battery = info.get("battery_levels", [100])[0]

            if not renderer.render(battery, steps, deliveries, pickups, carrying_after):
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
