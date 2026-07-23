import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor('white')

# ── Grid ──────────────────────────────────────────────────────────────────────
for x in np.arange(-75, 76, 25):
    ax.axvline(x, color='#cccccc', linewidth=0.5, zorder=0)
for y in np.arange(-75, 76, 25):
    ax.axhline(y, color='#cccccc', linewidth=0.5, zorder=0)

# ── Ocean Current Field ────────────────────────────────────────────────────────
# Constant current pointing top-right: (0.6, 0.4) m/s
curr_u, curr_v = 0.6, 0.4
xs = np.arange(-65, 70, 20)
ys = np.arange(-65, 70, 20)
XX, YY = np.meshgrid(xs, ys)
ax.quiver(XX, YY, curr_u, curr_v,
          color='#2196F3', alpha=0.3, scale=18,
          width=0.003, headwidth=4, label='Ocean Current Field', zorder=1)

# ── Static Obstacles ──────────────────────────────────────────────────────────
static_obs = [
    (-30, 20, 8),
    (15, -25, 10),
    (40, 30, 7),
    (-50, -10, 9),
    (10, 40, 6),
    (-10, -45, 8),
    (50, -40, 7),
]
for (ox, oy, r) in static_obs:
    c = plt.Circle((ox, oy), r, color='#607d8b', alpha=0.9, zorder=3)
    ax.add_patch(c)
ax.plot([], [], color='#607d8b', marker='o', linestyle='', markersize=10, label='Static Obstacles')

# ── Dynamic Obstacles ─────────────────────────────────────────────────────────
dyn_obs = [
    (-20, 10, 5, 1.2, -0.8),   # pos_x, pos_y, radius, vel_x, vel_y
    (30, -15, 5, -1.0,  1.0),
    (0,   35, 4,  0.5, -1.5),
]
for i, (dx, dy, dr, dvx, dvy) in enumerate(dyn_obs):
    c = plt.Circle((dx, dy), dr, color='#ef5350', alpha=0.9, zorder=4)
    ax.add_patch(c)
    ax.quiver(dx, dy, dvx, dvy, color='#ff8a80', scale=8,
              width=0.007, headwidth=5, zorder=5)
ax.plot([], [], color='#ef5350', marker='o', linestyle='', markersize=10, label='Dynamic Obstacles (with velocity)')

# ── Spawn Zone ────────────────────────────────────────────────────────────────
spawn_circle = plt.Circle((-55, -55), 12, color='#1565c0',
                           fill=False, linestyle='--', linewidth=2, zorder=6)
ax.add_patch(spawn_circle)
ax.text(-55, -55, 'Spawn\nZone', color='#42a5f5', ha='center', va='center',
        fontsize=8, fontweight='bold', zorder=7)

# ── Goal Zone ─────────────────────────────────────────────────────────────────
goal_circle = plt.Circle((55, 55), 10, color='#2e7d32',
                          fill=False, linestyle='--', linewidth=2, zorder=6)
ax.add_patch(goal_circle)
ax.text(55, 55, 'Goal\nZone', color='#66bb6a', ha='center', va='center',
        fontsize=8, fontweight='bold', zorder=7)

# ── AUV Agents ────────────────────────────────────────────────────────────────
agents = [
    (-60, -50, np.pi/5),
    (-55, -58, np.pi/4),
    (-50, -52, np.pi/6),
]
agent_colors = ['#e53935', '#8e24aa', '#00897b']  # Red, Purple, Teal
for i, (ax_, ay_, yaw) in enumerate(agents):
    triangle = patches.RegularPolygon((ax_, ay_), numVertices=3,
                                       radius=4, orientation=yaw,
                                       color=agent_colors[i], zorder=8)
    ax.add_patch(triangle)
    ax.plot([], [], color=agent_colors[i], marker='^', linestyle='',
            markersize=10, label=f'AUV Agent {i+1}')

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlim(-75, 75)
ax.set_ylim(-75, 75)
ax.set_aspect('equal')
ax.set_xlabel('X (meters)', color='black', fontsize=11)
ax.set_ylabel('Y (meters)', color='black', fontsize=11)
ax.set_title('150m × 150m USV Simulation Environment Preview',
             color='black', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors='black')
for spine in ax.spines.values():
    spine.set_edgecolor('#aaaaaa')

legend = ax.legend(loc='lower right', facecolor='white',
                   edgecolor='#aaaaaa', labelcolor='black', fontsize=9)

plt.tight_layout()
out = 'visualizations_contributions/small_env_preview.png'
plt.savefig(out, dpi=200, facecolor='white')
print(f'Saved to {out}')
