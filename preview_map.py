import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Map dimensions
map_size = 2000
half_size = map_size / 2

# Zones
spawn_center = np.array([-800, -800])
spawn_radius = 150
goal_center = np.array([800, 800])
goal_radius = 150

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-half_size, half_size)
ax.set_ylim(-half_size, half_size)
ax.set_facecolor('white')
ax.grid(True, color='#e0e0e0', alpha=0.8)
ax.set_title(f"2km x 2km Uniform Map Layout Preview", color='black', fontsize=16)

placed_obstacles = [] # list of (x, y, r)

def is_valid_position(x, y, r):
    if abs(x) + r > half_size or abs(y) + r > half_size:
        return False
    if np.linalg.norm(np.array([x, y]) - spawn_center) < (spawn_radius + r + 20.0):
        return False
    if np.linalg.norm(np.array([x, y]) - goal_center) < (goal_radius + r + 20.0):
        return False
    for ox, oy, orad in placed_obstacles:
        if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (orad + r + 15.0): # 15m padding
            return False
    return True

np.random.seed(111)

# 1. Stratified Uniform Sampling for Static Obstacles
num_static = 45 # Decreased from 60
static_obstacles = []

# Create a 7x7 grid across the 2000x2000 map (49 total cells)
grid_points = []
step = map_size / 7
for i in range(7):
    for j in range(7):
        base_x = -half_size + step/2 + i*step
        base_y = -half_size + step/2 + j*step
        grid_points.append((base_x, base_y))

np.random.shuffle(grid_points)

for base_x, base_y in grid_points:
    if len(static_obstacles) >= num_static:
        break
        
    attempts = 0
    while attempts < 100:
        attempts += 1
        r = np.random.uniform(20.0, 70.0)
        # Add random jitter within the cell
        jitter_x = np.random.uniform(-step/3, step/3)
        jitter_y = np.random.uniform(-step/3, step/3)
        x = base_x + jitter_x
        y = base_y + jitter_y
        
        if is_valid_position(x, y, r):
            static_obstacles.append((x, y, r))
            placed_obstacles.append((x, y, r))
            break

for x, y, r in static_obstacles:
    circ = patches.Circle((x, y), r, color='#555555', alpha=0.6)
    ax.add_patch(circ)
ax.scatter([], [], color='#555555', s=200, alpha=0.6, label=f'Static Obstacles ({len(static_obstacles)})')

# 2. Dynamic Obstacles
num_dynamic = 8
dynamic_obstacles = []
attempts = 0
while len(dynamic_obstacles) < num_dynamic and attempts < 10000:
    attempts += 1
    r = np.random.uniform(15.0, 40.0)
    x = np.random.uniform(-half_size + r, half_size - r)
    y = np.random.uniform(-half_size + r, half_size - r)
    if is_valid_position(x, y, r):
        dynamic_obstacles.append((x, y, r))
        placed_obstacles.append((x, y, r))

for x, y, r in dynamic_obstacles:
    vx = np.random.uniform(-15.0, 15.0)
    vy = np.random.uniform(-15.0, 15.0)
    circ = patches.Circle((x, y), r, color='red', alpha=0.5)
    ax.add_patch(circ)
    ax.arrow(x, y, vx*4, vy*4, head_width=20, head_length=20, fc='red', ec='red')

ax.scatter([], [], color='red', s=200, alpha=0.5, label='Dynamic Obstacles (with Velocity)')

# 3. RL Swarm Spawn & Goal Zones
spawn_circ = patches.Circle(spawn_center, spawn_radius, fill=False, color='blue', linestyle='--', linewidth=2, label='RL Swarm Spawn Zone')
goal_circ = patches.Circle(goal_center, goal_radius, fill=False, color='green', linestyle='--', linewidth=2, label='RL Swarm Goal Zone')
ax.add_patch(spawn_circ)
ax.add_patch(goal_circ)

ax.legend(loc='upper left', fontsize=12)
plt.savefig('/home/trizzz/.gemini/antigravity/brain/51b1597f-c9bf-41b0-a47d-c1293dbea528/artifacts/map_layout.png', dpi=300, bbox_inches='tight')
print(f"Generated uniform map layout. Placed {len(static_obstacles)} static and {len(dynamic_obstacles)} dynamic obstacles.")
