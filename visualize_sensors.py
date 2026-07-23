import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_sensing_modalities():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#f0f8ff') # Light ocean blue
    
    # 1. Plot the Ego AUV
    ego_pos = np.array([0, 0])
    ego_yaw = np.pi / 4  # Pointing top-right
    
    # Ego Boat shape
    boat = patches.Polygon([
        (ego_pos[0] + 5*np.cos(ego_yaw), ego_pos[1] + 5*np.sin(ego_yaw)),
        (ego_pos[0] + 3*np.cos(ego_yaw + 2.5), ego_pos[1] + 3*np.sin(ego_yaw + 2.5)),
        (ego_pos[0] + 3*np.cos(ego_yaw - 2.5), ego_pos[1] + 3*np.sin(ego_yaw - 2.5))
    ], closed=True, color='blue', zorder=5, label='Ego AUV')
    ax.add_patch(boat)
    
    # Ego Heading Vector
    ax.quiver(ego_pos[0], ego_pos[1], np.cos(ego_yaw), np.sin(ego_yaw), color='blue', scale=10, zorder=6)
    
    # Goal direction
    goal_pos = np.array([60, 50])
    ax.plot(goal_pos[0], goal_pos[1], marker='*', color='green', markersize=15, label='Target Goal')
    ax.plot([ego_pos[0], goal_pos[0]], [ego_pos[1], goal_pos[1]], 'g--', alpha=0.6, label='Relative Goal Vector (IMU)')
    
    # 2. Plot Static Obstacles
    obstacles = [
        np.array([-40, 30, 15]),
        np.array([20, -50, 20]),
        np.array([50, 10, 10])
    ]
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.8)
        ax.add_patch(circle)
        
    # 3. Plot LiDAR Beams (64 rays, 70m range)
    lidar_range = 70.0
    num_beams = 64
    angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
    
    # Simple raycast mock
    for i, angle in enumerate(angles):
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        hit_dist = lidar_range
        # Check intersection with static obstacles
        for obs in obstacles:
            to_obs = obs[:2] - ego_pos
            proj = np.dot(to_obs, ray_dir)
            if proj > 0:
                perp_dist = np.linalg.norm(to_obs - proj * ray_dir)
                if perp_dist < obs[2]:
                    dist_to_surface = proj - np.sqrt(obs[2]**2 - perp_dist**2)
                    if dist_to_surface < hit_dist:
                        hit_dist = dist_to_surface
                        
        end_pt = ego_pos + ray_dir * hit_dist
        if i == 0:
            ax.plot([ego_pos[0], end_pt[0]], [ego_pos[1], end_pt[1]], color='red', alpha=0.3, linewidth=1, label='64-Channel LiDAR')
        else:
            ax.plot([ego_pos[0], end_pt[0]], [ego_pos[1], end_pt[1]], color='red', alpha=0.3, linewidth=1)
            
    # Draw LiDAR Boundary
    lidar_boundary = plt.Circle((0, 0), lidar_range, color='red', fill=False, linestyle=':', alpha=0.5)
    ax.add_patch(lidar_boundary)
    
    # 4. Plot Neighbor Agents (V2V/AIS Tracking)
    neighbors = [
        np.array([-30, -20]),
        np.array([10, 40])
    ]
    vels = [
        np.array([10, 5]),
        np.array([-5, -10])
    ]
    
    for i, (n_pos, n_vel) in enumerate(zip(neighbors, vels)):
        n_boat = patches.Circle((n_pos[0], n_pos[1]), 3, color='orange', zorder=5, label='Neighbor (V2V/AIS)' if i==0 else "")
        ax.add_patch(n_boat)
        # Neighbor velocity vector
        ax.quiver(n_pos[0], n_pos[1], n_vel[0], n_vel[1], color='orange', scale=50, width=0.005)
        # Tracking line
        ax.plot([ego_pos[0], n_pos[0]], [ego_pos[1], n_pos[1]], color='orange', linestyle='-.', alpha=0.8)
        
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.set_title("AUV Local Sensing Modalities", fontsize=16, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    # Put legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    
    plt.tight_layout()
    out_file = "visualizations_contributions/sensing_modalities.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    plot_sensing_modalities()
