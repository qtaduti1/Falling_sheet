import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import csv

# Physical parameters
mass = 0.25
g = 9.81
height = 1.5
length = 0.20
width = 0.15
drag_coefficient = 1.2
air_density = 1.225
time_step = 0.01

# Corrected initial conditions
position = np.array([0.0, 0.0, 1.5])  # Starting at exactly (0, 0, 1.5)
velocity = np.array([0.2, 0.2, -0.2])  # Initial velocity exactly (0, 0, -0.2)
orientation = np.radians(30)
angular_velocity = 0.0

# Derived properties
area = length * width
moment_of_inertia = (1/12) * mass * (length**2 + width**2)

# Data collection arrays
times = []
positions = []
velocities = []
speeds = []
orientations = []
angular_velocities = []
projected_areas = []
angular_momentums = []

# Record initial state before any time steps
times.append(0.0)
positions.append(position.copy())
velocities.append(velocity.copy())
speeds.append(np.linalg.norm(velocity))
orientations.append(orientation)
angular_velocities.append(angular_velocity)
projected_area = width * abs(np.sin(orientation)) + length * abs(np.cos(orientation))
projected_areas.append(projected_area)
angular_momentums.append(moment_of_inertia * angular_velocity)

# Simulation loop starts after recording initial state
time = time_step
while position[2] > 0:
    # Calculate projected area based on orientation
    projected_area = width * abs(np.sin(orientation)) + length * abs(np.cos(orientation))
    
    # Calculate drag force with normal and tangential components
    speed = np.linalg.norm(velocity)
    if speed > 0:
        # Sheet normal vector
        normal_vector = np.array([np.sin(orientation), 0, np.cos(orientation)])
        normal_vector /= np.linalg.norm(normal_vector)  # Ensure normalization
        
        # Calculate normal component of velocity
        normal_component = np.dot(velocity, normal_vector)
        normal_vel = normal_component * normal_vector
        
        # Calculate tangential component
        tangential_vel = velocity - normal_vel
        
        # Calculate drag forces (normal has higher coefficient)
        normal_drag_magnitude = 0.5 * air_density * drag_coefficient * area * (np.linalg.norm(normal_vel)**2)
        tangential_drag_magnitude = 0.5 * air_density * (drag_coefficient*0.1) * area * (np.linalg.norm(tangential_vel)**2)
        
        # Apply drag forces in appropriate directions
        if np.linalg.norm(normal_vel) > 0:
            normal_drag = -normal_drag_magnitude * normal_vel / np.linalg.norm(normal_vel)
        else:
            normal_drag = np.array([0, 0, 0])
            
        if np.linalg.norm(tangential_vel) > 0:
            tangential_drag = -tangential_drag_magnitude * tangential_vel / np.linalg.norm(tangential_vel)
        else:
            tangential_drag = np.array([0, 0, 0])
            
        # Total drag force
        drag_force = normal_drag + tangential_drag
    else:
        drag_force = np.array([0, 0, 0])
    
    # Calculate net forces
    weight = np.array([0, 0, -mass * g])
    net_force = weight + drag_force
    acceleration = net_force / mass
    
    # Update linear motion
    velocity += acceleration * time_step
    position += velocity * time_step
    
    # Calculate torque
    if speed > 0:
        velocity_angle = np.arctan2(velocity[0], -velocity[2])
        relative_angle = orientation - velocity_angle
        torque = 0.01 * speed**2 * np.sin(2 * relative_angle)
    else:
        torque = 0
    
    # Update angular motion
    torque_damping = -0.005 * angular_velocity
    net_torque = torque + torque_damping
    angular_acceleration = net_torque / moment_of_inertia
    angular_velocity += angular_acceleration * time_step
    orientation += angular_velocity * time_step
    
    # Store data
    angular_momentum = moment_of_inertia * angular_velocity
    times.append(time)
    positions.append(position.copy())
    velocities.append(velocity.copy())
    speeds.append(speed)
    orientations.append(orientation)
    angular_velocities.append(angular_velocity)
    projected_areas.append(projected_area)
    angular_momentums.append(angular_momentum)
    
    time += time_step
# Convert data to numpy arrays for plotting and analysis (unchanged)
times = np.array(times)
positions = np.array(positions)
velocities = np.array(velocities)
speeds = np.array(speeds)
orientations = np.array(orientations)
angular_velocities = np.array(angular_velocities)
projected_areas = np.array(projected_areas)
angular_momentums = np.array(angular_momentums)

# Visualization code remains unchanged...
# Output data at intervals
interval = 0.05
interval_indices = []
for i in range(int(times[-1]/interval) + 1):
    target_time = i * interval
    closest_idx = np.argmin(np.abs(times - target_time))
    interval_indices.append(closest_idx)

print("\nSimulation Data at 0.05s Intervals:\n")
print(f"{'Time (s)':>8} {'X Pos (m)':>10} {'Y Pos (m)':>10} {'Z Pos (m)':>10} {'Velocity X (m/s)':>15} {'Velocity Y (m/s)':>15} {'Velocity Z (m/s)':>15} {'Velocity Mag (m/s)':>18} {'Angle (deg)':>12} {'Angular Mom':>12} {'Proj Area':>10}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*18} {'-'*12} {'-'*12} {'-'*10}")

for idx in interval_indices:
    t = times[idx]
    x_pos = positions[idx][0]
    y_pos = positions[idx][1]
    z_pos = positions[idx][2]
    vel_x = velocities[idx][0]
    vel_y = velocities[idx][1]
    vel_z = velocities[idx][2]
    vel_mag = speeds[idx]
    angle = np.degrees(orientations[idx])
    ang_mom = angular_momentums[idx]
    proj_area = projected_areas[idx]
    print(f"{t:8.2f} {x_pos:10.3f} {y_pos:10.3f} {z_pos:10.3f} {vel_x:15.3f} {vel_y:15.3f} {vel_z:15.3f} {vel_mag:18.3f} {angle:12.3f} {ang_mom:12.6f} {proj_area:10.3f}")

with open('sheet_data_intervals.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time (s)', 'X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'Velocity X (m/s)', 'Velocity Y (m/s)', 'Velocity Z (m/s)', 'Velocity Mag (m/s)', 'Angle (deg)', 'Angular Mom', 'Proj Area'])
    for idx in interval_indices:
        t = times[idx]
        x_pos = positions[idx][0]
        y_pos = positions[idx][1]
        z_pos = positions[idx][2]
        vel_x = velocities[idx][0]
        vel_y = velocities[idx][1]
        vel_z = velocities[idx][2]
        vel_mag = speeds[idx]
        angle = np.degrees(orientations[idx])
        ang_mom = angular_momentums[idx]
        proj_area = projected_areas[idx]
        writer.writerow([f"{t:.2f}", f"{x_pos:.3f}", f"{y_pos:.3f}", f"{z_pos:.3f}", f"{vel_x:.3f}", f"{vel_y:.3f}", f"{vel_z:.3f}", f"{vel_mag:.3f}", f"{angle:.3f}", f"{ang_mom:.6f}", f"{proj_area:.3f}"])

# Create data table figure
fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
ax.axis('off')
table_data = []
for idx in interval_indices:
    t = times[idx]
    x_pos = positions[idx][0]
    y_pos = positions[idx][1]
    z_pos = positions[idx][2]
    vel_x = velocities[idx][0]
    vel_y = velocities[idx][1]
    vel_z = velocities[idx][2]
    vel_mag = speeds[idx]
    angle = np.degrees(orientations[idx])
    ang_mom = angular_momentums[idx]
    proj_area = projected_areas[idx]
    table_data.append([f"{t:.2f}", f"{x_pos:.3f}", f"{y_pos:.3f}", f"{z_pos:.3f}", f"{vel_x:.3f}", f"{vel_y:.3f}", f"{vel_z:.3f}", f"{vel_mag:.3f}", f"{angle:.3f}", f"{ang_mom:.6f}", f"{proj_area:.3f}"])
table = ax.table(cellText=table_data, 
                loc='center', 
                colLabels=['Time (s)', 'X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'Velocity X (m/s)', 'Velocity Y (m/s)', 'Velocity Z (m/s)', 'Velocity Mag (m/s)', 'Angle (deg)', 'Angular Mom', 'Proj Area'],
                cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
plt.title('Sheet Simulation Data at 0.05s Intervals')
plt.tight_layout()
plt.savefig('sheet_data_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Create visualization plots with y-axis values included
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(positions[:, 0], positions[:, 2], 'b-')
plt.title('Sheet Trajectory (X vs Z)')
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(times, positions[:, 0], 'g-', label='X Position')
plt.plot(times, positions[:, 1], 'b-', label='Y Position')
plt.plot(times, positions[:, 2], 'r-', label='Z Position')
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(times, speeds, 'k-', label='Magnitude')
plt.plot(times, velocities[:, 0], 'g-', label='X Component')
plt.plot(times, velocities[:, 1], 'b-', label='Y Component')
plt.plot(times, velocities[:, 2], 'r-', label='Z Component')
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(times, np.degrees(orientations), 'b-')
plt.title('Orientation Angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(times, angular_momentums, 'm-')
plt.title('Angular Momentum vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Momentum (kg·m²/s)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(times, projected_areas, 'c-')
plt.title('Projected Area vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Projected Area (m²)')
plt.grid(True)

plt.tight_layout()
plt.savefig('sheet_graphs.png', dpi=300)
plt.show()

# 3D Animation 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Animation of Falling Sheet')

def update(frame):
    ax.clear()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Animation of Falling Sheet')
    
    max_x = max([abs(p[0]) for p in positions])
    x_min, x_max = -max_x-0.1, max_x+0.1
    y_min, y_max = -0.2, 0.2
    z_min, z_max = 0, height+0.1
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    if frame > 0:
        trajectory = np.array([positions[i] for i in range(frame)])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'k-', alpha=0.3)
    
    x, y, z = positions[frame]
    angle = orientations[frame]
    
    half_length = length / 2
    half_width = width / 2
    
    local_corners = np.array([
        [-half_length, -half_width, 0],
        [half_length, -half_width, 0],
        [half_length, half_width, 0],
        [-half_length, half_width, 0]
    ])
    
    c, s = np.cos(angle), np.sin(angle)
    rot_y = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    
    global_corners = np.zeros_like(local_corners)
    for i, corner in enumerate(local_corners):
        rotated = np.dot(rot_y, corner)
        global_corners[i] = [rotated[0] + x, rotated[1] + y, rotated[2] + z]
    
    verts = [list(zip(global_corners[:, 0], global_corners[:, 1], global_corners[:, 2]))]
    poly = Poly3DCollection(verts, alpha=0.7, linewidths=1, edgecolors='black')
    poly.set_facecolor('cyan')
    ax.add_collection3d(poly)
    
    ax.scatter([x], [y], [z], color='blue', s=30)
    
    if frame < len(velocities):
        vel = velocities[frame]
        vel_magnitude = speeds[frame]
        if vel_magnitude > 0:
            arrow_scale = 0.1
            ax.quiver(x, y, z, arrow_scale*vel[0], arrow_scale*vel[1], arrow_scale*vel[2], color='red', arrow_length_ratio=0.2)
    
    ax.view_init(elev=20, azim=30)
    return poly,

num_frames = 50
frame_indices = np.linspace(0, len(times)-1, num_frames, dtype=int)
ani = FuncAnimation(fig, update, frames=frame_indices, blit=True, interval=50)
writer = PillowWriter(fps=20)
ani.save('sheet_falling_animation.gif', writer=writer)
plt.show()
