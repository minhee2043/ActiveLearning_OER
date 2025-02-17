# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams.update({
    'font.family': 'Arial',      # Use Arial font for all text
    'font.size': 9,              # Base font size
    'figure.figsize': (3.2, 2.3),# Figure dimensions in inches
    'figure.dpi': 600,           # High resolution output
    'font.weight': 'bold'        # Bold text throughout
})

# Define physical constants and parameters
kb = 8.617333262e-5  # Boltzmann constant in eV/K
T = 300              # Temperature in Kelvin
E_opt = 5.3          # Optimal energy level in eV

# Generate composition ratios for ternary plot
# Creates a grid of possible compositions where x + y + z = 1
steps = 18          # Number of steps for discretization
step_size = 1.0 / (steps - 1)

possibleratio = []
for i in range(steps):
    x = round(i * step_size, 3)
    for j in range(steps - i):
        y = round(j * step_size, 3)
        z = round(1.0 - x - y, 3)
        # Ensure compositions sum to 1 within numerical precision
        if z >= 0 and abs(x + y + z - 1.0) < 1e-10:
            possibleratio.append((x, y, z))

# Calculate activities for each composition
activities = []
for ratio in possibleratio:
    activity = 0
    # Read activity data from CSV file
    with open('batch15_count.csv') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # Calculate probability based on composition
            possibility = math.pow(ratio[0], int(row[0])) * \
                         math.pow(ratio[1], int(row[1])) * \
                         math.pow(ratio[2], int(row[2]))
            
            # Calculate Boltzmann factor
            boltz_up = -abs(float(row[3])-E_opt)
            boltz_down = kb*T
            
            # Combine probability, Boltzmann factor, and count
            final = possibility * math.exp(boltz_up/boltz_down) * int(row[-1])
            activity += final
    activities.append(activity)

possibleratio = np.array(possibleratio)
activities = np.array(activities)

# Calculate coordinates for ternary plot
# Convert ternary coordinates to cartesian coordinates
y0 = 0.5 * np.sqrt(3) * possibleratio[:,2]
x0 = possibleratio[:,1] + y0/np.sqrt(3)

colors_custom = ['#3E1F00', '#FF9B66', '#FFFFFF', '#9B8CC5', '#4A3C89']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors_custom)

fig, ax = plt.subplots()

center_x = (0 + 1 + 0.5) / 3
center_y = (0 + 0 + 0.5*np.sqrt(3)) / 3

scale = 1.13 
triangle_points = np.array([
    [center_x + (0 - center_x)*scale, center_y + (0 - center_y)*scale],
    [center_x + (1 - center_x)*scale, center_y + (0 - center_y)*scale],
    [center_x + (0.5 - center_x)*scale, center_y + (0.5*np.sqrt(3) - center_y)*scale],
    [center_x + (0 - center_x)*scale, center_y + (0 - center_y)*scale]
])
plt.plot(triangle_points[:,0], triangle_points[:,1], 'k-', linewidth=0.4)

# Plot data points with activity-based coloring
scatter = plt.scatter(x0, y0, 
                     s=45,                
                     c=activities,        
                     cmap=custom_cmap,    
                     marker='o',
                     edgecolor='none',
                     vmin=0, vmax=0.01)   

# Add labels for triangle vertices (Ni, Fe, Co)
text_offset = 0.01
plt.text(triangle_points[0,0] - text_offset*2, triangle_points[0,1], 'Ni', 
         fontsize=9, ha='right', va='center', weight='bold')
plt.text(triangle_points[1,0] + text_offset*2, triangle_points[1,1], 'Fe', 
         fontsize=9, ha='left', va='center', weight='bold')
plt.text(triangle_points[2,0], triangle_points[2,1] + text_offset*1.5, 'Co', 
         fontsize=9, ha='center', va='bottom', weight='bold')


plt.axis('off')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.95)


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)

cbar = plt.colorbar(scatter, 
                   cax=cax,
                   ticks=np.linspace(0, 0.01, 6))


cbar.outline.set_linewidth(0.4)
cbar.ax.tick_params(width=0.4, length=2, labelsize=8)
cbar.set_label('Activity', size=8, labelpad=5, weight='bold')
cbar.ax.set_yticklabels([f'{x:.3f}' for x in np.linspace(0, 0.01, 6)], weight='bold')


plt.subplots_adjust(right=0.9)
plt.savefig('Activity_ternaryplot.png', bbox_inches='tight')
