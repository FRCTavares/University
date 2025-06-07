import matplotlib.pyplot as plt
import numpy as np

# Feature categories
categories = [
    "Edge Processing", 
    "Data Privacy", 
    "Real-Time Alerts", 
    "Modularity", 
    "Scalability", 
    "Ease of Integration", 
    "Cost Efficiency"
]
N = len(categories)

# Competitor scores on a 0–10 scale
smartvision = [9, 10, 9, 9, 9, 8, 8]
intenseye =    [2, 5, 7, 7, 7, 7, 4]
protex_ai =    [7, 9, 8, 8, 8, 6, 6]
modjoul =      [3, 6, 6, 5, 6, 5, 5]

# Close the radar loop
categories += [categories[0]]
smartvision += [smartvision[0]]
intenseye += [intenseye[0]]
protex_ai += [protex_ai[0]]
modjoul += [modjoul[0]]

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each company
ax.plot(angles, smartvision, label='SmartVision', linewidth=2, color='#0037FF') 
ax.fill(angles, smartvision, alpha=0.1, color="#0037FF")

ax.plot(angles, intenseye, label='Intenseye', linewidth=2, color="#36FF33")  
ax.fill(angles, intenseye, alpha=0.1, color='#36FF33')

ax.plot(angles, protex_ai, label='Protex AI', linewidth=2, color='#FF33A8')  
ax.fill(angles, protex_ai, alpha=0.1, color='#FF33A8')

ax.plot(angles, modjoul, label='Modjoul', linewidth=2, color="#B68F59")  
ax.fill(angles, modjoul, alpha=0.1, color='#B68F59')

# Configure chart
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=11)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10)
ax.set_rlabel_position(0)
ax.spines['polar'].set_color('gray')
ax.spines['polar'].set_linewidth(1)

# Add legend and adjust layout
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)
plt.tight_layout()
plt.show()
