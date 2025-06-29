import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define a hyperbolic paraboloid surface (saddle shape)
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)
z = y**2 - x**2

# Point on the surface
px, py = 0.5, -0.5
pz = py**2 - px**2
p = np.array([px, py, pz])

# Compute partial derivatives (tangent vectors)
dz_dx = -2 * px
dz_dy = 2 * py
du = np.array([1, 0, dz_dx])
dv = np.array([0, 1, dz_dy])

# Create grid for tangent plane
span = np.linspace(-0.5, 0.5, 10)
U, V = np.meshgrid(span, span)
tx = px + U
ty = py + V
tz = pz + dz_dx * U + dz_dy * V

# Define a simpler curve on the surface: a straight line in x, fixed y
t_vals = np.linspace(-0.5, 0.5, 100)
curve_x = px + t_vals
curve_y = np.full_like(t_vals, py)
curve_z = curve_y**2 - curve_x**2

# Projection of this curve onto the tangent plane
curve_u = curve_x - px
curve_v = curve_y - py
proj_x = px + curve_u
proj_y = py + curve_v
proj_z = pz + dz_dx * curve_u + dz_dy * curve_v

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, color='lightgray', alpha=0.6, linewidth=0)
ax.plot_surface(tx, ty, tz, alpha=0.8, color='#4C72B0')
ax.plot(curve_x, curve_y, curve_z, color='#117733', label='Curve on surface', linewidth=2)
ax.plot(proj_x, proj_y, proj_z, color='#882255', linestyle='--', label='Projection', linewidth=2)

ax.text(*(p + 0.5), "p", color='black', fontsize=20)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])
plt.legend()
plt.tight_layout()
plt.show()

