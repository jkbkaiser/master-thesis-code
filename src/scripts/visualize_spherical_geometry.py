import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DEEP_BLUE = '#4C72B0'
DEEP_ORANGE = '#DD8452'
DEEP_GREEN = '#55A868'
DEEP_RED = '#C44E52'
DEEP_PURPLE = '#8172B3'
DEEP_BROWN = '#937860'
NEUTRAL_GRAY = '#6A6A6A'

def sph_to_cart(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def plot_arc(p1, p2, num=100):
    p1, p2 = np.array(p1), np.array(p2)
    angle = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    if np.isclose(angle, 0.0):
        return np.array([p1])
    if np.isclose(angle, np.pi):
        return np.array([p1 + t * (p2 - p1) for t in np.linspace(0, 1, num)])

    return np.array([
        (np.sin((1 - t) * angle) * p1 + np.sin(t * angle) * p2) / np.sin(angle)
        for t in np.linspace(0, 1, num)
    ])

phi, theta = np.mgrid[0:np.pi:100j, 0:2 * np.pi:100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

points_spherical_positive_xyz = [
    (np.pi / 3, np.pi / 10),
    (np.pi / 3, np.pi / 2 - np.pi / 10),
    (np.pi / 6, np.pi / 4)
]

points_cartesian_positive_xyz = [sph_to_cart(1, t, p) for p, t in points_spherical_positive_xyz]
P1, P2, P3 = points_cartesian_positive_xyz

circle_theta = np.linspace(0, 2 * np.pi, 200)
circle1 = np.array([sph_to_cart(1, 0, t) for t in circle_theta])
circle2 = np.array([sph_to_cart(1, t, np.pi / 2) for t in circle_theta])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, rstride=1, cstride=1, edgecolor='none')
ax.scatter(*zip(*points_cartesian_positive_xyz), color=DEEP_ORANGE, s=50)

for start, end in [(P1, P2), (P2, P3), (P3, P1)]:
    arc = plot_arc(start, end)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=DEEP_ORANGE, linewidth=2) # Thicker lines for arcs

labels = ['A', 'B', 'C']
for point, label in zip(points_cartesian_positive_xyz, labels):
    ax.text(point[0]*1.05, point[1]*1.05, point[2]*1.05, label, fontsize=12, color=NEUTRAL_GRAY, ha='center', va='center')

ax.plot(circle1[:, 0], circle1[:, 1], circle1[:, 2], color=DEEP_GREEN, linestyle='--', linewidth=1.5, label='Great Circle 1 (XZ-plane)')
ax.plot(circle2[:, 0], circle2[:, 1], circle2[:, 2], color=DEEP_PURPLE, linestyle='--', linewidth=1.5, label='Great Circle 2 (XY-plane)')

ax.set_box_aspect([1, 1, 1])
ax.axis('off')

plt.tight_layout()
plt.show()

