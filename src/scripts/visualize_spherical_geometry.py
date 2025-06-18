import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define colors from a "deep" palette for consistent styling
# These hex codes are derived from matplotlib's default 'deep' color cycle
DEEP_BLUE = '#4C72B0'
DEEP_ORANGE = '#DD8452'
DEEP_GREEN = '#55A868'
DEEP_RED = '#C44E52'
DEEP_PURPLE = '#8172B3'
DEEP_BROWN = '#937860'
NEUTRAL_GRAY = '#6A6A6A' # A good neutral color for text if specific palette color isn't suitable

# Function to convert spherical to Cartesian coordinates
def sph_to_cart(r, theta, phi):
    """
    Converts spherical coordinates (radius, azimuthal angle, polar angle)
    to Cartesian coordinates (x, y, z).
    Args:
        r (float): Radius.
        theta (float): Azimuthal angle (radians, typically 0 to 2*pi).
        phi (float): Polar angle (radians, typically 0 to pi).
    Returns:
        tuple: (x, y, z) Cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

# Function to plot an arc (segment of a great circle)
def plot_arc(p1, p2, num=100):
    """
    Generates points for a great circle arc between two points on a unit sphere.
    Args:
        p1 (np.array): Cartesian coordinates of the first point on the unit sphere.
        p2 (np.array): Cartesian coordinates of the second point on the unit sphere.
        num (int): Number of points to generate for the arc.
    Returns:
        np.array: An array of (x, y, z) coordinates forming the arc.
    """
    p1, p2 = np.array(p1), np.array(p2)
    # Calculate the angle between the two points to determine the arc length
    angle = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    # Check for collinear points to avoid division by zero or invalid calculations
    if np.isclose(angle, 0.0): # Points are the same
        return np.array([p1])
    if np.isclose(angle, np.pi): # Points are antipodal
        # In this case, any great circle through them is valid.
        # We'll just return a line to handle this edge case gracefully,
        # though a proper great circle requires more definition.
        return np.array([p1 + t * (p2 - p1) for t in np.linspace(0, 1, num)])

    # Generate points along the great circle arc
    return np.array([
        (np.sin((1 - t) * angle) * p1 + np.sin(t * angle) * p2) / np.sin(angle)
        for t in np.linspace(0, 1, num)
    ])

# Define the sphere surface for plotting
phi, theta = np.mgrid[0:np.pi:100j, 0:2 * np.pi:100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Define spherical coordinates for the triangle points
points_spherical_positive_xyz = [
    (np.pi / 3, np.pi / 10),              # Point P1 (closer to X-axis, adjusted)
    (np.pi / 3, np.pi / 2 - np.pi / 10),  # Point P2 (mid-way, adjusted)
    (np.pi / 6, np.pi / 4)                # Point P3 (closer to Z-axis, adjusted)
]

# Convert triangle points to Cartesian coordinates (on a unit sphere)
points_cartesian_positive_xyz = [sph_to_cart(1, t, p) for p, t in points_spherical_positive_xyz]
P1, P2, P3 = points_cartesian_positive_xyz

# Define full great circles for visualization
circle_theta = np.linspace(0, 2 * np.pi, 200)

# Great circle in the XZ-plane (phi varies, theta constant at 0 or pi)
circle1 = np.array([sph_to_cart(1, 0, t) for t in circle_theta])

# Great circle in the XY-plane (phi constant at pi/2, theta varies)
circle2 = np.array([sph_to_cart(1, t, np.pi / 2) for t in circle_theta])

# Create figure and 3D axis for the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere surface with the requested lightgray color
ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, rstride=1, cstride=1, edgecolor='none')

# Plot triangle vertices using a deep palette color
ax.scatter(*zip(*points_cartesian_positive_xyz), color=DEEP_ORANGE, s=50)

# Plot triangle edges (great arcs) using a deep palette color
for start, end in [(P1, P2), (P2, P3), (P3, P1)]:
    arc = plot_arc(start, end)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=DEEP_ORANGE, linewidth=2) # Thicker lines for arcs

# Annotate triangle points with labels and a suitable neutral color
labels = ['A', 'B', 'C']
for point, label in zip(points_cartesian_positive_xyz, labels):
    # Adjust annotation position slightly to avoid overlap with the point
    ax.text(point[0]*1.05, point[1]*1.05, point[2]*1.05, label, fontsize=12, color=NEUTRAL_GRAY, ha='center', va='center')

# Plot full great circles using different deep palette colors
ax.plot(circle1[:, 0], circle1[:, 1], circle1[:, 2], color=DEEP_GREEN, linestyle='--', linewidth=1.5, label='Great Circle 1 (XZ-plane)')
ax.plot(circle2[:, 0], circle2[:, 1], circle2[:, 2], color=DEEP_PURPLE, linestyle='--', linewidth=1.5, label='Great Circle 2 (XY-plane)')


# Add quivers for X, Y, Z axes using deep palette colors and labels
#origin = [0, 0, 0] # Origin for the quivers
#
## X-axis quiver and label
#ax.quiver(*origin, 0.5, 0, 0, color=DEEP_RED, length=1.2, arrow_length_ratio=0.15, linewidth=2)
#ax.text(0.7, 0, 0, 'X', color=DEEP_RED, fontsize=12, ha='center', va='center')
#
## Y-axis quiver and label
#ax.quiver(*origin, 0, 0.5, 0, color=DEEP_BLUE, length=1.2, arrow_length_ratio=0.15, linewidth=2)
#ax.text(0, 0.7, 0, 'Y', color=DEEP_BLUE, fontsize=12, ha='center', va='center')
#
## Z-axis quiver and label
#ax.quiver(*origin, 0, 0, 0.5, color=DEEP_GREEN, length=1.2, arrow_length_ratio=0.15, linewidth=2)
#ax.text(0, 0, 0.7, 'Z', color=DEEP_GREEN, fontsize=12, ha='center', va='center')

# Set plot aesthetics
ax.set_box_aspect([1, 1, 1]) # Ensure equal aspect ratio for a spherical look
ax.axis('off') # Hide axes for a cleaner view

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

