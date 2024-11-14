import plotly.graph_objects as go
import numpy as np
from consts import POSITIONS

# Create the data for visualization
def create_soccer_field():
    # Field dimensions
    width = 6  # meters
    length = 9  # meters
    
    # Create the vertices for the field rectangle
    field_vertices = np.array([
        [-width/2, -length/2, 0],  # Bottom left
        [width/2, -length/2, 0],   # Bottom right
        [width/2, length/2, 0],    # Top right
        [-width/2, length/2, 0],   # Top left
        [-width/2, -length/2, 0],  # Close the loop
    ])
    
    return field_vertices

# Convert positions to arrays for plotting
positions = POSITIONS  # Using the provided POSITIONS list
x_coords = [p['x'] / 100 for p in positions]  # Convert mm to meters
y_coords = [p['y'] / 100 for p in positions]
z_coords = [p['z'] / 100 for p in positions]
labels = [f"ID: {p['ids']}<br>Code: {p['code']}<br>Height: {p['height']}" for p in positions]

# Create the figure
fig = go.Figure()

# Add the soccer field
field_vertices = create_soccer_field()
fig.add_trace(go.Scatter3d(
    x=field_vertices[:, 0],
    y=field_vertices[:, 1],
    z=field_vertices[:, 2],
    mode='lines',
    name='Soccer Field',
    line=dict(color='green', width=3),
))

# Add the markers
fig.add_trace(go.Scatter3d(
    x=x_coords,
    y=y_coords,
    z=z_coords,
    mode='markers+text',
    name='Markers',
    marker=dict(
        size=8,
        color='red',
    ),
    text=labels,
    hoverinfo='text'
))

# Update the layout
fig.update_layout(
    scene=dict(
        aspectmode='data',
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        zaxis_title='Z (meters)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    title='Soccer Field and Marker Positions',
    showlegend=True,
)

# Show the plot
fig.show()