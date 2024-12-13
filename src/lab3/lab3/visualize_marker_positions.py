import plotly.graph_objects as go
import numpy as np
from consts import POSITIONS, PATH


def show():
    # Convert positions to arrays for plotting
    positions = POSITIONS  # Using the provided POSITIONS list
    x_coords = [(420-p["x"]) / 100 for p in positions]  # Convert mm to meters
    y_coords = [p["y"] / 100 for p in positions]
    z_coords = [p["z"] / 100 for p in positions]
    labels = [
        f"ID: {p['ids'][0]}<br>Code: {p['ids'][0]}<br>Height: {p['height']}"
        for p in positions
    ]

    # Create the figure
    fig = go.Figure()

    # Add the markers
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers+text",
            name="Markers",
            marker=dict(
                size=8,
                color="red",
            ),
            text=labels,
            hoverinfo="text",
        )
    )

    # Update the layout
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        title="Soccer Field and Marker Positions",
        showlegend=True,
    )

    # add the path
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] / 100 for p in PATH],  # Convert mm to meters
            y=[p[1] / 100 for p in PATH],
            z=[0] * len(PATH),
            mode="lines",
            name="Path",
            line=dict(color="blue", width=3),
        )
    )   

    # Show the plot
    fig.show()


if __name__ == "__main__":
    show()
