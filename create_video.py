#!/usr/bin/env python3
"""
Script to create a video (myfile.mp4) showing piston compression simulation.
Extracted from compression.ipynb notebook.

This script simulates a piston with particles and creates a video showing
three different views (xy, xz, zy projections) of the simulation.
"""

import argparse
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from tqdm.auto import tqdm

from jarzynski import init_piston, forward

# Configuration constants
DEFAULT_NUM_PARTICLES = 50
DEFAULT_PARTICLE_RADIUS = 1e-1
DEFAULT_FPS = 10
DEFAULT_DPI = 200
DEFAULT_FIGURE_SIZE = (9, 3)
DEFAULT_PLOT_LIMITS = (-1.1, 1.1)
DEFAULT_CIRCLE_POINTS = 100
DEFAULT_INITIAL_DURATION = 3.0
DEFAULT_COMPRESSION_DURATION = 10.0
DEFAULT_FINAL_DURATION = 3.0
DEFAULT_VELOCITY = 1.0

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

# JIT compile functions for performance
init_piston_j = jax.jit(init_piston, static_argnums=1)
forward_j = jax.jit(forward)


def plot_piston(state: Dict[str, Any], proj: Callable, draw_walls: bool = True) -> None:
    """
    Plot the piston state using a given projection function.

    Args:
        state: Dictionary containing piston state with 'balls' and 'walls' keys
        proj: Projection function to map 3D coordinates to 2D
        draw_walls: Whether to draw the piston walls
    """
    plt.axis("off")
    plt.axis("square")
    plt.xlim(*DEFAULT_PLOT_LIMITS)
    plt.ylim(*DEFAULT_PLOT_LIMITS)

    # Plot particles as circles
    radius = state["balls"]["r"]
    positions_2d = proj(state["balls"]["x"])
    phi = jnp.linspace(0, 2 * jnp.pi, DEFAULT_CIRCLE_POINTS)

    x_coords = positions_2d[:, 0] + radius * jnp.cos(phi[:, None])
    y_coords = positions_2d[:, 1] + radius * jnp.sin(phi[:, None])
    plt.plot(x_coords, y_coords)

    # Draw walls if requested
    if draw_walls:
        wall_origin = proj(state["walls"]["x"])
        wall_j = proj(state["walls"]["j"])
        wall_k = proj(state["walls"]["k"])
        wall_path = [
            wall_origin,
            wall_origin + wall_j,
            wall_origin + wall_j + wall_k,
            wall_origin + wall_k,
            wall_origin,
        ]
        plt.plot(
            [point[:, 0] for point in wall_path],
            [point[:, 1] for point in wall_path],
            "white",
        )


def xy_projection(pos: jnp.ndarray) -> jnp.ndarray:
    """Project 3D coordinates to xy plane."""
    return pos[..., [0, 1]]


def xz_projection(pos: jnp.ndarray) -> jnp.ndarray:
    """Project 3D coordinates to xz plane."""
    return pos[..., [0, 2]]


def zy_projection(pos: jnp.ndarray) -> jnp.ndarray:
    """Project 3D coordinates to zy plane."""
    return pos[..., [2, 1]]


def setup_figure() -> tuple:
    """
    Set up matplotlib figure with dark background and three subplots.

    Returns:
        Tuple of (figure, [ax1, ax2, ax3])
    """
    plt.style.use("dark_background")
    return plt.subplots(1, 3, figsize=DEFAULT_FIGURE_SIZE)


def update_views(state: Dict[str, Any], ax1, ax2, ax3) -> None:
    """
    Update all three views of the piston state.

    Args:
        state: Current piston state
        ax1, ax2, ax3: Matplotlib axes for the three views
    """
    # XZ view (side view)
    plt.sca(ax1)
    plt.cla()
    plot_piston(state, xz_projection)

    # XY view (top view) with circular boundary
    plt.sca(ax2)
    plt.cla()
    plot_piston(state, xy_projection, False)
    phi = jnp.linspace(0, 2 * jnp.pi, DEFAULT_CIRCLE_POINTS)
    plt.plot(jnp.cos(phi), jnp.sin(phi), "white")

    # ZY view (front view)
    plt.sca(ax3)
    plt.cla()
    plot_piston(state, zy_projection)

    plt.tight_layout()


def initialize_simulation(
    num_particles: int = DEFAULT_NUM_PARTICLES,
    particle_radius: float = DEFAULT_PARTICLE_RADIUS,
) -> Dict[str, Any]:
    """
    Initialize the piston simulation state.

    Args:
        num_particles: Number of particles in the simulation
        particle_radius: Radius of each particle

    Returns:
        Initial simulation state
    """
    print(f"Initializing piston state with {num_particles} particles...")
    return init_piston_j(jax.random.PRNGKey(0), num_particles, particle_radius)


def record_simulation(
    state: Dict[str, Any],
    duration_seconds: float,
    fps: int,
    dt: float,
    moviewriter,
    ax1,
    ax2,
    ax3,
) -> Dict[str, Any]:
    """
    Record simulation frames for a given duration.

    Args:
        state: Current simulation state
        duration_seconds: How long to record in seconds
        fps: Frames per second
        dt: Time step for simulation
        moviewriter: FFMpeg writer for video
        ax1, ax2, ax3: Matplotlib axes

    Returns:
        Updated simulation state
    """
    total_frames = round(duration_seconds * fps)
    for _ in tqdm(range(total_frames), desc=f"Recording {duration_seconds}s"):
        update_views(state, ax1, ax2, ax3)
        moviewriter.grab_frame()
        _, state, _ = forward_j(dt, state)
    return state


def create_video(
    filename: str = "myfile.mp4",
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    initial_duration: float = DEFAULT_INITIAL_DURATION,
    compression_duration: float = DEFAULT_COMPRESSION_DURATION,
    final_duration: float = DEFAULT_FINAL_DURATION,
    velocity: float = DEFAULT_VELOCITY,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    particle_radius: float = DEFAULT_PARTICLE_RADIUS,
    enable_compression: bool = True,
) -> None:
    """
    Create a video showing piston compression simulation.

    Args:
        filename: Output video filename
        fps: Frames per second for the video
        dpi: DPI for video quality
        initial_duration: Duration of initial phase (particles settling)
        compression_duration: Duration of compression phase
        final_duration: Duration of final phase (after compression)
        velocity: Compression velocity
        num_particles: Number of particles in simulation
        particle_radius: Radius of each particle
        enable_compression: Whether to include compression phases
    """
    print("Creating video...")
    total_duration = initial_duration + (
        compression_duration + final_duration if enable_compression else 0
    )
    print(f"Total video duration: {total_duration:.1f} seconds")

    try:
        # Set up visualization
        fig, [ax1, ax2, ax3] = setup_figure()
        moviewriter = FFMpegFileWriter(fps=fps)

        # Calculate time step for simulation
        dt = (1.5 / velocity) / (compression_duration * fps)

        # Initialize simulation
        state = initialize_simulation(num_particles, particle_radius)

        # Record video
        print(f"Recording video to {filename}...")
        with moviewriter.saving(fig, filename, dpi=dpi):
            # Phase 1: Initial state (particles settling)
            print("Phase 1: Initial state")
            state = record_simulation(
                state, initial_duration, fps, dt, moviewriter, ax1, ax2, ax3
            )

            if enable_compression:
                # Phase 2: Compression
                print("Phase 2: Compression")
                state["walls"]["v"] = jnp.array(
                    [
                        [0.0, 0.0, velocity / 2],
                        [0.0, 0.0, -velocity / 2],
                    ]
                )
                state = record_simulation(
                    state, compression_duration, fps, dt, moviewriter, ax1, ax2, ax3
                )

                # Phase 3: Stop compression and equilibrate
                print("Phase 3: Final equilibration")
                state["walls"]["v"] = jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
                state = record_simulation(
                    state, final_duration, fps, dt, moviewriter, ax1, ax2, ax3
                )

        plt.close(fig)
        print(f"Video created successfully: {filename}")

    except Exception as e:
        print(f"Error creating video: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Create a video showing piston compression simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o", "--output", default="myfile.mp4", help="Output video filename"
    )

    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS, help="Frames per second for the video"
    )

    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help="DPI for video quality"
    )

    parser.add_argument(
        "--initial-duration",
        type=float,
        default=DEFAULT_INITIAL_DURATION,
        help="Duration of initial phase in seconds",
    )

    parser.add_argument(
        "--compression-duration",
        type=float,
        default=DEFAULT_COMPRESSION_DURATION,
        help="Duration of compression phase in seconds",
    )

    parser.add_argument(
        "--final-duration",
        type=float,
        default=DEFAULT_FINAL_DURATION,
        help="Duration of final phase in seconds",
    )

    parser.add_argument(
        "--velocity", type=float, default=DEFAULT_VELOCITY, help="Compression velocity"
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_NUM_PARTICLES,
        help="Number of particles in simulation",
    )

    parser.add_argument(
        "--radius", type=float, default=DEFAULT_PARTICLE_RADIUS, help="Particle radius"
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable compression phases (only show initial state)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the video creation with command-line arguments.
    """
    args = parse_arguments()

    create_video(
        filename=args.output,
        fps=args.fps,
        dpi=args.dpi,
        initial_duration=args.initial_duration,
        compression_duration=args.compression_duration,
        final_duration=args.final_duration,
        velocity=args.velocity,
        num_particles=args.particles,
        particle_radius=args.radius,
        enable_compression=not args.no_compression,
    )


if __name__ == "__main__":
    main()
