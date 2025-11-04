"""
Sample demonstrating Zivid Visualizer functionality.

This sample shows how to use the Visualizer class to display point clouds
using a file camera for simplicity.
"""

from pathlib import Path

import zivid

file_camera_file_path = Path(__file__).parent.parent / "test" / "test_data" / "FileCameraZivid2M70.zfc"


def _main():
    print("Starting Visualizer sample")

    app = zivid.Application()

    camera = app.create_file_camera(file_camera_file_path)
    print("Created file camera")

    # Simple settings for capture
    settings = zivid.Settings(
        acquisitions=[zivid.Settings.Acquisition()],
        color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
    )

    frame = camera.capture_2d_3d(settings)
    print("Frame captured successfully")

    # Create visualizer
    visualizer = zivid.visualization.Visualizer()
    visualizer.set_window_title("Zivid Visualizer Sample")

    # Configure visualization settings
    visualizer.colors_enabled = True
    visualizer.meshing_enabled = True
    visualizer.axis_indicator_enabled = True

    print(f"Colors enabled: {visualizer.colors_enabled}")
    print(f"Meshing enabled: {visualizer.meshing_enabled}")
    print(f"Axis indicator enabled: {visualizer.axis_indicator_enabled}")

    # Show the frame
    print("Displaying frame...")
    visualizer.show(frame)
    visualizer.reset_to_fit()

    # Run the event loop - this will show the window and allow interaction
    print("Starting visualizer. Close the window to exit.")
    exit_code = visualizer.run()
    print(f"Visualizer exited with code: {exit_code}")


if __name__ == "__main__":
    _main()
