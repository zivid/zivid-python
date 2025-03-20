import time
import zivid
import numpy as np
import threading
from pathlib import Path
from datetime import timedelta
import open3d as o3d
import zivid.calibration


def transform_to_marker_frame(point_cloud, marker):
    boardPose = marker.pose
    transform = np.linalg.inv(boardPose.to_matrix())
    point_cloud.transform(transform)


def get_next_point_cloud(camera, settings):

    frame = camera.capture(settings)

    markers = zivid.calibration.detect_markers(
        frame,
        allowed_marker_ids=[4],
        marker_dictionary=zivid.calibration.MarkerDictionary.aruco4x4_100,
    )

    if markers.valid() and len(markers.detected_markers()) == 1:
        print("Board/Marker detected")
        marker = markers.detected_markers()[0]

        unorganized_point_cloud = frame.point_cloud().to_unorganized_point_cloud()
        transform_to_marker_frame(unorganized_point_cloud, marker)

        return unorganized_point_cloud
    else:
        print("Board/Marker not detected")
        return None


def to_open3d_point_cloud(point_cloud):

    if point_cloud is None:
        return o3d.geometry.PointCloud()

    xyz_flattened = point_cloud.copy_data("xyz")
    rgb_flattened = point_cloud.copy_data("rgba")[:, 0:3].astype(np.float32) / 255.0

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_flattened))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb_flattened)

    return point_cloud_open3d


app = zivid.Application()
camera = app.connect_camera()


settings = zivid.Settings.load(Path(__file__).parent / "demo_settings.yml")


class PointCloudApp:
    def __init__(self):
        o3d.visualization.gui.Application.instance.initialize()

        self.vis = o3d.visualization.O3DVisualizer("Point Cloud Viewer", 1024, 768)

        # Capture initial point cloud
        self.combined_upc = get_next_point_cloud(camera, settings)
        if self.combined_upc is None:
            raise RuntimeError("Failed to detect marker in initial capture.")
        self.open3d_pc = to_open3d_point_cloud(self.combined_upc)
        self.vis.add_geometry("pointcloud", self.open3d_pc)

        self.vis.setup_camera(
            field_of_view=90, eye=np.array([0, 0, -1000]), center=np.array([0, 0, 0]), up=np.array([0, -1, 0])
        )

        # Start a background thread to update the point cloud
        self.running = True
        self.iframes = 0
        threading.Thread(target=self.update_thread, daemon=True).start()

    def update_geometry(self, new_pcd):
        """Safely remove old geometry and add new geometry."""
        t0 = time.time()
        self.vis.remove_geometry("pointcloud")  # Remove old point cloud
        self.vis.add_geometry("pointcloud", new_pcd)  # Add new point cloud
        print(f"Update geometry time: {1000*(time.time() - t0):.2f} ms")

    def update_thread(self):
        assert self.combined_upc is not None
        while self.running:

            t0tot = time.time()

            print(self.iframes)

            if self.iframes == 0:
                input("Press enter to voxel downsample")
                self.combined_upc = self.combined_upc.voxel_downsampled(
                    voxel_size=4.0,
                    min_points_per_voxel=2,
                )
                open3d_pc = to_open3d_point_cloud(self.combined_upc)
                o3d.visualization.gui.Application.instance.post_to_main_thread(
                    self.vis, lambda: self.update_geometry(open3d_pc)
                )
                input("Press Enter to continue...")

            if self.iframes % 20 == 0:
                input("Press Enter to continue...")

            t0 = time.time()
            new_upc = get_next_point_cloud(camera, settings)
            print(f"get_next_point_cloud time: {1000*(time.time() - t0):.2f} ms")

            if new_upc is not None:
                # Got more data
                # Combine existing data with new data
                self.combined_upc = self.combined_upc.extended(new_upc)

                # Voxel downsample after combining to reduce data amount
                self.combined_upc = self.combined_upc.voxel_downsampled(
                    voxel_size=4.0,
                    min_points_per_voxel=1,
                )

            # Safely update geometry in the main thread
            open3d_pc = to_open3d_point_cloud(self.combined_upc)
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.vis, lambda: self.update_geometry(open3d_pc)
            )

            print(f"Total time: {1000*(time.time() - t0tot):.2f} ms")

            self.iframes += 1

    def run(self):
        o3d.visualization.gui.Application.instance.add_window(self.vis)
        o3d.visualization.gui.Application.instance.run()


if __name__ == "__main__":
    app = PointCloudApp()
    app.run()

    # o3d.visualization.draw_geometries(
    #    [app.pcd],
    #    zoom=0.7,
    #    front=[0.0, 0.0, -1.0],
    #    lookat=[0.0, 0.0, 600.0],
    #    up=[0.0, -1.0, 0.0],
    # )

    # time.sleep(0.5)
    # o3d.visualization.gui.Application.instance.initialize()
    # vis = o3d.visualization.O3DVisualizer("Point Cloud Viewer", 1024, 768)
    # vis.add_geometry("pointcloud", app.pcd)

    # vis.setup_camera(field_of_view=90, eye=np.array([0, 0, -1000]), center=np.array([0, 0, 0]), up=np.array([0, -1, 0]))
