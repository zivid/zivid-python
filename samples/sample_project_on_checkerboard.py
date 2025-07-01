"""Sample demonstrating how to project an image onto a point in 3D space."""

from datetime import timedelta

import numpy as np
from zivid import Application, Settings, Settings2D
from zivid.calibration import detect_feature_points
from zivid.projection import pixels_from_3d_points, projector_resolution, show_image_bgra


def _detect_checkerboard(camera):
    print("Detecting checkerboard...")
    settings = Settings()
    settings.acquisitions.append(Settings.Acquisition())
    settings.color = Settings2D()
    settings.color.acquisitions.append(Settings2D.Acquisition())
    with camera.capture_2d_3d(settings) as frame:
        detection_result = detect_feature_points(frame.point_cloud())
        if not detection_result.valid():
            raise RuntimeError("Failed to detect checkerboard")
        print("Successfully detected checkerboard")
        return detection_result


def _create_image_bgra_array(camera, detection_result):
    resolution = projector_resolution(camera)
    print(f"Projector resolution: {resolution}")

    channels_bgra = 4
    resolution_bgra = resolution + (channels_bgra,)
    image_bgra = np.zeros(resolution_bgra, dtype=np.uint8)

    # Draw frame around projector FOV
    image_bgra[:5, :, :] = 255
    image_bgra[-5:, :, :] = 255
    image_bgra[:, :5, :] = 255
    image_bgra[:, -5:, :] = 255

    # Draw circle at checkerboard centroid
    centroid_xyz = list(detection_result.centroid())
    print(f"Located checkerboard at xyz={centroid_xyz}")
    centroid_projector_xy = pixels_from_3d_points(camera, [centroid_xyz])[0]
    print(f"Projector coords (x,y) corresponding to centroid: {centroid_projector_xy}")
    col = round(centroid_projector_xy[0])
    row = round(centroid_projector_xy[1])
    print(f"Projector pixel corresponding to centroid: row={row}, col={col}")
    for i in np.arange(-10, 10, 1):
        for j in np.arange(-10, 10, 1):
            dist = np.sqrt(i**2 + j**2)
            if dist <= 5:
                color = (0, 0, 255, 0)
            elif 5 < dist < 7:
                color = (255, 255, 255, 0)
            else:
                color = (0, 0, 0, 0)
            image_bgra[row + i, col + j, :] = color

    return image_bgra


def _capture_image_of_projection(projected_image):
    settings_2d = Settings2D()
    settings_2d.acquisitions.append(
        Settings2D.Acquisition(
            brightness=0.0,
            exposure_time=timedelta(milliseconds=50),
        )
    )
    settings_2d.processing.color.gamma = 0.75

    with projected_image.capture(settings_2d) as frame2d:
        filename = "projection_picture.png"
        print(f"Saving image of projection to {filename}")
        frame2d.image_rgba().save(filename)


def _main():
    app = Application()

    print("Connecting to camera...")
    with app.connect_camera() as camera:
        detection_result = _detect_checkerboard(camera)
        image_bgra = _create_image_bgra_array(camera, detection_result)

        with show_image_bgra(camera, image_bgra) as projected_image:
            _capture_image_of_projection(projected_image)
            input("Press enter to stop projection")


if __name__ == "__main__":
    _main()
