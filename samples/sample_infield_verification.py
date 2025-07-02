"""Sample demonstrating in-field verification (experimental)."""

import zivid
from zivid.experimental.calibration import (
    InfieldCorrectionInput,
    camera_correction_timestamp,
    detect_feature_points,
    has_camera_correction,
    verify_camera,
)


def _main():
    app = zivid.Application()
    with app.connect_camera() as camera:
        print("Checking current correction status")
        if has_camera_correction(camera):
            timestamp = camera_correction_timestamp(camera)
            print(f"Camera currently has a correction. Timestamp: {timestamp.strftime(r'%Y-%m-%d %H:%M:%S')}")

        print("Detecting feature points for verification...")
        detection_result = detect_feature_points(camera)
        print(f"Feature point detection: {detection_result}")
        infield_input = InfieldCorrectionInput(detection_result)
        print(f"In-field correction input: {infield_input}")
        if not infield_input.valid():
            raise RuntimeError(
                f"Capture not appropriate for in-field correction/verification."
                f" Reason: {infield_input.status_description()}"
            )
        camera_verification = verify_camera(infield_input)
        print(f"Local dimension trueness: {camera_verification.local_dimension_trueness() * 100:.3f}%")


if __name__ == "__main__":
    _main()
