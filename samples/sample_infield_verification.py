"""Sample demonstrating in-field verification (experimental)"""
import zivid

from zivid.experimental.calibration import (
    detect_feature_points,
    InfieldCorrectionInput,
    verify_camera,
    has_camera_correction,
    camera_correction_timestamp,
)


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    # Check current correction status
    if has_camera_correction(camera):
        timestamp = camera_correction_timestamp(camera)
        print(
            f"Camera currently has a correction. Timestamp: {timestamp.strftime(r'%Y-%m-%d %H:%M:%S')}"
        )

    # Perform verification
    print("Detecting feature points for verification...")
    detection_result = detect_feature_points(camera)
    print(f"Feature point detection: {detection_result}")
    infield_input = InfieldCorrectionInput(detection_result)
    print(f"In-field correction input: {infield_input}")
    if not infield_input.valid():
        raise RuntimeError(
            f"Capture not appropriate for in-field correction/verification. Reason: {infield_input.status_description()}"
        )
    camera_verification = verify_camera(infield_input)
    print(
        f"Local dimension trueness: {camera_verification.local_dimension_trueness()*100:.3f}%"
    )


if __name__ == "__main__":
    _main()
