"""Sample demonstrating in-field correction (experimental)"""
import zivid

from zivid.experimental.calibration import (
    detect_feature_points,
    InfieldCorrectionInput,
    compute_camera_correction,
    write_camera_correction,
)


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    infield_inputs = []

    try:
        while True:
            detection_result = detect_feature_points(camera)
            infield_input = InfieldCorrectionInput(detection_result)
            if infield_input.valid():
                print("Measurement OK. Appending to dataset.")
                infield_inputs.append(infield_input)
            else:
                print(
                    f"Warning: Measurement invalid [{infield_input.status_description()}]. Please try again."
                )

            input(
                "Hit [Enter] to continue, or [Ctrl+C] to end capture loop and calculate correction."
            )
    except KeyboardInterrupt:
        print("\nCapture loop ended.")

    correction = compute_camera_correction(infield_inputs)
    print("Successfully calculated camera correction")
    print(correction.accuracy_estimate())
    print(correction.accuracy_estimate().dimension_accuracy())

    answer = input("Write correction to camera? (y/n)")
    if answer == "y":
        print("Writing correction to camera")
        write_camera_correction(camera, correction)
    print("Finished")


if __name__ == "__main__":
    _main()
