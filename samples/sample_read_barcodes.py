import zivid
from zivid.experimental.toolbox.barcode import BarcodeDetector, LinearBarcodeFormat, MatrixBarcodeFormat


def _main():
    app = zivid.Application()

    linear_barcode_formats = {
        LinearBarcodeFormat.code128,
        LinearBarcodeFormat.code93,
        LinearBarcodeFormat.code39,
        LinearBarcodeFormat.ean13,
        LinearBarcodeFormat.ean8,
        LinearBarcodeFormat.upcA,
        LinearBarcodeFormat.upcE,
    }

    matrix_barcode_formats = {
        MatrixBarcodeFormat.qrcode,
        MatrixBarcodeFormat.dataMatrix,
    }

    with app.connect_camera() as camera, BarcodeDetector() as detector:
        settings2d = detector.suggest_settings(camera)

        while True:
            print("-" * 70)
            print("Capturing frame. Press Ctrl-C to stop.")
            with camera.capture(settings2d) as frame2d:

                # Read linear codes
                all_results_linear = detector.read_linear_codes(
                    frame2d,
                    format_filter=linear_barcode_formats,
                )

                # Read matrix codes
                all_results_matrix = detector.read_matrix_codes(
                    frame2d,
                    format_filter=matrix_barcode_formats,
                )

                # Print results
                print(f"Found {len(all_results_linear)} linear codes")
                for result in all_results_linear:
                    print(f"  -{result}")
                print(f"Found {len(all_results_matrix)} matrix codes")
                for result in all_results_matrix:
                    print(f"  -{result}")


if __name__ == "__main__":
    _main()
