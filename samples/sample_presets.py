"""Presets sample."""

from zivid import Application
from zivid.presets import categories


def _main():
    app = Application()

    with app.connect_camera() as camera:
        available_categories = categories(camera.info.model)
        print("The following preset categories are available:")
        for index, category in enumerate(available_categories):
            print("    {}: {}".format(index, category.name))
        chosen_category = available_categories[
            int(input("Choose a category (enter a number): "))
        ]

        print(
            "The following presets are available in category '{}':".format(
                chosen_category.name
            )
        )
        for index, preset in enumerate(chosen_category.presets):
            print("    {}: {}".format(index, preset.name))
        chosen_preset = chosen_category.presets[
            int(input("Choose a preset (enter a number): "))
        ]

        print("Capturing point cloud with preset '{}' ...".format(chosen_preset.name))
        with camera.capture_2d_3d(chosen_preset.settings) as frame:
            frame.save("result.zdf")

        settings_file = chosen_preset.name + ".yml"
        print("Saving settings to {}".format(settings_file))
        chosen_preset.settings.save(settings_file)


if __name__ == "__main__":
    _main()
