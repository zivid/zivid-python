"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class SceneConditions:

    class AmbientLight:

        class FlickerClassification:

            grid50hz = "grid50hz"
            grid60hz = "grid60hz"
            noFlicker = "noFlicker"
            unknownFlicker = "unknownFlicker"

            _valid_values = {
                "grid50hz": _zivid.SceneConditions.AmbientLight.FlickerClassification.grid50hz,
                "grid60hz": _zivid.SceneConditions.AmbientLight.FlickerClassification.grid60hz,
                "noFlicker": _zivid.SceneConditions.AmbientLight.FlickerClassification.noFlicker,
                "unknownFlicker": _zivid.SceneConditions.AmbientLight.FlickerClassification.unknownFlicker,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            flicker_classification=_zivid.SceneConditions.AmbientLight.FlickerClassification().value,
        ):

            if isinstance(
                flicker_classification,
                _zivid.SceneConditions.AmbientLight.FlickerClassification.enum,
            ):
                self._flicker_classification = (
                    _zivid.SceneConditions.AmbientLight.FlickerClassification(
                        flicker_classification
                    )
                )
            elif isinstance(flicker_classification, str):
                self._flicker_classification = (
                    _zivid.SceneConditions.AmbientLight.FlickerClassification(
                        self.FlickerClassification._valid_values[flicker_classification]
                    )
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(flicker_classification)
                    )
                )

        @property
        def flicker_classification(self):
            if self._flicker_classification.value is None:
                return None
            for key, internal_value in self.FlickerClassification._valid_values.items():
                if internal_value == self._flicker_classification.value:
                    return key
            raise ValueError(
                "Unsupported value {value}".format(value=self._flicker_classification)
            )

        @flicker_classification.setter
        def flicker_classification(self, value):
            if isinstance(value, str):
                self._flicker_classification = (
                    _zivid.SceneConditions.AmbientLight.FlickerClassification(
                        self.FlickerClassification._valid_values[value]
                    )
                )
            elif isinstance(
                value, _zivid.SceneConditions.AmbientLight.FlickerClassification.enum
            ):
                self._flicker_classification = (
                    _zivid.SceneConditions.AmbientLight.FlickerClassification(value)
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._flicker_classification == other._flicker_classification:
                return True
            return False

        def __str__(self):
            return str(_to_internal_scene_conditions_ambient_light(self))

    def __init__(
        self,
        ambient_light=None,
    ):

        if ambient_light is None:
            ambient_light = self.AmbientLight()
        if not isinstance(ambient_light, self.AmbientLight):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(ambient_light))
            )
        self._ambient_light = ambient_light

    @property
    def ambient_light(self):
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, value):
        if not isinstance(value, self.AmbientLight):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._ambient_light = value

    @classmethod
    def load(cls, file_name):
        return _to_scene_conditions(_zivid.SceneConditions(str(file_name)))

    def save(self, file_name):
        _to_internal_scene_conditions(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_scene_conditions(_zivid.SceneConditions.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_scene_conditions(self).serialize()

    def __eq__(self, other):
        if self._ambient_light == other._ambient_light:
            return True
        return False

    def __str__(self):
        return str(_to_internal_scene_conditions(self))


def _to_scene_conditions_ambient_light(internal_ambient_light):
    return SceneConditions.AmbientLight(
        flicker_classification=internal_ambient_light.flicker_classification.value,
    )


def _to_scene_conditions(internal_scene_conditions):
    return SceneConditions(
        ambient_light=_to_scene_conditions_ambient_light(
            internal_scene_conditions.ambient_light
        ),
    )


def _to_internal_scene_conditions_ambient_light(ambient_light):
    internal_ambient_light = _zivid.SceneConditions.AmbientLight()

    internal_ambient_light.flicker_classification = (
        _zivid.SceneConditions.AmbientLight.FlickerClassification(
            ambient_light._flicker_classification.value
        )
    )

    return internal_ambient_light


def _to_internal_scene_conditions(scene_conditions):
    internal_scene_conditions = _zivid.SceneConditions()

    internal_scene_conditions.ambient_light = (
        _to_internal_scene_conditions_ambient_light(scene_conditions.ambient_light)
    )
    return internal_scene_conditions
