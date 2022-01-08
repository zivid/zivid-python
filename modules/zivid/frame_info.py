"""Auto generated, do not edit."""
# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches,too-many-boolean-expressions
import datetime
import _zivid


class FrameInfo:
    class SoftwareVersion:
        def __init__(
            self,
            core=_zivid.FrameInfo.SoftwareVersion.Core().value,
        ):

            if isinstance(core, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(core)
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(
                        value_type=type(core)
                    )
                )

        @property
        def core(self):
            return self._core.value

        @core.setter
        def core(self, value):
            if isinstance(value, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._core == other._core:
                return True
            return False

        def __str__(self):
            return str(_to_internal_frame_info_software_version(self))

    class SystemInfo:
        class CPU:
            def __init__(
                self,
                model=_zivid.FrameInfo.SystemInfo.CPU.Model().value,
            ):

                if isinstance(model, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.CPU.Model(model)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(
                            value_type=type(model)
                        )
                    )

            @property
            def model(self):
                return self._model.value

            @model.setter
            def model(self, value):
                if isinstance(value, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.CPU.Model(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: str, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._model == other._model:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_frame_info_system_info_cpu(self))

        class ComputeDevice:
            def __init__(
                self,
                model=_zivid.FrameInfo.SystemInfo.ComputeDevice.Model().value,
                vendor=_zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor().value,
            ):

                if isinstance(model, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model(model)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(
                            value_type=type(model)
                        )
                    )

                if isinstance(vendor, (str,)):
                    self._vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(
                        vendor
                    )
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(
                            value_type=type(vendor)
                        )
                    )

            @property
            def model(self):
                return self._model.value

            @property
            def vendor(self):
                return self._vendor.value

            @model.setter
            def model(self, value):
                if isinstance(value, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: str, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @vendor.setter
            def vendor(self, value):
                if isinstance(value, (str,)):
                    self._vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(
                        value
                    )
                else:
                    raise TypeError(
                        "Unsupported type, expected: str, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._model == other._model and self._vendor == other._vendor:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_frame_info_system_info_compute_device(self))

        def __init__(
            self,
            operating_system=_zivid.FrameInfo.SystemInfo.OperatingSystem().value,
            cpu=None,
            compute_device=None,
        ):

            if isinstance(operating_system, (str,)):
                self._operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(
                    operating_system
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(
                        value_type=type(operating_system)
                    )
                )

            if cpu is None:
                cpu = self.CPU()
            if not isinstance(cpu, self.CPU):
                raise TypeError("Unsupported type: {value}".format(value=type(cpu)))
            self._cpu = cpu

            if compute_device is None:
                compute_device = self.ComputeDevice()
            if not isinstance(compute_device, self.ComputeDevice):
                raise TypeError(
                    "Unsupported type: {value}".format(value=type(compute_device))
                )
            self._compute_device = compute_device

        @property
        def operating_system(self):
            return self._operating_system.value

        @property
        def cpu(self):
            return self._cpu

        @property
        def compute_device(self):
            return self._compute_device

        @operating_system.setter
        def operating_system(self, value):
            if isinstance(value, (str,)):
                self._operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(
                    value
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @cpu.setter
        def cpu(self, value):
            if not isinstance(value, self.CPU):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._cpu = value

        @compute_device.setter
        def compute_device(self, value):
            if not isinstance(value, self.ComputeDevice):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._compute_device = value

        def __eq__(self, other):
            if (
                self._operating_system == other._operating_system
                and self._cpu == other._cpu
                and self._compute_device == other._compute_device
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_frame_info_system_info(self))

    def __init__(
        self,
        time_stamp=_zivid.FrameInfo.TimeStamp().value,
        software_version=None,
        system_info=None,
    ):

        if isinstance(time_stamp, (datetime.datetime,)):
            self._time_stamp = _zivid.FrameInfo.TimeStamp(time_stamp)
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.datetime,), got {value_type}".format(
                    value_type=type(time_stamp)
                )
            )

        if software_version is None:
            software_version = self.SoftwareVersion()
        if not isinstance(software_version, self.SoftwareVersion):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(software_version))
            )
        self._software_version = software_version

        if system_info is None:
            system_info = self.SystemInfo()
        if not isinstance(system_info, self.SystemInfo):
            raise TypeError("Unsupported type: {value}".format(value=type(system_info)))
        self._system_info = system_info

    @property
    def time_stamp(self):
        return self._time_stamp.value

    @property
    def software_version(self):
        return self._software_version

    @property
    def system_info(self):
        return self._system_info

    @time_stamp.setter
    def time_stamp(self, value):
        if isinstance(value, (datetime.datetime,)):
            self._time_stamp = _zivid.FrameInfo.TimeStamp(value)
        else:
            raise TypeError(
                "Unsupported type, expected: datetime.datetime, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @software_version.setter
    def software_version(self, value):
        if not isinstance(value, self.SoftwareVersion):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._software_version = value

    @system_info.setter
    def system_info(self, value):
        if not isinstance(value, self.SystemInfo):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._system_info = value

    @classmethod
    def load(cls, file_name):
        return _to_frame_info(_zivid.FrameInfo(str(file_name)))

    def save(self, file_name):
        _to_internal_frame_info(self).save(str(file_name))

    def __eq__(self, other):
        if (
            self._time_stamp == other._time_stamp
            and self._software_version == other._software_version
            and self._system_info == other._system_info
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_frame_info(self))


def _to_frame_info_software_version(internal_software_version):
    return FrameInfo.SoftwareVersion(
        core=internal_software_version.core.value,
    )


def _to_frame_info_system_info_cpu(internal_cpu):
    return FrameInfo.SystemInfo.CPU(
        model=internal_cpu.model.value,
    )


def _to_frame_info_system_info_compute_device(internal_compute_device):
    return FrameInfo.SystemInfo.ComputeDevice(
        model=internal_compute_device.model.value,
        vendor=internal_compute_device.vendor.value,
    )


def _to_frame_info_system_info(internal_system_info):
    return FrameInfo.SystemInfo(
        cpu=_to_frame_info_system_info_cpu(internal_system_info.cpu),
        compute_device=_to_frame_info_system_info_compute_device(
            internal_system_info.compute_device
        ),
        operating_system=internal_system_info.operating_system.value,
    )


def _to_frame_info(internal_frame_info):
    return FrameInfo(
        software_version=_to_frame_info_software_version(
            internal_frame_info.software_version
        ),
        system_info=_to_frame_info_system_info(internal_frame_info.system_info),
        time_stamp=internal_frame_info.time_stamp.value,
    )


def _to_internal_frame_info_software_version(software_version):
    internal_software_version = _zivid.FrameInfo.SoftwareVersion()

    internal_software_version.core = _zivid.FrameInfo.SoftwareVersion.Core(
        software_version.core
    )

    return internal_software_version


def _to_internal_frame_info_system_info_cpu(cpu):
    internal_cpu = _zivid.FrameInfo.SystemInfo.CPU()

    internal_cpu.model = _zivid.FrameInfo.SystemInfo.CPU.Model(cpu.model)

    return internal_cpu


def _to_internal_frame_info_system_info_compute_device(compute_device):
    internal_compute_device = _zivid.FrameInfo.SystemInfo.ComputeDevice()

    internal_compute_device.model = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model(
        compute_device.model
    )
    internal_compute_device.vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(
        compute_device.vendor
    )

    return internal_compute_device


def _to_internal_frame_info_system_info(system_info):
    internal_system_info = _zivid.FrameInfo.SystemInfo()

    internal_system_info.operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(
        system_info.operating_system
    )

    internal_system_info.cpu = _to_internal_frame_info_system_info_cpu(system_info.cpu)
    internal_system_info.compute_device = (
        _to_internal_frame_info_system_info_compute_device(system_info.compute_device)
    )
    return internal_system_info


def _to_internal_frame_info(frame_info):
    internal_frame_info = _zivid.FrameInfo()

    internal_frame_info.time_stamp = _zivid.FrameInfo.TimeStamp(frame_info.time_stamp)

    internal_frame_info.software_version = _to_internal_frame_info_software_version(
        frame_info.software_version
    )
    internal_frame_info.system_info = _to_internal_frame_info_system_info(
        frame_info.system_info
    )
    return internal_frame_info
