"""Auto generated, do not edit."""

from __future__ import annotations

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime

import _zivid


class FrameInfo:

    class Diagnostics:

        def __init__(
            self,
            packet_loss: bool = _zivid.FrameInfo.Diagnostics.PacketLoss().value,
        ):

            if isinstance(packet_loss, (bool,)):
                self._packet_loss = _zivid.FrameInfo.Diagnostics.PacketLoss(packet_loss)
            else:
                raise TypeError(
                    "Unsupported type, expected: (bool,), got {value_type}".format(value_type=type(packet_loss))
                )

        @property
        def packet_loss(self):
            return self._packet_loss.value

        @packet_loss.setter
        def packet_loss(self, value):
            if isinstance(value, (bool,)):
                self._packet_loss = _zivid.FrameInfo.Diagnostics.PacketLoss(value)
            else:
                raise TypeError("Unsupported type, expected: bool, got {value_type}".format(value_type=type(value)))

        def __eq__(self, other):
            if self._packet_loss == other._packet_loss:
                return True
            return False

        def __str__(self):
            return str(_to_internal_frame_info_diagnostics(self))

    class Metrics:

        def __init__(
            self,
            acquisition_time: datetime.timedelta = _zivid.FrameInfo.Metrics.AcquisitionTime().value,
            capture_time: datetime.timedelta = _zivid.FrameInfo.Metrics.CaptureTime().value,
            kernel_compute_time: datetime.timedelta = _zivid.FrameInfo.Metrics.KernelComputeTime().value,
            reprocessing_time: datetime.timedelta | None = _zivid.FrameInfo.Metrics.ReprocessingTime().value,
            throttling_time: datetime.timedelta = _zivid.FrameInfo.Metrics.ThrottlingTime().value,
        ):

            if isinstance(acquisition_time, (datetime.timedelta,)):
                self._acquisition_time = _zivid.FrameInfo.Metrics.AcquisitionTime(acquisition_time)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                        value_type=type(acquisition_time)
                    )
                )

            if isinstance(capture_time, (datetime.timedelta,)):
                self._capture_time = _zivid.FrameInfo.Metrics.CaptureTime(capture_time)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                        value_type=type(capture_time)
                    )
                )

            if isinstance(kernel_compute_time, (datetime.timedelta,)):
                self._kernel_compute_time = _zivid.FrameInfo.Metrics.KernelComputeTime(kernel_compute_time)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                        value_type=type(kernel_compute_time)
                    )
                )

            if isinstance(reprocessing_time, (datetime.timedelta,)) or reprocessing_time is None:
                self._reprocessing_time = _zivid.FrameInfo.Metrics.ReprocessingTime(reprocessing_time)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,) or None, got {value_type}".format(
                        value_type=type(reprocessing_time)
                    )
                )

            if isinstance(throttling_time, (datetime.timedelta,)):
                self._throttling_time = _zivid.FrameInfo.Metrics.ThrottlingTime(throttling_time)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                        value_type=type(throttling_time)
                    )
                )

        @property
        def acquisition_time(self):
            return self._acquisition_time.value

        @property
        def capture_time(self):
            return self._capture_time.value

        @property
        def kernel_compute_time(self):
            return self._kernel_compute_time.value

        @property
        def reprocessing_time(self):
            return self._reprocessing_time.value

        @property
        def throttling_time(self):
            return self._throttling_time.value

        @acquisition_time.setter
        def acquisition_time(self, value):
            if isinstance(value, (datetime.timedelta,)):
                self._acquisition_time = _zivid.FrameInfo.Metrics.AcquisitionTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta, got {value_type}".format(value_type=type(value))
                )

        @capture_time.setter
        def capture_time(self, value):
            if isinstance(value, (datetime.timedelta,)):
                self._capture_time = _zivid.FrameInfo.Metrics.CaptureTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta, got {value_type}".format(value_type=type(value))
                )

        @kernel_compute_time.setter
        def kernel_compute_time(self, value):
            if isinstance(value, (datetime.timedelta,)):
                self._kernel_compute_time = _zivid.FrameInfo.Metrics.KernelComputeTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta, got {value_type}".format(value_type=type(value))
                )

        @reprocessing_time.setter
        def reprocessing_time(self, value):
            if isinstance(value, (datetime.timedelta,)) or value is None:
                self._reprocessing_time = _zivid.FrameInfo.Metrics.ReprocessingTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @throttling_time.setter
        def throttling_time(self, value):
            if isinstance(value, (datetime.timedelta,)):
                self._throttling_time = _zivid.FrameInfo.Metrics.ThrottlingTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if (
                self._acquisition_time == other._acquisition_time
                and self._capture_time == other._capture_time
                and self._kernel_compute_time == other._kernel_compute_time
                and self._reprocessing_time == other._reprocessing_time
                and self._throttling_time == other._throttling_time
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_frame_info_metrics(self))

    class SoftwareVersion:

        def __init__(
            self,
            core: str = _zivid.FrameInfo.SoftwareVersion.Core().value,
        ):

            if isinstance(core, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(core)
            else:
                raise TypeError("Unsupported type, expected: (str,), got {value_type}".format(value_type=type(core)))

        @property
        def core(self):
            return self._core.value

        @core.setter
        def core(self, value):
            if isinstance(value, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

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
                model: str = _zivid.FrameInfo.SystemInfo.CPU.Model().value,
            ):

                if isinstance(model, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.CPU.Model(model)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(model))
                    )

            @property
            def model(self):
                return self._model.value

            @model.setter
            def model(self, value):
                if isinstance(value, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.CPU.Model(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            def __eq__(self, other):
                if self._model == other._model:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_frame_info_system_info_cpu(self))

        class ComputeDevice:

            def __init__(
                self,
                model: str = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model().value,
                vendor: str = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor().value,
            ):

                if isinstance(model, (str,)):
                    self._model = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model(model)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(model))
                    )

                if isinstance(vendor, (str,)):
                    self._vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(vendor)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(vendor))
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
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            @vendor.setter
            def vendor(self, value):
                if isinstance(value, (str,)):
                    self._vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            def __eq__(self, other):
                if self._model == other._model and self._vendor == other._vendor:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_frame_info_system_info_compute_device(self))

        def __init__(
            self,
            operating_system: str = _zivid.FrameInfo.SystemInfo.OperatingSystem().value,
            cpu: FrameInfo.SystemInfo.CPU | None = None,
            compute_device: FrameInfo.SystemInfo.ComputeDevice | None = None,
        ):

            if isinstance(operating_system, (str,)):
                self._operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(operating_system)
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(operating_system))
                )

            if cpu is None:
                cpu = self.CPU()
            if not isinstance(cpu, self.CPU):
                raise TypeError("Unsupported type: {value}".format(value=type(cpu)))
            self._cpu = cpu

            if compute_device is None:
                compute_device = self.ComputeDevice()
            if not isinstance(compute_device, self.ComputeDevice):
                raise TypeError("Unsupported type: {value}".format(value=type(compute_device)))
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
                self._operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

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
        time_stamp: datetime.datetime = _zivid.FrameInfo.TimeStamp().value,
        diagnostics: FrameInfo.Diagnostics | None = None,
        metrics: FrameInfo.Metrics | None = None,
        software_version: FrameInfo.SoftwareVersion | None = None,
        system_info: FrameInfo.SystemInfo | None = None,
    ):

        if isinstance(time_stamp, (datetime.datetime,)):
            self._time_stamp = _zivid.FrameInfo.TimeStamp(time_stamp)
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.datetime,), got {value_type}".format(value_type=type(time_stamp))
            )

        if diagnostics is None:
            diagnostics = self.Diagnostics()
        if not isinstance(diagnostics, self.Diagnostics):
            raise TypeError("Unsupported type: {value}".format(value=type(diagnostics)))
        self._diagnostics = diagnostics

        if metrics is None:
            metrics = self.Metrics()
        if not isinstance(metrics, self.Metrics):
            raise TypeError("Unsupported type: {value}".format(value=type(metrics)))
        self._metrics = metrics

        if software_version is None:
            software_version = self.SoftwareVersion()
        if not isinstance(software_version, self.SoftwareVersion):
            raise TypeError("Unsupported type: {value}".format(value=type(software_version)))
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
    def diagnostics(self):
        return self._diagnostics

    @property
    def metrics(self):
        return self._metrics

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
                "Unsupported type, expected: datetime.datetime, got {value_type}".format(value_type=type(value))
            )

    @diagnostics.setter
    def diagnostics(self, value):
        if not isinstance(value, self.Diagnostics):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._diagnostics = value

    @metrics.setter
    def metrics(self, value):
        if not isinstance(value, self.Metrics):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._metrics = value

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

    @classmethod
    def from_serialized(cls, value):
        return _to_frame_info(_zivid.FrameInfo.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_frame_info(self).serialize()

    def __eq__(self, other):
        if (
            self._time_stamp == other._time_stamp
            and self._diagnostics == other._diagnostics
            and self._metrics == other._metrics
            and self._software_version == other._software_version
            and self._system_info == other._system_info
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_frame_info(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_frame_info(_to_internal_frame_info(self))


def _to_frame_info_diagnostics(internal_diagnostics):
    return FrameInfo.Diagnostics(
        packet_loss=internal_diagnostics.packet_loss.value,
    )


def _to_frame_info_metrics(internal_metrics):
    return FrameInfo.Metrics(
        acquisition_time=internal_metrics.acquisition_time.value,
        capture_time=internal_metrics.capture_time.value,
        kernel_compute_time=internal_metrics.kernel_compute_time.value,
        reprocessing_time=internal_metrics.reprocessing_time.value,
        throttling_time=internal_metrics.throttling_time.value,
    )


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
        compute_device=_to_frame_info_system_info_compute_device(internal_system_info.compute_device),
        operating_system=internal_system_info.operating_system.value,
    )


def _to_frame_info(internal_frame_info):
    return FrameInfo(
        diagnostics=_to_frame_info_diagnostics(internal_frame_info.diagnostics),
        metrics=_to_frame_info_metrics(internal_frame_info.metrics),
        software_version=_to_frame_info_software_version(internal_frame_info.software_version),
        system_info=_to_frame_info_system_info(internal_frame_info.system_info),
        time_stamp=internal_frame_info.time_stamp.value,
    )


def _to_internal_frame_info_diagnostics(diagnostics):
    internal_diagnostics = _zivid.FrameInfo.Diagnostics()

    internal_diagnostics.packet_loss = _zivid.FrameInfo.Diagnostics.PacketLoss(diagnostics.packet_loss)

    return internal_diagnostics


def _to_internal_frame_info_metrics(metrics):
    internal_metrics = _zivid.FrameInfo.Metrics()

    internal_metrics.acquisition_time = _zivid.FrameInfo.Metrics.AcquisitionTime(metrics.acquisition_time)
    internal_metrics.capture_time = _zivid.FrameInfo.Metrics.CaptureTime(metrics.capture_time)
    internal_metrics.kernel_compute_time = _zivid.FrameInfo.Metrics.KernelComputeTime(metrics.kernel_compute_time)
    internal_metrics.reprocessing_time = _zivid.FrameInfo.Metrics.ReprocessingTime(metrics.reprocessing_time)
    internal_metrics.throttling_time = _zivid.FrameInfo.Metrics.ThrottlingTime(metrics.throttling_time)

    return internal_metrics


def _to_internal_frame_info_software_version(software_version):
    internal_software_version = _zivid.FrameInfo.SoftwareVersion()

    internal_software_version.core = _zivid.FrameInfo.SoftwareVersion.Core(software_version.core)

    return internal_software_version


def _to_internal_frame_info_system_info_cpu(cpu):
    internal_cpu = _zivid.FrameInfo.SystemInfo.CPU()

    internal_cpu.model = _zivid.FrameInfo.SystemInfo.CPU.Model(cpu.model)

    return internal_cpu


def _to_internal_frame_info_system_info_compute_device(compute_device):
    internal_compute_device = _zivid.FrameInfo.SystemInfo.ComputeDevice()

    internal_compute_device.model = _zivid.FrameInfo.SystemInfo.ComputeDevice.Model(compute_device.model)
    internal_compute_device.vendor = _zivid.FrameInfo.SystemInfo.ComputeDevice.Vendor(compute_device.vendor)

    return internal_compute_device


def _to_internal_frame_info_system_info(system_info):
    internal_system_info = _zivid.FrameInfo.SystemInfo()

    internal_system_info.operating_system = _zivid.FrameInfo.SystemInfo.OperatingSystem(system_info.operating_system)

    internal_system_info.cpu = _to_internal_frame_info_system_info_cpu(system_info.cpu)
    internal_system_info.compute_device = _to_internal_frame_info_system_info_compute_device(system_info.compute_device)
    return internal_system_info


def _to_internal_frame_info(frame_info):
    internal_frame_info = _zivid.FrameInfo()

    internal_frame_info.time_stamp = _zivid.FrameInfo.TimeStamp(frame_info.time_stamp)

    internal_frame_info.diagnostics = _to_internal_frame_info_diagnostics(frame_info.diagnostics)
    internal_frame_info.metrics = _to_internal_frame_info_metrics(frame_info.metrics)
    internal_frame_info.software_version = _to_internal_frame_info_software_version(frame_info.software_version)
    internal_frame_info.system_info = _to_internal_frame_info_system_info(frame_info.system_info)
    return internal_frame_info
