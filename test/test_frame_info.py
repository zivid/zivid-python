from datetime import datetime
import zivid


def test_time_stamp(frame_info):
    time_stamp = frame_info.time_stamp
    assert time_stamp
    assert isinstance(time_stamp, datetime)


def test_software_version(frame_info):
    software_version = frame_info.software_version
    assert software_version
    assert isinstance(software_version, zivid.frame_info.FrameInfo.SoftwareVersion)
    assert software_version.core
    assert isinstance(software_version.core, str)


def test_system_info(frame_info):
    system_info = frame_info.system_info
    assert isinstance(system_info, zivid.FrameInfo.SystemInfo)

    cpu = system_info.cpu
    assert isinstance(cpu, zivid.FrameInfo.SystemInfo.CPU)
    cpu_model = cpu.model
    assert cpu_model
    assert isinstance(cpu_model, str)

    compute_device = system_info.compute_device
    assert isinstance(compute_device, zivid.FrameInfo.SystemInfo.ComputeDevice)
    compute_model = compute_device.model
    assert compute_model
    assert isinstance(compute_model, str)
    compute_vendor = compute_device.vendor
    assert compute_vendor
    assert isinstance(compute_vendor, str)

    os_name = system_info.operating_system
    assert os_name
    assert isinstance(os_name, str)


def test_set_time_stamp(frame_info):
    assert isinstance(frame_info.time_stamp, datetime)
    assert isinstance(str(frame_info.time_stamp), str)
    new_time_stamp = datetime(1992, 2, 7)
    frame_info.time_stamp = new_time_stamp
    assert isinstance(str(frame_info.time_stamp), str)
    assert isinstance(frame_info.time_stamp, datetime)
    assert frame_info.time_stamp == new_time_stamp
