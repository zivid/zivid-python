import operator

import pytest
import zivid


def test_compute_device(application):
    compute_device = application.compute_device()
    assert isinstance(compute_device, zivid.ComputeDevice)
    assert isinstance(compute_device.backend, zivid.ComputeBackend)
    assert compute_device.model
    assert compute_device.vendor
    assert str(compute_device)


def test_compute_device_memory_info(application):
    compute_device = application.compute_device()
    memory_info = compute_device.memory_info
    if compute_device.backend == zivid.ComputeBackend.cuda:
        assert isinstance(memory_info, zivid.ComputeDevice.MemoryInfo)
        assert memory_info.total_bytes > 0
        assert 0 <= memory_info.free_bytes <= memory_info.total_bytes
    else:
        assert memory_info is None


def test_compute_device_illegal_init():
    with pytest.raises(TypeError):
        zivid.ComputeDevice("this should fail")


def test_sdk_stream_or_queue_matches_native_stream_handle(application):
    compute_device = application.compute_device()
    stream_or_queue = compute_device.sdk_stream_or_queue()
    assert isinstance(stream_or_queue, zivid.StreamOrQueue)
    if compute_device.backend == zivid.ComputeBackend.cuda:
        assert stream_or_queue.stream.stream == compute_device.native_stream_handle()
        assert stream_or_queue.command_queue.command_queue == 0
    else:
        assert stream_or_queue.command_queue.command_queue == compute_device.native_stream_handle()
        assert stream_or_queue.stream.stream == 0
    zivid.synchronize_stream(stream_or_queue)


def test_handles_default_to_null():
    assert zivid.CUDAContextPtr().context == 0
    assert zivid.CUDAStreamPtr().stream == 0
    assert zivid.OpenCLCommandQueuePtr().command_queue == 0


def test_handles_hold_value():
    assert zivid.CUDAContextPtr(1234).context == 1234
    assert zivid.CUDAStreamPtr(1234).stream == 1234
    assert zivid.OpenCLCommandQueuePtr(1234).command_queue == 1234


def test_handles_are_settable():
    cuda_stream = zivid.CUDAStreamPtr()
    cuda_stream.stream = 42
    assert cuda_stream.stream == 42


def test_stream_or_queue_from_cuda_stream():
    for stream_or_queue in (
        zivid.StreamOrQueue(zivid.CUDAStreamPtr(1234)),
        zivid.StreamOrQueue(stream=zivid.CUDAStreamPtr(1234)),
    ):
        assert isinstance(stream_or_queue.stream, zivid.CUDAStreamPtr)
        assert stream_or_queue.stream.stream == 1234
        assert stream_or_queue.command_queue.command_queue == 0


def test_stream_or_queue_from_command_queue():
    for stream_or_queue in (
        zivid.StreamOrQueue(zivid.OpenCLCommandQueuePtr(1234)),
        zivid.StreamOrQueue(command_queue=zivid.OpenCLCommandQueuePtr(1234)),
    ):
        assert isinstance(stream_or_queue.command_queue, zivid.OpenCLCommandQueuePtr)
        assert stream_or_queue.command_queue.command_queue == 1234
        assert stream_or_queue.stream.stream == 0


def test_stream_or_queue_requires_exactly_one_argument():
    with pytest.raises(TypeError):
        zivid.StreamOrQueue()
    with pytest.raises(TypeError):
        zivid.StreamOrQueue(stream=zivid.CUDAStreamPtr(), command_queue=zivid.OpenCLCommandQueuePtr())
    with pytest.raises(TypeError):
        zivid.StreamOrQueue(zivid.CUDAStreamPtr(), stream=zivid.CUDAStreamPtr())


def test_stream_or_queue_rejects_wrong_types():
    with pytest.raises(TypeError):
        zivid.StreamOrQueue(stream=zivid.OpenCLCommandQueuePtr())
    with pytest.raises(TypeError):
        zivid.StreamOrQueue(command_queue=zivid.CUDAStreamPtr())
    with pytest.raises(TypeError):
        zivid.StreamOrQueue(1234)


def test_enum_values_match_the_backend():
    assert zivid.FrameFileType.frame.value == 0
    assert zivid.FrameFileType.frame_2d.value == 1
    assert zivid.ComputeBackend.cuda.value == 0
    assert zivid.ComputeBackend.opencl.value == 1


def test_enums_convert_to_their_value():
    assert int(zivid.FrameFileType.frame_2d) == 1
    assert operator.index(zivid.ComputeBackend.opencl) == 1


@pytest.fixture(name="torch_cuda")
def torch_cuda_fixture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Test requires PyTorch with CUDA")
    return torch


@pytest.mark.parametrize(
    "factory, channels", [("create_device_array_view_rgba", 4), ("create_device_array_view_bgr", 3)]
)
def test_compute_device_creates_device_array_views(application, torch_cuda, factory, channels):
    compute_device = application.compute_device()
    if compute_device.backend != zivid.ComputeBackend.cuda:
        pytest.skip("Test requires the CUDA backend")
    height, width = 4, 6
    tensor = torch_cuda.zeros((height, width, channels), dtype=torch_cuda.uint8, device="cuda")
    torch_cuda.cuda.synchronize()
    torch_stream = torch_cuda.cuda.Stream()
    stream = zivid.CUDAStreamPtr(torch_stream.cuda_stream)
    view = getattr(compute_device, factory)(tensor.data_ptr(), width, height, stream)
    assert isinstance(view, zivid.DeviceArrayView)
    assert list(view.shape[:2]) == [height, width]


def test_compute_device_rejects_wrong_stream_type(application):
    with pytest.raises(TypeError):
        application.compute_device().create_device_array_view_rgba(0, 1, 1, "not a stream")
