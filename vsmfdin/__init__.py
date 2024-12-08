from __future__ import annotations

import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .MFDIN_arch import MFDIN_OLD2P

__version__ = "1.0.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
model_path = os.path.join(model_dir, "MFDIN_old_2P.pth")


@torch.inference_mode()
def mfdin(
    clip: vs.VideoNode,
    device_index: int = 0,
    double_rate: bool = False,
    trt: bool = False,
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Multiframe Joint Enhancement for Early Interlaced Videos

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param double_rate:             Output with same frame rate or double frame rate.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("mfdin: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("mfdin: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("mfdin: CUDA is not available")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    pad_w = math.ceil(clip.width / 8) * 8
    pad_h = math.ceil(clip.height / 8) * 8
    padding = (0, pad_w - clip.width, 0, pad_h - clip.height, 0, 0)
    need_pad = any(p > 0 for p in padding)

    if trt:
        import tensorrt
        import torch_tensorrt

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                "MFDIN_old_2P.pth"
                + f"_{pad_w}x{pad_h}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            module = init_module(device, dtype)

            inputs = [torch.zeros(1, 3, 3, pad_h, pad_w, dtype=dtype, device=device)]

            module = torch_tensorrt.compile(
                module,
                "dynamo",
                inputs,
                device=device,
                enabled_precisions={dtype},
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
            )

            torch_tensorrt.save(module, trt_engine_path, output_format="torchscript", inputs=inputs)

        module = torch.jit.load(trt_engine_path).eval()
    else:
        module = init_module(device, dtype)

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        img = torch.stack([frame_to_tensor(f[i], device) for i in range(3)]).unsqueeze(0)

        if need_pad:
            img = F.pad(img, padding, "replicate")

        output = module(img).squeeze(0)

        if need_pad:
            output = output[:, :, : clip.height, : clip.width]

        frame = tensor_to_frame(output[0], f[0].copy(), device)
        if double_rate:
            frame.props["vsmfdin_double_rate_frame"] = tensor_to_frame(output[1], f[0].copy(), device)
        return frame

    clips = [clip.std.DuplicateFrames(0)[:-1], clip, clip.std.DuplicateFrames(clip.num_frames - 1)[1:]]

    output = clip.std.ModifyFrame(clips, inference)
    if double_rate:
        double_rate_frame = output.std.PropToClip("vsmfdin_double_rate_frame")
        output = vs.core.std.Interleave([output, double_rate_frame])
    return output


def init_module(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)

    with torch.device("meta"):
        module = MFDIN_OLD2P()
    module.load_state_dict(state_dict, assign=True)
    return module.eval().to(device, dtype)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, device: torch.device) -> vs.VideoFrame:
    tensor = tensor.detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    torch.cuda.current_stream(device).synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame
