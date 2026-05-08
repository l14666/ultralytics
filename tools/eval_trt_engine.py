#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate a TensorRT YOLO engine with Ultralytics detection or pose metrics.

This script runs a TensorRT engine directly through TensorRT Python bindings,
postprocesses raw YOLO outputs with Ultralytics NMS, and reuses Ultralytics
validators for mAP computation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import LOGGER, nms


class ModelStub:
    """Minimal model object required by Ultralytics validators."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = names
        self.end2end = False


class CudaRuntimeBackend:
    """CUDA buffer backend using NVIDIA CUDA Python runtime bindings."""

    def __init__(self, cudart: Any, name: str) -> None:
        self.cudart = cudart
        self.name = name

    @staticmethod
    def _error_code(result: Any) -> int:
        """Extract the CUDA error code from cuda-python return values."""
        return int(result[0] if isinstance(result, tuple) else result)

    def _check(self, result: Any, op: str) -> None:
        """Raise a RuntimeError when a CUDA runtime call fails."""
        err = self._error_code(result)
        if err != 0:
            raise RuntimeError(f"{op} failed with CUDA error code {err}")

    def stream_create(self):
        """Create a CUDA stream."""
        result = self.cudart.cudaStreamCreate()
        self._check(result, "cudaStreamCreate")
        return result[1]

    def stream_destroy(self, stream) -> None:
        """Destroy a CUDA stream."""
        self._check(self.cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def malloc(self, size: int):
        """Allocate device memory."""
        result = self.cudart.cudaMalloc(size)
        self._check(result, "cudaMalloc")
        return result[1]

    def free(self, ptr) -> None:
        """Free device memory."""
        self._check(self.cudart.cudaFree(ptr), "cudaFree")

    def memcpy_h2d_async(self, dst, src: np.ndarray, stream) -> None:
        """Copy host memory to device asynchronously."""
        self._check(
            self.cudart.cudaMemcpyAsync(
                dst,
                src.ctypes.data,
                int(src.nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(input H2D)",
        )

    def memcpy_d2h_async(self, dst: np.ndarray, src, stream) -> None:
        """Copy device memory to host asynchronously."""
        self._check(
            self.cudart.cudaMemcpyAsync(
                dst.ctypes.data,
                src,
                int(dst.nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(output D2H)",
        )

    def stream_synchronize(self, stream) -> None:
        """Synchronize a CUDA stream."""
        self._check(self.cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class PyCudaBackend:
    """CUDA buffer backend using PyCUDA."""

    name = "pycuda.driver"

    def __init__(self) -> None:
        import pycuda.driver as cuda

        cuda.init()
        try:
            self._context = cuda.Context.attach()
            self._autoinit = None
        except cuda.LogicError:
            import pycuda.autoinit as autoinit

            self._context = None
            self._autoinit = autoinit
        self.cuda = cuda

    def stream_create(self):
        """Create a CUDA stream."""
        return self.cuda.Stream()

    def stream_destroy(self, stream) -> None:
        """Destroy a CUDA stream."""
        del stream

    def malloc(self, size: int):
        """Allocate device memory."""
        return self.cuda.mem_alloc(size)

    def free(self, ptr) -> None:
        """Free device memory."""
        ptr.free()

    def memcpy_h2d_async(self, dst, src: np.ndarray, stream) -> None:
        """Copy host memory to device asynchronously."""
        self.cuda.memcpy_htod_async(dst, src, stream)

    def memcpy_d2h_async(self, dst: np.ndarray, src, stream) -> None:
        """Copy device memory to host asynchronously."""
        self.cuda.memcpy_dtoh_async(dst, src, stream)

    def stream_synchronize(self, stream) -> None:
        """Synchronize a CUDA stream."""
        stream.synchronize()


def load_cuda_backend() -> CudaRuntimeBackend | PyCudaBackend:
    """Load a supported CUDA Python backend."""
    errors = []
    try:
        from cuda import cudart

        backend = CudaRuntimeBackend(cudart, "cuda.cudart")
        stream = backend.stream_create()
        backend.stream_destroy(stream)
        return backend
    except Exception as e:
        errors.append(f"cuda.cudart: {e}")
    try:
        from cuda.bindings import runtime as cudart

        backend = CudaRuntimeBackend(cudart, "cuda.bindings.runtime")
        stream = backend.stream_create()
        backend.stream_destroy(stream)
        return backend
    except Exception as e:
        errors.append(f"cuda.bindings.runtime: {e}")
    try:
        backend = PyCudaBackend()
        stream = backend.stream_create()
        backend.stream_destroy(stream)
        return backend
    except Exception as e:
        errors.append(f"pycuda.driver: {e}")
    raise ImportError("TensorRT evaluation requires a CUDA Python backend. Tried: " + "; ".join(errors))


class TrtRunner:
    """Small TensorRT engine runner for a single input and one or more outputs."""

    def __init__(self, engine_path: Path, input_name: str | None = None, output_name: str | None = None) -> None:
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("TensorRT evaluation requires the Python package 'tensorrt'.") from e

        self.trt = trt
        self.cuda = load_cuda_backend()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with engine_path.open("rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.input_names, self.output_names = self._collect_io_names()
        if input_name is not None:
            if input_name not in self.input_names:
                raise ValueError(f"Input tensor '{input_name}' not found. Available inputs: {self.input_names}")
            self.input_name = input_name
        elif len(self.input_names) == 1:
            self.input_name = self.input_names[0]
        else:
            raise ValueError(f"Engine has multiple inputs. Pass --input-name. Available inputs: {self.input_names}")

        if output_name is not None:
            if output_name not in self.output_names:
                raise ValueError(f"Output tensor '{output_name}' not found. Available outputs: {self.output_names}")
            self.output_names = [output_name]

        self.stream = self.cuda.stream_create()

    def _collect_io_names(self) -> tuple[list[str], list[str]]:
        """Collect TensorRT IO tensor names from an engine using TensorRT 10 API."""
        trt = self.trt
        inputs, outputs = [], []
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("This script expects TensorRT 10+ Python bindings with num_io_tensors support.")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if not inputs or not outputs:
            raise RuntimeError(f"Unable to identify engine IO tensors. inputs={inputs}, outputs={outputs}")
        return inputs, outputs

    def close(self) -> None:
        """Destroy the CUDA stream."""
        if getattr(self, "stream", None):
            self.cuda.stream_destroy(self.stream)
            self.stream = None

    def __enter__(self) -> "TrtRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _dtype(self, name: str) -> np.dtype:
        """Return the NumPy dtype for a TensorRT tensor."""
        return np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))

    def _resolved_shape(self, name: str) -> tuple[int, ...]:
        """Return a resolved tensor shape after input shape binding."""
        shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
        if any(x < 0 for x in shape):
            raise RuntimeError(f"Tensor '{name}' still has unresolved shape {shape}.")
        return shape

    def run(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Run one TensorRT inference batch.

        Args:
            images: Normalized BCHW image batch from the Ultralytics dataloader.

        Returns:
            Mapping from output tensor name to NumPy array.
        """
        input_dtype = self._dtype(self.input_name)
        images = np.ascontiguousarray(images.astype(input_dtype, copy=False))

        if not self.context.set_input_shape(self.input_name, tuple(images.shape)):
            raise RuntimeError(f"Failed to set input shape {tuple(images.shape)} for tensor '{self.input_name}'.")

        buffers: list[Any] = []
        output_buffers: dict[str, Any] = {}
        outputs: dict[str, np.ndarray] = {}
        try:
            input_size = int(images.nbytes)
            input_ptr = self.cuda.malloc(input_size)
            buffers.append(input_ptr)
            self.context.set_tensor_address(self.input_name, int(input_ptr))
            self.cuda.memcpy_h2d_async(input_ptr, images, self.stream)

            for name in self.output_names:
                shape = self._resolved_shape(name)
                dtype = self._dtype(name)
                host = np.empty(shape, dtype=dtype)
                ptr = self.cuda.malloc(int(host.nbytes))
                buffers.append(ptr)
                self.context.set_tensor_address(name, int(ptr))
                outputs[name] = host
                output_buffers[name] = ptr

            if not self.context.execute_async_v3(self.stream):
                raise RuntimeError("TensorRT execute_async_v3 failed.")

            for name, host in outputs.items():
                self.cuda.memcpy_d2h_async(host, output_buffers[name], self.stream)
            self.cuda.stream_synchronize(self.stream)
            return outputs
        finally:
            for ptr in buffers:
                self.cuda.free(ptr)


def parse_classes(value: str | None) -> list[int] | None:
    """Parse comma-separated class IDs."""
    if value is None or value.strip() == "":
        return None
    classes = [int(x) for x in value.replace(" ", "").split(",") if x != ""]
    return classes or None


def parse_imgsz(value: str | int | list[int] | tuple[int, ...]) -> int | list[int]:
    """Parse image size from 640, 480,864, or 480x864."""
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 2:
            return [int(value[0]), int(value[1])]
        raise ValueError(f"imgsz must have one or two values, got {value}.")
    parts = [p for p in str(value).lower().replace("x", ",").replace(" ", ",").split(",") if p]
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise ValueError(f"Invalid --imgsz '{value}'. Use 640, 480,864, or 480x864.")


def dataset_imgsz(imgsz: int | list[int]) -> int:
    """Return the scalar pre-resize size required by YOLODataset."""
    return imgsz if isinstance(imgsz, int) else max(imgsz)


def imgsz_shape(imgsz: int | list[int]) -> tuple[int, int]:
    """Return image size as (height, width)."""
    return (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))


def select_output(outputs: dict[str, np.ndarray], output_name: str | None) -> np.ndarray:
    """Select the model output tensor used for YOLO postprocessing."""
    if output_name:
        return outputs[output_name]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    ranked = sorted(outputs.items(), key=lambda item: item[1].size, reverse=True)
    LOGGER.warning(
        f"Multiple outputs found ({list(outputs)}); using largest tensor '{ranked[0][0]}'. "
        "Pass --output-name to choose explicitly."
    )
    return ranked[0][1]


def to_bcn_prediction(
    output: np.ndarray,
    task: str,
    nc: int,
    kpt_shape: list[int] | None,
    has_objectness: bool,
) -> torch.Tensor:
    """Convert raw TensorRT output to the BCN tensor expected by Ultralytics NMS."""
    pred = torch.from_numpy(np.asarray(output)).float()
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if pred.ndim != 3:
        raise ValueError(f"Expected raw output with 2 or 3 dimensions, got shape {tuple(pred.shape)}.")

    if pred.shape[-1] == 6:
        return pred

    extra = int(np.prod(kpt_shape)) if task == "pose" and kpt_shape else 0
    expected = 4 + nc + extra
    expected_with_obj = 5 + nc + extra

    if pred.shape[1] == expected or pred.shape[1] == expected_with_obj:
        bcn = pred
    elif pred.shape[2] == expected or pred.shape[2] == expected_with_obj:
        bcn = pred.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            "Cannot infer YOLO output layout. "
            f"shape={tuple(pred.shape)}, expected channels={expected} or {expected_with_obj}. "
            "Use --nc/--task/--kpt-shape correctly, and add --has-objectness for YOLOv5-style outputs."
        )

    if has_objectness:
        if bcn.shape[1] != expected_with_obj:
            raise ValueError(f"--has-objectness was set, but output channels are {bcn.shape[1]}, not {expected_with_obj}.")
        boxes = bcn[:, :4]
        obj = bcn[:, 4:5]
        cls = bcn[:, 5 : 5 + nc] * obj
        extra_values = bcn[:, 5 + nc :]
        bcn = torch.cat((boxes, cls, extra_values), dim=1)
    elif bcn.shape[1] != expected:
        raise ValueError(
            f"Output has {bcn.shape[1]} channels. Expected {expected}. "
            "If this is xywh+objectness+classes, pass --has-objectness."
        )
    return bcn


def postprocess_batch(
    raw_output: np.ndarray,
    task: str,
    nc: int,
    kpt_shape: list[int] | None,
    conf: float,
    iou: float,
    classes: list[int] | None,
    max_det: int,
    agnostic_nms: bool,
    has_objectness: bool,
) -> list[dict[str, torch.Tensor]]:
    """Run Ultralytics NMS and return validator prediction dictionaries."""
    prediction = to_bcn_prediction(raw_output, task, nc, kpt_shape, has_objectness)
    outputs = nms.non_max_suppression(
        prediction,
        conf_thres=conf,
        iou_thres=iou,
        classes=classes,
        agnostic=agnostic_nms,
        multi_label=True,
        max_det=max_det,
        nc=0 if task == "detect" else nc,
    )
    preds = []
    for x in outputs:
        pred = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]}
        if task == "pose":
            if kpt_shape is None:
                raise ValueError("Pose evaluation requires --kpt-shape or kpt_shape in data YAML.")
            pred["keypoints"] = x[:, 6:].view(-1, *kpt_shape)
        preds.append(pred)
    return preds


def make_validator(args: argparse.Namespace, data: dict[str, Any], save_dir: Path):
    """Create and initialize the appropriate Ultralytics validator."""
    classes = parse_classes(args.classes)
    overrides = {
        "task": args.task,
        "mode": "val",
        "data": str(args.data),
        "imgsz": dataset_imgsz(args.imgsz),
        "batch": args.batch,
        "workers": args.workers,
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "classes": classes,
        "agnostic_nms": args.agnostic_nms,
        "single_cls": False,
        "rect": args.rect,
        "cache": False,
        "fraction": 1.0,
        "half": False,
        "plots": False,
        "visualize": False,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "val": True,
        "project": str(save_dir.parent),
        "name": save_dir.name,
        "exist_ok": True,
    }
    validator_cls = PoseValidator if args.task == "pose" else DetectionValidator
    validator = validator_cls(args=get_cfg(overrides=overrides), save_dir=save_dir)
    validator.device = torch.device("cpu")
    validator.data = data
    validator.stride = args.stride
    validator.training = False
    validator.init_metrics(ModelStub(data["names"]))
    return validator


def build_loader(args: argparse.Namespace, data: dict[str, Any], validator) -> Any:
    """Build an Ultralytics validation dataloader."""
    img_path = data[args.split]
    dataset = build_yolo_dataset(validator.args, img_path, args.batch, data, mode="val", stride=args.stride)
    if not isinstance(args.imgsz, int):
        if args.rect:
            raise ValueError("--rect is not supported with non-square --imgsz because it overrides the fixed shape.")
        dataset.transforms.transforms[0].new_shape = imgsz_shape(args.imgsz)
    return build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False)


def write_results(save_dir: Path, results: dict[str, Any], image_metrics: dict[str, Any] | None) -> None:
    """Write evaluation results to JSON files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if image_metrics is not None:
        with (save_dir / "image_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(image_metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="Path to TensorRT .engine file.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data YAML.")
    parser.add_argument("--task", choices=("detect", "pose"), required=True, help="Evaluation task.")
    parser.add_argument("--imgsz", default="640", help="Validation image size, e.g. 640, 480,864, or 480x864.")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size. Use 1 for static batch engines.")
    parser.add_argument("--split", default="val", help="Dataset split key to evaluate.")
    parser.add_argument("--classes", default=None, help="Comma-separated class IDs, for example '0,2,5'.")
    parser.add_argument("--conf", type=float, default=0.001, help="NMS confidence threshold for mAP evaluation.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--stride", type=int, default=32, help="Model stride used by the validation dataset.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular validation batches.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS.")
    parser.add_argument("--has-objectness", action="store_true", help="Raw output is xywh+objectness+classes(+extra).")
    parser.add_argument("--input-name", default=None, help="TensorRT input tensor name if the engine has multiple inputs.")
    parser.add_argument("--output-name", default=None, help="TensorRT output tensor name to postprocess.")
    parser.add_argument("--kpt-shape", default=None, help="Pose keypoint shape, for example '17,3'. Defaults to data YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of batches to evaluate for smoke tests.")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/trt_eval"), help="Directory for result JSON files.")
    return parser.parse_args()


def main() -> None:
    """Evaluate a TensorRT engine."""
    args = parse_args()
    args.imgsz = parse_imgsz(args.imgsz)
    classes = parse_classes(args.classes)
    data = check_det_dataset(str(args.data), autodownload=False)
    if args.task == "pose":
        if args.kpt_shape:
            data["kpt_shape"] = [int(x) for x in args.kpt_shape.replace(" ", "").split(",")]
        if "kpt_shape" not in data:
            raise ValueError("Pose evaluation requires 'kpt_shape' in data YAML or --kpt-shape.")
    kpt_shape = data.get("kpt_shape") if args.task == "pose" else None
    nc = len(data["names"])

    validator = make_validator(args, data, args.save_dir)
    dataloader = build_loader(args, data, validator)
    validator.dataloader = dataloader

    LOGGER.info(
        f"Evaluating engine={args.engine} task={args.task} imgsz={args.imgsz} batch={args.batch} "
        f"classes={classes if classes is not None else 'all'}"
    )

    preprocess_time = inference_time = postprocess_time = 0.0
    seen_batches = 0
    with TrtRunner(args.engine, input_name=args.input_name, output_name=args.output_name) as runner:
        for batch_i, batch in enumerate(dataloader):
            if args.limit is not None and batch_i >= args.limit:
                break
            t0 = time.perf_counter()
            batch = validator.preprocess(batch)
            images = batch["img"].cpu().numpy()
            t1 = time.perf_counter()
            outputs = runner.run(images)
            raw_output = select_output(outputs, args.output_name)
            t2 = time.perf_counter()
            preds = postprocess_batch(
                raw_output=raw_output,
                task=args.task,
                nc=nc,
                kpt_shape=kpt_shape,
                conf=args.conf,
                iou=args.iou,
                classes=classes,
                max_det=args.max_det,
                agnostic_nms=args.agnostic_nms,
                has_objectness=args.has_objectness,
            )
            validator.update_metrics(preds, batch)
            t3 = time.perf_counter()
            preprocess_time += t1 - t0
            inference_time += t2 - t1
            postprocess_time += t3 - t2
            seen_batches += 1

    stats = validator.get_stats()
    images_seen = max(validator.seen or 0, 1)
    validator.metrics.speed = {
        "preprocess": preprocess_time / images_seen * 1000.0,
        "inference": inference_time / images_seen * 1000.0,
        "loss": 0.0,
        "postprocess": postprocess_time / images_seen * 1000.0,
    }
    results = {
        **validator.metrics.results_dict,
        "speed": validator.metrics.speed,
        "seen_images": int(validator.seen or 0),
        "seen_batches": int(seen_batches),
        "stats_shapes": {k: list(v.shape) for k, v in stats.items()},
    }
    image_metrics = validator.metrics.box.image_metrics
    if args.task == "pose":
        image_metrics = {
            "box": validator.metrics.box.image_metrics,
            "pose": validator.metrics.pose.image_metrics,
        }
    write_results(args.save_dir, results, image_metrics)
    LOGGER.info(json.dumps(results, indent=2))
    LOGGER.info(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
