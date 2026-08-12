"""Thin direct MuJoCo Warp backend used for bring-up and raw benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_loader import load_project_model
from .sensors import SensorLayout, build_sensor_layout


@dataclass(frozen=True)
class WarpAllocation:
    worlds: int
    contacts_per_world: int
    total_contact_capacity: int
    constraints_per_world: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class MujocoWarpBackend:
    """Own a native-rigid batched MJWarp model/data pair and PyTorch views.

    This class intentionally contains no Gymnasium task logic yet. It is the Phase-B
    capability layer on which the batched task is built.  Flex models are
    rejected because their contact/tactile dynamics did not pass the matched-state
    parity gate; the production path never installs the experimental workaround.
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        worlds: int,
        device: str = "cuda:0",
        contacts_per_world: int = 8192,
        constraints_per_world: int = 4096,
        use_cuda_graphs: bool = True,
    ) -> None:
        if worlds < 1:
            raise ValueError("worlds must be at least 1")
        if contacts_per_world < 1 or constraints_per_world < 1:
            raise ValueError("contact and constraint capacities must be positive")

        try:
            import mujoco_warp as mjw
            import torch
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "Direct GPU simulation requires requirements-gpu.txt in an isolated environment"
            ) from exc

        self.mjw = mjw
        self.wp = wp
        self.torch = torch
        self.device = device
        self.worlds = int(worlds)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.model, self.model_report = load_project_model(xml_path, reference_compat=True)
        self.sensor_layout: SensorLayout = build_sensor_layout(self.model)
        if int(self.model.nflex) != 0:
            raise NotImplementedError(
                "The production CUDA backend supports compiled rigid geom models, including "
                "native mesh geoms, but not flex collision models; "
                f"{Path(xml_path)} compiled with nflex={int(self.model.nflex)}. "
                f"Detected {self.model_report.object_collision_representation}."
            )
        self.allocation = WarpAllocation(
            worlds=self.worlds,
            contacts_per_world=int(contacts_per_world),
            total_contact_capacity=int(contacts_per_world) * self.worlds,
            constraints_per_world=int(constraints_per_world),
        )

        with wp.ScopedDevice(device):
            self.warp_model = mjw.put_model(self.model)
            self.data = mjw.make_data(
                self.model,
                nworld=self.worlds,
                nconmax=self.allocation.contacts_per_world,
                njmax=self.allocation.constraints_per_world,
            )

        self.torch_device = torch.device(device)
        # CUDA graph capture requires a Warp-owned stream. PyTorch records
        # device-side wait events before and after MJWarp work, preserving zero-copy
        # ordering without a host synchronization.
        self.warp_stream = wp.get_stream(device)
        self.torch_warp_stream = torch.cuda.ExternalStream(
            self.warp_stream.cuda_stream,
            device=self.torch_device,
        )
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.time = wp.to_torch(self.data.time)
        self.qacc_warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.sensordata = wp.to_torch(self.data.sensordata)
        self.active_contact_counts = wp.to_torch(self.data.nacon)
        self.collision_counts = wp.to_torch(self.data.ncollision)
        self.constraint_counts = wp.to_torch(self.data.nefc)
        self.overflow_flags = wp.to_torch(self.data.overflow)
        self._step_graph = None

        touch_span = self.sensor_layout.contiguous_touch_span
        self._touch_span = touch_span
        self._touch_indices = None
        if touch_span is None and self.sensor_layout.touch_data_indices:
            self._touch_indices = torch.tensor(
                self.sensor_layout.touch_data_indices,
                dtype=torch.long,
                device=self.torch_device,
            )

    @property
    def touch(self) -> Any:
        """Return the cached-order tactile view/gather without leaving CUDA."""
        if self._touch_span is not None:
            start, stop = self._touch_span
            return self.sensordata[:, start:stop]
        if self._touch_indices is None:
            return self.sensordata[:, :0]
        return self.sensordata.index_select(1, self._touch_indices)

    def step(self, physics_steps: int = 1) -> None:
        if physics_steps < 1:
            raise ValueError("physics_steps must be at least 1")
        caller_stream = self.torch.cuda.current_stream(self.torch_device)
        self.torch_warp_stream.wait_stream(caller_stream)
        with self.wp.ScopedStream(self.warp_stream):
            if self.use_cuda_graphs:
                if self._step_graph is None:
                    with self.wp.ScopedCapture(
                        device=self.device,
                        stream=self.warp_stream,
                    ) as capture:
                        self.mjw.step(self.warp_model, self.data)
                    self._step_graph = capture.graph
                for _ in range(physics_steps):
                    self.wp.capture_launch(self._step_graph, stream=self.warp_stream)
            else:
                for _ in range(physics_steps):
                    self.mjw.step(self.warp_model, self.data)
        caller_stream.wait_stream(self.torch_warp_stream)

    def forward(self) -> None:
        """Recompute derived dynamics/sensor workspace after external state restore."""
        caller_stream = self.torch.cuda.current_stream(self.torch_device)
        self.torch_warp_stream.wait_stream(caller_stream)
        with self.wp.ScopedStream(self.warp_stream):
            self.mjw.forward(self.warp_model, self.data)
        caller_stream.wait_stream(self.torch_warp_stream)

    def _state_tensor(self, value: Any, *, width: int, name: str) -> Any:
        tensor = self.torch.as_tensor(value, dtype=self.qpos.dtype, device=self.torch_device)
        if tensor.ndim == 1:
            if tensor.shape[0] != width:
                raise ValueError(f"{name} must have width {width}, found {tuple(tensor.shape)}")
            tensor = tensor.unsqueeze(0).expand(self.worlds, -1)
        if tuple(tensor.shape) != (self.worlds, width):
            raise ValueError(
                f"{name} must have shape ({self.worlds}, {width}) or ({width},), "
                f"found {tuple(tensor.shape)}"
            )
        return tensor

    def set_state(
        self,
        *,
        qpos: Any,
        qvel: Any,
        ctrl: Any | None = None,
        qacc_warmstart: Any | None = None,
        time: Any | None = None,
    ) -> None:
        """Copy integration state into every world without a host round trip.

        One-dimensional inputs are broadcast to all worlds. Batched inputs must have
        exactly ``worlds`` rows. The copy is enqueued on the PyTorch stream shared
        with Warp; callers do not need an intermediate synchronization.
        """
        qpos_tensor = self._state_tensor(qpos, width=int(self.model.nq), name="qpos")
        qvel_tensor = self._state_tensor(qvel, width=int(self.model.nv), name="qvel")
        self.qpos.copy_(qpos_tensor)
        self.qvel.copy_(qvel_tensor)
        if ctrl is not None:
            self.ctrl.copy_(self._state_tensor(ctrl, width=int(self.model.nu), name="ctrl"))
        else:
            self.ctrl.zero_()
        if qacc_warmstart is not None:
            self.qacc_warmstart.copy_(
                self._state_tensor(qacc_warmstart, width=int(self.model.nv), name="qacc_warmstart")
            )
        else:
            self.qacc_warmstart.zero_()

        if time is None:
            self.time.zero_()
        else:
            time_tensor = self.torch.as_tensor(time, dtype=self.time.dtype, device=self.torch_device)
            if time_tensor.ndim == 0:
                time_tensor = time_tensor.expand_as(self.time)
            if tuple(time_tensor.shape) != tuple(self.time.shape):
                raise ValueError(
                    f"time must be scalar or have shape {tuple(self.time.shape)}, "
                    f"found {tuple(time_tensor.shape)}"
                )
            self.time.copy_(time_tensor)

    def synchronize(self) -> None:
        self.wp.synchronize_device(self.device)

    def report(self) -> dict[str, Any]:
        warp_device = self.wp.get_device(self.device)
        return {
            "model": self.model_report.to_dict(),
            "allocation": self.allocation.to_dict(),
            "mujoco_warp_version": str(self.mjw.__version__),
            "warp_version": str(self.wp.__version__),
            "device": self.device,
            "device_name": str(warp_device.name),
            "device_total_bytes": int(warp_device.total_memory),
            "device_free_bytes": int(warp_device.free_memory),
            "model_support": self.model_report.gpu_collision_support,
            "cuda_graphs": self.use_cuda_graphs,
            "qpos_shape": tuple(self.qpos.shape),
            "qvel_shape": tuple(self.qvel.shape),
            "ctrl_shape": tuple(self.ctrl.shape),
            "time_shape": tuple(self.time.shape),
            "sensordata_shape": tuple(self.sensordata.shape),
            "active_contact_count_shape": tuple(self.active_contact_counts.shape),
            "collision_count_shape": tuple(self.collision_counts.shape),
            "constraint_count_shape": tuple(self.constraint_counts.shape),
            "overflow_shape": tuple(self.overflow_flags.shape),
            "touch_shape": tuple(self.touch.shape),
            "sensor_layout": self.sensor_layout.to_dict(
                include_sensors=False,
                include_touch_entries=False,
            ),
        }
