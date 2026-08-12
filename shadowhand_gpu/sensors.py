"""Compiled sensor layout and tactile extraction metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TOUCH_NAME_TOKEN = "robot0:TS_"


@dataclass(frozen=True)
class SensorSlice:
    sensor_id: int
    name: str
    sensor_type: int
    object_type: int
    object_id: int
    data_address: int
    dimension: int
    is_touch: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensorLayout:
    nsensor: int
    nsensordata: int
    sensors: tuple[SensorSlice, ...]
    touch_sensor_ids: tuple[int, ...]
    touch_data_indices: tuple[int, ...]
    touch_names: tuple[str, ...]

    @property
    def touch_count(self) -> int:
        return len(self.touch_data_indices)

    @property
    def contiguous_touch_span(self) -> tuple[int, int] | None:
        if not self.touch_data_indices:
            return None
        start = self.touch_data_indices[0]
        expected = tuple(range(start, start + len(self.touch_data_indices)))
        if self.touch_data_indices == expected:
            return start, start + len(self.touch_data_indices)
        return None

    def to_dict(
        self,
        *,
        include_sensors: bool = True,
        include_touch_entries: bool = True,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "nsensor": self.nsensor,
            "nsensordata": self.nsensordata,
            "touch_count": self.touch_count,
            "contiguous_touch_span": self.contiguous_touch_span,
        }
        if include_touch_entries:
            result.update(
                {
                    "touch_sensor_ids": list(self.touch_sensor_ids),
                    "touch_data_indices": list(self.touch_data_indices),
                    "touch_names": list(self.touch_names),
                }
            )
        if include_sensors:
            result["sensors"] = [sensor.to_dict() for sensor in self.sensors]
        return result


def build_sensor_layout(model: Any, *, touch_name_token: str = TOUCH_NAME_TOKEN) -> SensorLayout:
    """Build sensor slices in compiled MuJoCo order.

    The legacy environment identifies tactile sensors by the ``robot0:TS_`` name
    token. This function intentionally preserves that selection rule while using
    ``sensor_adr`` and ``sensor_dim`` for data extraction.
    """
    import mujoco

    sensors: list[SensorSlice] = []
    touch_sensor_ids: list[int] = []
    touch_data_indices: list[int] = []
    touch_names: list[str] = []

    for sensor_id in range(int(model.nsensor)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or ""
        data_address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        is_touch = touch_name_token in name
        sensor = SensorSlice(
            sensor_id=sensor_id,
            name=name,
            sensor_type=int(model.sensor_type[sensor_id]),
            object_type=int(model.sensor_objtype[sensor_id]),
            object_id=int(model.sensor_objid[sensor_id]),
            data_address=data_address,
            dimension=dimension,
            is_touch=is_touch,
        )
        sensors.append(sensor)
        if is_touch:
            touch_sensor_ids.append(sensor_id)
            touch_names.append(name)
            touch_data_indices.extend(range(data_address, data_address + dimension))

    if touch_data_indices and max(touch_data_indices) >= int(model.nsensordata):
        raise ValueError("Compiled touch sensor slice exceeds model.nsensordata")
    if len(touch_data_indices) != len(set(touch_data_indices)):
        raise ValueError("Compiled touch sensor slices overlap")

    return SensorLayout(
        nsensor=int(model.nsensor),
        nsensordata=int(model.nsensordata),
        sensors=tuple(sensors),
        touch_sensor_ids=tuple(touch_sensor_ids),
        touch_data_indices=tuple(touch_data_indices),
        touch_names=tuple(touch_names),
    )
