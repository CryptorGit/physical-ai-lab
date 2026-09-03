"""Four-foot mapping and contact trace helpers."""

from __future__ import annotations

LEG_PREFIXES = {
    "front-left": "FL",
    "front-right": "FR",
    "rear-left": "RL",
    "rear-right": "RR",
}


def resolve_foot_mapping(robot, sensor) -> list[dict]:
    mapping = []
    used = set()
    for anatomical, prefix in LEG_PREFIXES.items():
        body_ids, names = robot.find_bodies(f"{prefix}_foot")
        if len(body_ids) != 1 or len(names) != 1:
            raise RuntimeError(f"unambiguous foot mapping unavailable for {anatomical}: {names}")
        body_name = names[0]
        if body_name not in sensor.body_names:
            raise RuntimeError(f"foot {body_name} absent from contact sensor")
        sensor_id = sensor.body_names.index(body_name)
        if sensor_id in used:
            raise RuntimeError("duplicate contact sensor index")
        used.add(sensor_id)
        mapping.append(
            {
                "anatomical_name": anatomical,
                "asset_body_name": body_name,
                "robot_body_index": int(body_ids[0]),
                "contact_sensor_index": int(sensor_id),
                "force_tensor_index": int(sensor_id),
                "air_time_field": "contact_forces.data.current_air_time[:, contact_sensor_index]",
                "foot_position_field": "robot.data.body_pos_w[:, robot_body_index, :]",
                "foot_velocity_field": "robot.data.body_lin_vel_w[:, robot_body_index, :]",
            }
        )
    return mapping

