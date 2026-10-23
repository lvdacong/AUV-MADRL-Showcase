from __future__ import annotations

import base64
import json
import math
import secrets
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
import orjson

from .actor_variants import ROLE_DESCRIPTORS, actor_variant, entity_tensor_prefix


def _expected_desired_velocity(
    location: torch.Tensor,
    scale: torch.Tensor,
    stochastic: bool,
    maximum_surge_mps: float,
    maximum_yaw_radps: float,
    action_distribution: str = 'tanh_normal',
) -> list[float]:
    location_values = location.detach().cpu().tolist()
    if action_distribution == 'clipped_normal':
        means = []
        for mean, std in zip(location_values, scale.detach().cpu().tolist(), strict=True):
            if not stochastic:
                means.append(max(-1.0,min(1.0,mean)))
                continue
            lower, upper = (-1-mean)/std, (1-mean)/std
            cdf_lower, cdf_upper = (0.5*math.erfc(-x/math.sqrt(2)) for x in (lower,upper))
            density_lower, density_upper = (math.exp(-0.5*x*x)/math.sqrt(2*math.pi) for x in (lower,upper))
            means.append(-cdf_lower+mean*(cdf_upper-cdf_lower)+std*(density_lower-density_upper)+1-cdf_upper)
        return [0.5*(means[0]+1)*maximum_surge_mps,means[1]*maximum_yaw_radps]
    if not stochastic:
        action_mean = [math.tanh(float(value)) for value in location_values]
        return [
            0.5 * (action_mean[0] + 1.0) * maximum_surge_mps,
            maximum_yaw_radps * action_mean[1],
        ]

    scale_values = scale.detach().cpu().tolist()
    nodes = (
        0.0,
        -0.9585724646138185,
        0.9585724646138185,
        -2.020182870456086,
        2.020182870456086,
    )
    weights = (
        0.9453087204829419,
        0.3936193231522412,
        0.3936193231522412,
        0.01995324205904591,
        0.01995324205904591,
    )
    action_mean: list[float] = []
    for mean, std in zip(location_values, scale_values, strict=True):
        samples = [math.tanh(float(mean) + math.sqrt(2.0) * float(std) * node) for node in nodes]
        expected = sum(
            weight * sample for weight, sample in zip(weights, samples, strict=True)
        ) / math.sqrt(math.pi)
        action_mean.append(expected)
    return [
        0.5 * (action_mean[0] + 1.0) * maximum_surge_mps,
        maximum_yaw_radps * action_mean[1],
    ]


def _encode_actor_input(
    observation: dict[str, torch.Tensor],
    actor_index: int,
    action_location: torch.Tensor | None,
    action_scale: torch.Tensor | None,
    control_mode: str,
    applied_action: torch.Tensor,
    *,
    architecture_id: str,
    value: torch.Tensor | None = None,
    stochastic_action: bool = False,
    policy_label: str = "NN",
    weights_state: str = "runtime_weights",
    maximum_surge_mps: float = 2.0,
    maximum_yaw_radps: float = 0.2,
    horizontal_thruster_limit_n: float = 35.0,
) -> dict[str, object]:
    variant = actor_variant(architecture_id)
    architecture = variant.architecture_contract()
    if (action_location is None) != (action_scale is None):
        raise ValueError("Actor policy location and scale must both be present or absent")
    if action_location is None:
        normalized_action = applied_action.detach().cpu().tolist()
        desired_velocity_mean = [
            0.5 * (float(normalized_action[0]) + 1.0) * maximum_surge_mps,
            float(normalized_action[1]) * maximum_yaw_radps,
        ]
        neural_outputs: dict[str, object] = {}
    else:
        desired_velocity_mean = _expected_desired_velocity(
            action_location[actor_index],
            action_scale[actor_index],
            stochastic_action,
            maximum_surge_mps,
            maximum_yaw_radps,
            variant.action_distribution,
        )
        neural_outputs = {
            "actor_mean_mu_pre_tanh": (
                action_location[actor_index].detach().cpu().tolist()
            ),
            "policy_std_sigma": action_scale[actor_index].detach().cpu().tolist(),
        }
    semantic_map = observation["semantic_map"][actor_index].contiguous().cpu()
    vector = observation["vector"][actor_index].to(torch.float32).cpu()
    semantic_payload: dict[str, object] = {
        "encoding": "uint8-base64-chw",
        "shape": architecture["semantic_input_shape"],
        "coordinate_frame": variant.coordinate_frame,
        "column_order": "x_min_to_x_max",
        "row_order": "z_min_to_z_max",
        "channels": architecture["semantic_channels"],
        "values": architecture["semantic_values"],
        "data": base64.b64encode(semantic_map.numpy().tobytes()).decode("ascii"),
    }
    if "map_center" in observation:
        semantic_payload["rotation_center_m"] = (
            observation["map_center"][actor_index].cpu().tolist()
        )
    if "map_extent" in observation:
        semantic_payload["physical_extent_m"] = float(
            observation["map_extent"][actor_index]
        )
    if "meters_per_pixel" in observation:
        semantic_payload["meters_per_pixel"] = float(
            observation["meters_per_pixel"][actor_index]
        )
    entity_buffers: dict[str, object] = {}
    schemas = architecture["entity_buffer_schemas"]
    for buffer_name in variant.entity_sets:
        prefix = entity_tensor_prefix(buffer_name)
        value_key = f"{prefix}_entities"
        mask_key = f"{prefix}_mask"
        if value_key not in observation:
            continue
        entity_buffers[buffer_name] = {
            "feature_labels": schemas[buffer_name]["viewer_labels"],
            "feature_ranges": schemas[buffer_name]["viewer_ranges"],
            "mask": observation[mask_key][actor_index].to(torch.bool).cpu().tolist(),
            "values": observation[value_key][actor_index].to(torch.float32).cpu().tolist(),
        }
    return {
        "actor_kind": variant.actor_kind,
        "actor_index": actor_index,
        "control_mode": control_mode,
        "applied_action_normalized": applied_action.detach().cpu().tolist(),
        "semantic_map": {
            **semantic_payload,
        },
        "vector": vector.tolist(),
        "vector_labels": architecture["viewer_vector_labels"],
        "vector_ranges": architecture["viewer_vector_ranges"],
        "entity_buffers": entity_buffers,
        "policy": {
            "policy_label": policy_label,
            "weights_state": weights_state,
            "architecture": architecture,
            "outputs": {
                "expected_desired_velocity": desired_velocity_mean,
                "applied_desired_velocity": [
                    0.5 * (float(applied_action[0]) + 1.0) * maximum_surge_mps,
                    float(applied_action[1]) * maximum_yaw_radps,
                ],
                "maximum_surge_mps": maximum_surge_mps,
                "maximum_yaw_radps": maximum_yaw_radps,
                "horizontal_thruster_limit_n": horizontal_thruster_limit_n,
                **neural_outputs,
                **(
                    {
                        "critic_value_state_output": float(
                            value[actor_index, 0].detach().cpu()
                        )
                    }
                    if value is not None
                    else {}
                ),
            },
        },
    }


class _MemorySnapshot:
    def __init__(self):
        self._condition = threading.Condition()
        self._payload = b"{}"
        self._dynamic_payload = b"{}"
        self._block_source_revisions: dict[str, int] = {}
        self._block_revisions: dict[str, int] = {}
        self._block_payloads: dict[str, bytes] = {}
        self._sequence = -1

    @staticmethod
    def _encode(snapshot: dict[str, object]) -> bytes:
        return orjson.dumps(snapshot)

    @staticmethod
    def _merge_payloads(*payloads: bytes) -> bytes:
        bodies = [payload[1:-1] for payload in payloads if len(payload) > 2]
        return b"{" + b",".join(bodies) + b"}"

    def store(
        self,
        snapshot: dict[str, object],
        *,
        blocks: dict[str, tuple[frozenset[str], int]] | None = None,
    ) -> None:
        if not blocks:
            payload = self._encode(snapshot)
            with self._condition:
                self._payload = payload
                self._dynamic_payload = payload
                self._block_source_revisions = {}
                self._block_revisions = {}
                self._block_payloads = {}
                self._sequence = int(snapshot["sequence"])
                self._condition.notify_all()
            return

        block_keys: set[str] = set()
        for keys, _source_revision in blocks.values():
            if block_keys.intersection(keys):
                raise ValueError("Snapshot block keys must not overlap")
            block_keys.update(keys)

        with self._condition:
            source_revisions = dict(self._block_source_revisions)
            block_revisions = dict(self._block_revisions)
            block_payloads = dict(self._block_payloads)

        for name, (keys, source_revision) in blocks.items():
            resolved_source_revision = int(source_revision)
            if source_revisions.get(name) == resolved_source_revision:
                continue
            payload = self._encode({key: snapshot[key] for key in keys})
            if block_payloads.get(name) != payload:
                block_revisions[name] = block_revisions.get(name, -1) + 1
                block_payloads[name] = payload
            source_revisions[name] = resolved_source_revision

        dynamic_snapshot = {
            key: value for key, value in snapshot.items() if key not in block_keys
        }
        for name in blocks:
            dynamic_snapshot[f"{name}_revision"] = block_revisions[name]
        dynamic_payload = self._encode(dynamic_snapshot)
        full_payload = self._merge_payloads(
            dynamic_payload,
            *(block_payloads[name] for name in blocks),
            b'{"snapshot_delta":false}',
        )
        with self._condition:
            self._payload = full_payload
            self._dynamic_payload = dynamic_payload
            self._block_source_revisions = source_revisions
            self._block_revisions = block_revisions
            self._block_payloads = block_payloads
            self._sequence = int(snapshot["sequence"])
            self._condition.notify_all()

    def read(
        self,
        after_sequence: int | None = None,
        timeout_s: float = 0.0,
        block_revisions: dict[str, int] | None = None,
    ) -> bytes | None:
        with self._condition:
            if after_sequence is not None:
                self._condition.wait_for(
                    lambda: self._sequence > after_sequence,
                    timeout=max(0.0, float(timeout_s)),
                )
                if self._sequence <= after_sequence:
                    return None
            if not self._block_payloads:
                return self._payload
            client_revisions = block_revisions or {}
            changed_blocks = [
                name
                for name, revision in self._block_revisions.items()
                if client_revisions.get(name) != revision
            ]
            if not changed_blocks:
                return self._merge_payloads(
                    self._dynamic_payload,
                    b'{"snapshot_delta":true}',
                )
            if len(changed_blocks) == len(self._block_payloads):
                return self._payload
            payloads = [self._block_payloads[name] for name in changed_blocks]
            return self._merge_payloads(
                self._dynamic_payload,
                *payloads,
                b'{"snapshot_delta":true}',
            )


def _handler_factory(
    viewer_html: Path,
    snapshot_reader: Callable[[int | None, float, dict[str, int]], bytes | None],
    command_sink: Callable[[dict[str, object]], None],
    training_metrics_reader: Callable[..., bytes | None],
    training_lineage_reader: Callable[..., bytes | None],
    terrain_reader: Callable[..., bytes] | None = None,
) -> type[BaseHTTPRequestHandler]:
    instance_token = secrets.token_urlsafe(24)
    source = viewer_html.read_text(encoding="utf-8")
    if '<!--__PASSAGE_UI__-->' in source:
        source = source.replace('<!--__PASSAGE_UI__-->', '<script>' +
            (viewer_html.parent / 'passage.js').read_text(encoding='utf-8') + '</script>')
    html = source.replace(
        "__VIEWER_INSTANCE_TOKEN__",
        instance_token,
    ).replace("__VIEWER_ROLE_DESCRIPTORS__", json.dumps(ROLE_DESCRIPTORS)).encode("utf-8")
    audio_assets = {
        f"/audio/{path.name}": path.read_bytes()
        for path in sorted((viewer_html.parent / "audio").glob("*.mp3"))
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _headers(
            self,
            status: HTTPStatus,
            content_type: str,
            length: int,
            cache_control: str = "no-store",
            history_state: dict[str, object] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", cache_control)
            if history_state is not None:
                self.send_header("X-Training-Pending", str(history_state.get("pending", "")))
                self.send_header("X-Training-Generation", str(history_state["generation"]))
            self.end_headers()

        def _write(self, payload: bytes) -> None:
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _instance_conflict(self) -> None:
            payload = b'{"error":"viewer backend instance changed"}'
            self._headers(HTTPStatus.CONFLICT, "application/json", len(payload))
            self._write(payload)

        def do_GET(self) -> None:  # noqa: N802
            request = urlsplit(self.path)
            cache_control = "no-store"
            if request.path == "/api/snapshot":
                query = parse_qs(request.query)
                if query.get("viewer_instance", [""])[-1] != instance_token:
                    self._instance_conflict()
                    return
                after_sequence = int(query.get("after_sequence", ["-1"])[-1])
                block_revisions = {
                    name: int(query[f"{name}_revision"][-1])
                    for name in ("static", "status", "actor_inputs", "terrain", "world", "passage")
                    if f"{name}_revision" in query
                }
                # An unchanged/paused scene must answer before the client's
                # five-second request deadline, without fabricating a revision.
                payload = snapshot_reader(after_sequence, 1.0, block_revisions)
                if payload is None:
                    self._headers(HTTPStatus.NO_CONTENT, "application/json", 0)
                    return
                content_type = "application/json; charset=utf-8"
            elif request.path == "/api/terrain":
                from .viewer_terrain import TerrainReadError
                query = parse_qs(request.query)
                if query.get("viewer_instance", [""])[-1] != instance_token:
                    self._instance_conflict()
                    return
                try:
                    if terrain_reader is None:
                        raise TerrainReadError(503, "Terrain is unavailable")
                    payload = terrain_reader(query)
                    status = HTTPStatus.OK
                except TerrainReadError as error:
                    status = HTTPStatus(error.status)
                    payload = json.dumps({'error': str(error)}).encode()
                self._headers(status, "application/json; charset=utf-8", len(payload))
                self._write(payload)
                return
            elif request.path in {"/api/training-metrics", "/api/training-lineage"}:
                query = parse_qs(request.query)
                if query.get("viewer_instance", [""])[-1] != instance_token:
                    self._instance_conflict()
                    return
                metrics = request.path == "/api/training-metrics"
                reader = training_metrics_reader if metrics else training_lineage_reader
                args = [query.get("after_version", [None])[-1]]
                if metrics:
                    args.append(query.get("task_id", [None])[-1])
                history_state: dict[str, object] = {}
                payload = reader(
                    *args,
                    ready_only=query.get("ready_only", ["0"])[-1] == "1",
                    response_state=history_state,
                )
                if history_state.get("error"):
                    payload = b'{"error":"training history unavailable"}'
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                else:
                    status = HTTPStatus.NO_CONTENT if payload is None else HTTPStatus.OK
                self._headers(
                    status, "application/json; charset=utf-8", len(payload or b""),
                    history_state=history_state,
                )
                if payload is not None:
                    self._write(payload)
                return
            elif request.path in audio_assets:
                payload = audio_assets[request.path]
                content_type = "audio/mpeg"
                cache_control = "public, max-age=3600"
            else:
                payload = html
                content_type = "text/html; charset=utf-8"
            self._headers(
                HTTPStatus.OK,
                content_type,
                len(payload),
                cache_control,
            )
            self._write(payload)

        def do_POST(self) -> None:  # noqa: N802
            if self.headers.get("X-Viewer-Instance", "") != instance_token:
                self._instance_conflict()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Control message must be an object")
                value['_viewer_instance'] = instance_token
                command_sink(value)
            except (ValueError, json.JSONDecodeError) as error:
                payload = json.dumps({"error": str(error)}).encode("utf-8")
                self._headers(HTTPStatus.BAD_REQUEST, "application/json", len(payload))
                self._write(payload)
                return
            payload = b'{"accepted":true}'
            self._headers(HTTPStatus.ACCEPTED, "application/json", len(payload))
            self._write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler
