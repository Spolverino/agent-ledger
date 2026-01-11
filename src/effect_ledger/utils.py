from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from effect_ledger.types import ResourceDescriptor, ToolCall


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sort_keys_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_keys_recursive(item) for item in obj]
    return obj


def canonicalize(obj: Any) -> str:
    sorted_obj = _sort_keys_recursive(obj)
    return json.dumps(sorted_obj, separators=(",", ":"), ensure_ascii=False)


def resource_id_canonical(resource: ResourceDescriptor) -> str:
    id_parts = "/".join(
        f"{k}={v}" for k, v in sorted(resource.id.items(), key=lambda x: x[0])
    )
    return f"{resource.namespace}/{resource.type}/{id_parts}"


def _pick(obj: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: obj[k] for k in keys if k in obj}


def compute_idem_key(call: ToolCall) -> str:
    parts: list[str] = [call.workflow_id, call.tool]

    if call.resource is not None:
        parts.append(resource_id_canonical(call.resource))
    elif call.idempotency_keys and len(call.idempotency_keys) > 0:
        selected = _pick(call.args, call.idempotency_keys)
        parts.append(canonicalize(selected))
    else:
        parts.append(canonicalize(call.args))

    return _sha256("|".join(parts))


def generate_id() -> str:
    return str(uuid.uuid4())
