from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

import rfc8785

if TYPE_CHECKING:
    from effect_ledger.types import ResourceDescriptor, ToolCall


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize(obj: Any) -> str:
    return rfc8785.dumps(obj).decode("utf-8")


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

    return _sha256("|".join(parts).encode("utf-8"))


def generate_id() -> str:
    return str(uuid.uuid4())
