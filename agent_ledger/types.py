from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]

# Reusable annotated types for boundary validation
NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0, le=1)]


class EffectStatus(str, Enum):
    REQUIRES_APPROVAL = "requires_approval"
    READY = "ready"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    DENIED = "denied"


TERMINAL_STATUSES: frozenset[EffectStatus] = frozenset(
    {
        EffectStatus.SUCCEEDED,
        EffectStatus.FAILED,
        EffectStatus.CANCELED,
        EffectStatus.DENIED,
    }
)

AWAITING_STATUSES: frozenset[EffectStatus] = frozenset(
    {
        EffectStatus.PROCESSING,
        EffectStatus.REQUIRES_APPROVAL,
    }
)

ALLOWED_TRANSITIONS: dict[EffectStatus, frozenset[EffectStatus]] = {
    EffectStatus.PROCESSING: frozenset(
        {
            EffectStatus.SUCCEEDED,
            EffectStatus.FAILED,
            EffectStatus.REQUIRES_APPROVAL,
        }
    ),
    EffectStatus.REQUIRES_APPROVAL: frozenset(
        {
            EffectStatus.READY,
            EffectStatus.DENIED,
            EffectStatus.CANCELED,
        }
    ),
    EffectStatus.READY: frozenset(
        {
            EffectStatus.PROCESSING,
        }
    ),
}


def is_terminal_status(status: EffectStatus) -> bool:
    return status in TERMINAL_STATUSES


def is_awaiting_status(status: EffectStatus) -> bool:
    return status in AWAITING_STATUSES


def is_valid_transition(from_status: EffectStatus, to_status: EffectStatus) -> bool:
    if from_status in TERMINAL_STATUSES:
        return False
    allowed = ALLOWED_TRANSITIONS.get(from_status)
    if allowed is None:
        return False
    return to_status in allowed


IdempotencyStatus = Literal["fresh", "replayed"]


class ResourceDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: NonEmptyStr
    type: NonEmptyStr
    id: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("id must not be empty")
        return v


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: NonEmptyStr
    tool: NonEmptyStr
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    resource: ResourceDescriptor | None = None
    idempotency_keys: list[str] | None = None

    @field_validator("call_id")
    @classmethod
    def validate_call_id_not_empty(cls, v: str | None) -> str | None:
        if v is not None and len(v) == 0:
            raise ValueError("call_id must not be empty if provided")
        return v

    @field_validator("idempotency_keys")
    @classmethod
    def validate_idempotency_keys_format(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if len(v) == 0:
                raise ValueError("idempotency_keys must not be empty if provided")
            for key in v:
                if not key:
                    raise ValueError("idempotency_keys must contain non-empty strings")
            if len(set(v)) != len(v):
                raise ValueError("idempotency_keys must not contain duplicates")
        return v

    @model_validator(mode="after")
    def validate_idempotency_keys_exist_in_args(self) -> ToolCall:
        if self.idempotency_keys is not None and self.resource is None:
            missing = [k for k in self.idempotency_keys if k not in self.args]
            if missing:
                raise ValueError(
                    f"idempotency_keys {missing} not found in args; "
                    "this would result in an empty hash component"
                )
        return self


@dataclass(frozen=True, slots=True)
class EffectError:
    message: str
    code: str | None = None


@dataclass(slots=True)
class Effect:
    id: str
    idem_key: str
    workflow_id: str
    call_id: str
    tool: str
    status: EffectStatus
    args_canonical: str
    resource_id_canonical: str
    dedup_count: int
    created_at: datetime
    updated_at: datetime
    result: JsonValue = None
    error: EffectError | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CommitSucceeded:
    status: Literal["succeeded"] = field(default="succeeded", init=False)
    result: JsonValue = None


@dataclass(frozen=True, slots=True)
class CommitFailed:
    error: EffectError
    status: Literal["failed"] = field(default="failed", init=False)


CommitOutcome = CommitSucceeded | CommitFailed


@dataclass(frozen=True, slots=True)
class BeginResult:
    effect: Effect
    cached: bool
    idempotency_status: IdempotencyStatus
    cached_result: JsonValue = None


@dataclass(frozen=True, slots=True)
class UpsertEffectInput:
    idem_key: str
    workflow_id: str
    call_id: str
    tool: str
    status: EffectStatus
    args_canonical: str
    resource_id_canonical: str
    result: JsonValue = None
    error: EffectError | None = None


@dataclass(frozen=True, slots=True)
class UpsertEffectResult:
    effect: Effect
    created: bool


class ConcurrencyOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    wait_timeout_ms: PositiveInt = 30_000
    initial_interval_ms: PositiveInt = 50
    max_interval_ms: PositiveInt = 1_000
    backoff_multiplier: PositiveFloat = 1.5
    jitter_factor: UnitFloat = 0.3

    @model_validator(mode="after")
    def validate_interval_ordering(self) -> ConcurrencyOptions:
        if self.initial_interval_ms > self.max_interval_ms:
            raise ValueError(
                f"initial_interval_ms ({self.initial_interval_ms}) must be <= "
                f"max_interval_ms ({self.max_interval_ms})"
            )
        return self


class StaleOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    after_ms: NonNegativeInt = 0


class RunOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concurrency: ConcurrencyOptions | None = None
    stale: StaleOptions | None = None
    requires_approval: bool = False


class LedgerDefaults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunOptions | None = None


class LedgerHooks(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    on_approval_required: Any | None = None

    @field_validator("on_approval_required")
    @classmethod
    def validate_callable(cls, v: Any) -> Any:
        if v is not None and not callable(v):
            raise ValueError("on_approval_required must be callable")
        return v
