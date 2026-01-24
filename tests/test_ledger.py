from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent_ledger import (
    ConcurrencyOptions,
    EffectDeniedError,
    EffectFailedError,
    EffectLedger,
    EffectLedgerOptions,
    EffectLedgerValidationError,
    EffectStatus,
    EffectStoreError,
    EffectTimeoutError,
    LedgerDefaults,
    LedgerHooks,
    MemoryStore,
    RunOptions,
    StaleOptions,
    ToolCall,
)
from agent_ledger.types import Effect, UpsertEffectInput

memory_only = pytest.mark.parametrize("store", ["memory"], indirect=True)


def make_call(**overrides: Any) -> ToolCall:
    return ToolCall(
        workflow_id=overrides.get("workflow_id", "test-workflow"),
        tool=overrides.get("tool", "test.tool"),
        args=overrides.get("args", {"key": "value"}),
        call_id=overrides.get("call_id"),
        resource=overrides.get("resource"),
        idempotency_keys=overrides.get("idempotency_keys"),
    )


# -----------------------------------------------------------------------------
# Core API: begin(), commit(), run()
# -----------------------------------------------------------------------------


class TestBegin:
    async def test_creates_fresh_effect_on_first_call(
        self, ledger: EffectLedger[None]
    ) -> None:
        result = await ledger.begin(make_call())

        assert result.idempotency_status == "fresh"
        assert result.cached is False
        assert result.effect.status.value == "processing"
        assert result.effect.tool == "test.tool"

    async def test_returns_replayed_status_on_duplicate_call(
        self, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()

        await ledger.begin(call)
        result = await ledger.begin(call)

        assert result.idempotency_status == "replayed"
        assert result.cached is False

    async def test_returns_cached_result_for_terminal_effect(
        self, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()
        begin_result = await ledger.begin(call)

        from agent_ledger import CommitSucceeded

        await ledger.commit(begin_result.effect.id, CommitSucceeded(result="done"))

        result = await ledger.begin(call)

        assert result.idempotency_status == "replayed"
        assert result.cached is True
        assert result.cached_result == "done"

    async def test_increments_dedup_count_on_replays(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()

        await ledger.begin(call)
        await ledger.begin(call)
        await ledger.begin(call)

        result = await ledger.begin(call)
        effect = await store.find_by_idem_key(result.effect.idem_key)
        assert effect is not None
        assert effect.dedup_count == 3


class TestCommit:
    async def test_transitions_to_succeeded_with_result(
        self, ledger: EffectLedger[None]
    ) -> None:
        begin_result = await ledger.begin(make_call())

        from agent_ledger import CommitSucceeded

        await ledger.commit(
            begin_result.effect.id,
            CommitSucceeded(result={"data": 123}),
        )

        updated = await ledger.get_effect(begin_result.effect.id)
        assert updated is not None
        assert updated.status.value == "succeeded"
        assert updated.result == {"data": 123}

    async def test_transitions_to_failed_with_error(
        self, ledger: EffectLedger[None]
    ) -> None:
        begin_result = await ledger.begin(make_call())

        from agent_ledger import CommitFailed, EffectError

        await ledger.commit(
            begin_result.effect.id,
            CommitFailed(
                error=EffectError(code="ERR_TEST", message="Something went wrong")
            ),
        )

        updated = await ledger.get_effect(begin_result.effect.id)
        assert updated is not None
        assert updated.status.value == "failed"
        assert updated.error is not None
        assert updated.error.code == "ERR_TEST"


class TestRun:
    async def test_executes_handler_and_commits_success(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"handler": "success_test"})

        async def handler(effect):
            return {"executed": True}

        result = await ledger.run(call, handler)

        assert result == {"executed": True}

        effect = await store.find_by_idem_key(
            (await ledger.begin(call)).effect.idem_key
        )
        assert effect is not None
        assert effect.status.value == "succeeded"

    async def test_returns_cached_result_on_replay(
        self, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()
        call_count = 0

        async def handler(effect):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        first = await ledger.run(call, handler)
        second = await ledger.run(call, handler)
        third = await ledger.run(call, handler)

        assert first == {"count": 1}
        assert second == {"count": 1}
        assert third == {"count": 1}
        assert call_count == 1

    async def test_commits_failure_and_rethrows_on_error(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"handler": "failure_test"})

        async def handler(effect):
            raise ValueError("Handler failed")

        with pytest.raises(ValueError, match="Handler failed"):
            await ledger.run(call, handler)

        begin_result = await ledger.begin(call)
        effect = begin_result.effect
        assert effect.status.value == "failed"
        assert effect.error is not None
        assert effect.error.message == "Handler failed"

    async def test_throws_effect_failed_error_on_replayed_failure(
        self, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()

        async def failing_handler(effect):
            raise ValueError("Original error")

        async def success_handler(effect):
            return "should not run"

        with pytest.raises(ValueError, match="Original error"):
            await ledger.run(call, failing_handler)

        with pytest.raises(EffectFailedError):
            await ledger.run(call, success_handler)

    async def test_commit_race_returns_winner_result_on_success(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"commit_race": "success"})

        handler_started = asyncio.Event()
        handler_continue = asyncio.Event()

        async def slow_handler(eff):
            handler_started.set()
            await handler_continue.wait()
            return {"from": "slow_handler"}

        task = asyncio.create_task(ledger.run(call, slow_handler))
        await handler_started.wait()

        effect = (await store.list_effects())[0]
        await store.transition(
            effect.id,
            EffectStatus.PROCESSING,
            EffectStatus.SUCCEEDED,
            result={"from": "fast_worker"},
        )

        handler_continue.set()
        result = await task

        assert result == {"from": "fast_worker"}

    async def test_commit_race_returns_winner_result_on_failure(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"commit_race": "failure"})

        handler_started = asyncio.Event()
        handler_continue = asyncio.Event()

        async def failing_handler(eff):
            handler_started.set()
            await handler_continue.wait()
            raise ValueError("Handler failed")

        task = asyncio.create_task(ledger.run(call, failing_handler))
        await handler_started.wait()

        effect = (await store.list_effects())[0]
        await store.transition(
            effect.id,
            EffectStatus.PROCESSING,
            EffectStatus.SUCCEEDED,
            result={"from": "successful_worker"},
        )

        handler_continue.set()
        result = await task

        assert result == {"from": "successful_worker"}

    async def test_result_with_none_value_replays_correctly(
        self, ledger: EffectLedger[None]
    ) -> None:
        """Handler returning None should replay correctly."""
        call = make_call(args={"result": "none_value"})

        async def handler(eff):
            return None

        result = await ledger.run(call, handler)
        assert result is None

        result2 = await ledger.run(call, handler)
        assert result2 is None

    async def test_result_with_nested_structures(
        self, ledger: EffectLedger[None]
    ) -> None:
        """Complex nested JSON should round-trip correctly."""
        call = make_call(args={"result": "nested"})

        complex_result = {
            "string": "hello",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "null": None,
            "array": [1, 2, {"nested": "value"}],
            "object": {"deep": {"deeper": {"deepest": "found"}}},
        }

        async def handler(eff):
            return complex_result

        result = await ledger.run(call, handler)
        assert result == complex_result

        result2 = await ledger.run(call, handler)
        assert result2 == complex_result

    async def test_result_with_unicode(self, ledger: EffectLedger[None]) -> None:
        """Unicode strings should be preserved."""
        call = make_call(args={"result": "unicode"})

        unicode_result = {
            "emoji": "🚀💡🎉",
            "chinese": "你好世界",
            "arabic": "مرحبا",
            "mixed": "Hello 世界 🌍",
        }

        async def handler(eff):
            return unicode_result

        result = await ledger.run(call, handler)
        assert result == unicode_result

    async def test_result_with_large_numbers(self, ledger: EffectLedger[None]) -> None:
        """Large integers should be preserved (within JSON limits)."""
        call = make_call(args={"result": "large_numbers"})

        large_result = {
            "big_int": 9007199254740991,
            "negative": -9007199254740991,
            "zero": 0,
        }

        async def handler(eff):
            return large_result

        result = await ledger.run(call, handler)
        assert result == large_result

    async def test_run_respects_external_transaction(
        self, store: Any, ledger: EffectLedger[Any]
    ) -> None:
        """Operations within run() should use provided transaction."""
        call = make_call(args={"tx": "external"})

        async def handler(eff):
            return {"in_tx": True}

        result = await ledger.run(call, handler, tx=None)
        assert result == {"in_tx": True}


# -----------------------------------------------------------------------------
# Idempotency Key Computation
# -----------------------------------------------------------------------------


class TestIdempotencyKey:
    async def test_same_key_for_same_tool_call(
        self, ledger: EffectLedger[None]
    ) -> None:
        call1 = make_call(args={"a": 1, "b": 2})
        call2 = make_call(args={"b": 2, "a": 1})

        e1 = await ledger.begin(call1)
        e2 = await ledger.begin(call2)

        assert e1.effect.idem_key == e2.effect.idem_key

    async def test_different_keys_for_different_args(
        self, ledger: EffectLedger[None]
    ) -> None:
        e1 = await ledger.begin(make_call(args={"x": 1}))
        e2 = await ledger.begin(make_call(args={"x": 2}))

        assert e1.effect.idem_key != e2.effect.idem_key

    async def test_uses_resource_descriptor_when_provided(
        self, ledger: EffectLedger[None]
    ) -> None:
        from agent_ledger import ResourceDescriptor

        call1 = make_call(
            resource=ResourceDescriptor(
                namespace="slack",
                type="channel",
                id={"name": "#general"},
            ),
            args={"text": "hello"},
        )
        call2 = make_call(
            resource=ResourceDescriptor(
                namespace="slack",
                type="channel",
                id={"name": "#general"},
            ),
            args={"text": "different"},
        )

        e1 = await ledger.begin(call1)
        e2 = await ledger.begin(call2)

        assert e1.effect.idem_key == e2.effect.idem_key

    async def test_uses_idempotency_keys_subset(
        self, ledger: EffectLedger[None]
    ) -> None:
        call1 = make_call(
            args={"user_id": "u1", "timestamp": 1000, "data": "a"},
            idempotency_keys=["user_id"],
        )
        call2 = make_call(
            args={"user_id": "u1", "timestamp": 2000, "data": "b"},
            idempotency_keys=["user_id"],
        )

        e1 = await ledger.begin(call1)
        e2 = await ledger.begin(call2)

        assert e1.effect.idem_key == e2.effect.idem_key


# -----------------------------------------------------------------------------
# Lookup
# -----------------------------------------------------------------------------


class TestLookup:
    async def test_find_by_idem_key(self, ledger: EffectLedger[None]) -> None:
        begin_result = await ledger.begin(make_call())

        found = await ledger.find_by_idem_key(begin_result.effect.idem_key)

        assert found is not None
        assert found.id == begin_result.effect.id

    async def test_find_by_idem_key_unknown_returns_none(
        self, ledger: EffectLedger[None]
    ) -> None:
        found = await ledger.find_by_idem_key("unknown-key")
        assert found is None

    async def test_get_effect_unknown_returns_none(
        self, ledger: EffectLedger[None]
    ) -> None:
        result = await ledger.get_effect("00000000-0000-0000-0000-000000000000")
        assert result is None


# -----------------------------------------------------------------------------
# Approval Flow
# -----------------------------------------------------------------------------


class TestApprovalFlow:
    async def test_request_approval(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approval": "test1"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        updated = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert updated is not None
        assert updated.status.value == "requires_approval"

    async def test_approve_transitions_to_ready(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approval": "test2"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)
        await ledger.approve(begin_result.effect.idem_key)

        updated = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert updated is not None
        assert updated.status.value == "ready"

    async def test_deny_with_reason(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approval": "test3"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)
        await ledger.deny(begin_result.effect.idem_key, "Not authorized")

        updated = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert updated is not None
        assert updated.status.value == "denied"
        assert updated.error is not None
        assert updated.error.message == "Not authorized"

    async def test_approve_nonexistent_returns_false(
        self, ledger: EffectLedger[None]
    ) -> None:
        result = await ledger.approve("nonexistent-key")
        assert result is False

    async def test_deny_nonexistent_returns_false(
        self, ledger: EffectLedger[None]
    ) -> None:
        result = await ledger.deny("nonexistent-key", "reason")
        assert result is False

    async def test_request_approval_nonexistent_returns_false(
        self, ledger: EffectLedger[None]
    ) -> None:
        result = await ledger.request_approval("nonexistent-key")
        assert result is False

    async def test_approve_wrong_status_returns_false(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approve": "wrong_status"})
        begin_result = await ledger.begin(call)

        result = await ledger.approve(begin_result.effect.idem_key)
        assert result is False

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert effect is not None
        assert effect.status == EffectStatus.PROCESSING

    async def test_deny_wrong_status_returns_false(
        self, store: Any, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"deny": "wrong_status"})
        begin_result = await ledger.begin(call)

        result = await ledger.deny(begin_result.effect.idem_key, "reason")
        assert result is False

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert effect is not None
        assert effect.status == EffectStatus.PROCESSING

    async def test_denied_during_wait_raises_error(self, store: MemoryStore) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"denied": "during_wait"})

        waiter_started = asyncio.Event()

        async def handler(eff):
            return {"should": "not run"}

        async def waiter():
            waiter_started.set()
            return await ledger.run(
                call,
                handler,
                hooks=LedgerHooks(requires_approval=lambda _: True),
            )

        task = asyncio.create_task(waiter())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        effect = (await store.list_effects())[0]
        await ledger.deny(effect.idem_key, "Access denied")

        with pytest.raises(EffectDeniedError) as exc_info:
            await task

        assert exc_info.value.reason == "Access denied"

    async def test_canceled_during_wait_raises_error(self, store: MemoryStore) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"canceled": "during_wait"})

        waiter_started = asyncio.Event()

        async def handler(eff):
            return {"should": "not run"}

        async def waiter():
            waiter_started.set()
            return await ledger.run(
                call,
                handler,
                hooks=LedgerHooks(requires_approval=lambda _: True),
            )

        task = asyncio.create_task(waiter())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        effect = (await store.list_effects())[0]
        await store.transition(
            effect.id,
            EffectStatus.REQUIRES_APPROVAL,
            EffectStatus.CANCELED,
            error={"message": "System shutdown"},
        )

        with pytest.raises(EffectDeniedError):
            await task

    async def test_approve_deny_race_only_one_succeeds(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"approve_deny": "race"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        approve_result, deny_result = await asyncio.gather(
            ledger.approve(begin_result.effect.idem_key),
            ledger.deny(begin_result.effect.idem_key, "denied"),
        )

        assert (approve_result, deny_result) in [(True, False), (False, True)]

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert effect is not None
        assert effect.status in (EffectStatus.READY, EffectStatus.DENIED)

    async def test_cancel_pending_approval(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"approval": "will_cancel"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert effect is not None
        assert effect.status == EffectStatus.REQUIRES_APPROVAL

        success = await store.transition(
            effect.id,
            EffectStatus.REQUIRES_APPROVAL,
            EffectStatus.CANCELED,
            error={"message": "Operation canceled by admin"},
        )
        assert success is True

        canceled = await store.find_by_id(effect.id)
        assert canceled is not None
        assert canceled.status == EffectStatus.CANCELED
        assert canceled.error is not None
        assert canceled.error.message == "Operation canceled by admin"

    async def test_run_with_canceled_effect_raises_denied(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"approval": "already_canceled"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        await store.transition(
            effect.id,
            EffectStatus.REQUIRES_APPROVAL,
            EffectStatus.CANCELED,
            error={"message": "Canceled"},
        )

        async def handler(eff):
            return {"should": "not run"}

        with pytest.raises(EffectDeniedError):
            await ledger.run(call, handler)


# -----------------------------------------------------------------------------
# Concurrency
# -----------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_ready_claim_only_one_executes(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"concurrent": "ready_claim"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)
        await ledger.approve(begin_result.effect.idem_key)

        effect = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert effect is not None
        assert effect.status == EffectStatus.READY

        execution_count = 0
        execution_lock = asyncio.Lock()

        async def counting_handler(eff):
            nonlocal execution_count
            async with execution_lock:
                execution_count += 1
                current = execution_count
            await asyncio.sleep(0.01)
            return {"executed_by": current}

        tasks = [ledger.run(call, counting_handler) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(r == {"executed_by": 1} for r in results)
        assert execution_count == 1

    async def test_concurrent_processing_only_one_executes(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"concurrent": "processing"})

        execution_count = 0
        execution_lock = asyncio.Lock()
        started = asyncio.Event()

        async def slow_handler(eff):
            nonlocal execution_count
            async with execution_lock:
                execution_count += 1
            started.set()
            await asyncio.sleep(0.05)
            return {"result": "done"}

        async def worker():
            return await ledger.run(call, slow_handler)

        task1 = asyncio.create_task(worker())
        await started.wait()

        tasks = [asyncio.create_task(worker()) for _ in range(4)]
        tasks.insert(0, task1)

        results = await asyncio.gather(*tasks)

        assert all(r == {"result": "done"} for r in results)
        assert execution_count == 1

    @memory_only
    async def test_stale_takeover_returns_winner_result(
        self, store: MemoryStore
    ) -> None:
        stale_options = RunOptions(stale=StaleOptions(after_ms=1000))
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"stale": "takeover"})

        worker_a_started = asyncio.Event()
        worker_a_continue = asyncio.Event()

        async def slow_handler_a(eff):
            worker_a_started.set()
            await worker_a_continue.wait()
            return {"from": "worker_a"}

        async def fast_handler_b(eff):
            return {"from": "worker_b"}

        task_a = asyncio.create_task(
            ledger.run(call, slow_handler_a, run_options=stale_options)
        )
        await worker_a_started.wait()

        effects = await store.list_effects()
        assert len(effects) == 1
        effect = effects[0]

        old_time = datetime.now(tz=timezone.utc) - timedelta(milliseconds=1500)
        aged_effect = Effect(
            id=effect.id,
            idem_key=effect.idem_key,
            workflow_id=effect.workflow_id,
            call_id=effect.call_id,
            tool=effect.tool,
            status=effect.status,
            args_canonical=effect.args_canonical,
            resource_id_canonical=effect.resource_id_canonical,
            result=effect.result,
            error=effect.error,
            dedup_count=effect.dedup_count,
            created_at=effect.created_at,
            updated_at=old_time,
            completed_at=effect.completed_at,
        )
        store._cache[effect.id] = aged_effect

        result_b = await ledger.run(call, fast_handler_b, run_options=stale_options)
        assert result_b == {"from": "worker_b"}

        worker_a_continue.set()
        result_a = await task_a
        assert result_a == {"from": "worker_b"}

    async def test_transition_failure_returns_committed_result(
        self, store: Any
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"transition": "race"})

        begin_result = await ledger.begin(call)
        effect_id = begin_result.effect.id

        await store.transition(
            effect_id,
            EffectStatus.PROCESSING,
            EffectStatus.SUCCEEDED,
            result={"from": "other_worker"},
        )

        async def our_handler(eff):
            return {"from": "our_handler"}

        result = await ledger.run(call, our_handler)
        assert result == {"from": "other_worker"}

    async def test_concurrent_approval_only_one_executes(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"approval": "concurrent"})

        execution_count = 0
        waiters_ready = asyncio.Event()
        waiter_count = 0
        waiter_lock = asyncio.Lock()

        async def handler(eff):
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.01)
            return {"executed": True}

        async def waiter():
            nonlocal waiter_count
            async with waiter_lock:
                waiter_count += 1
                if waiter_count >= 3:
                    waiters_ready.set()
            return await ledger.run(
                call,
                handler,
                hooks=LedgerHooks(requires_approval=lambda _: True),
            )

        tasks = [asyncio.create_task(waiter()) for _ in range(3)]

        await waiters_ready.wait()
        await asyncio.sleep(0.02)

        effect = (await store.list_effects())[0]
        await ledger.approve(effect.idem_key)

        results = await asyncio.gather(*tasks)

        assert all(r == {"executed": True} for r in results)
        assert execution_count == 1

    @memory_only
    async def test_multiple_stale_takeover_only_one_executes(
        self, store: MemoryStore
    ) -> None:
        stale_options = RunOptions(stale=StaleOptions(after_ms=1000))
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"multi_stale": "race"})

        begin_result = await ledger.begin(call)
        effect = begin_result.effect

        old_time = datetime.now(tz=timezone.utc) - timedelta(milliseconds=1500)
        aged_effect = Effect(
            id=effect.id,
            idem_key=effect.idem_key,
            workflow_id=effect.workflow_id,
            call_id=effect.call_id,
            tool=effect.tool,
            status=effect.status,
            args_canonical=effect.args_canonical,
            resource_id_canonical=effect.resource_id_canonical,
            result=effect.result,
            error=effect.error,
            dedup_count=effect.dedup_count,
            created_at=effect.created_at,
            updated_at=old_time,
            completed_at=effect.completed_at,
        )
        store._cache[effect.id] = aged_effect

        execution_count = 0
        execution_lock = asyncio.Lock()

        async def handler(eff):
            nonlocal execution_count
            async with execution_lock:
                execution_count += 1
                current = execution_count
            await asyncio.sleep(0.01)
            return {"executed_by": current}

        tasks = [ledger.run(call, handler, run_options=stale_options) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(r == {"executed_by": 1} for r in results)
        assert execution_count == 1

    async def test_wait_timeout_raises_error(self, store: Any) -> None:
        short_timeout = RunOptions(
            concurrency=ConcurrencyOptions(
                effect_timeout_s=0.05,
                initial_interval_s=0.01,
                max_interval_s=0.02,
            )
        )
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"timeout": "test"})

        worker_started = asyncio.Event()

        async def slow_handler(eff):
            worker_started.set()
            await asyncio.sleep(10)
            return {"done": True}

        task_a = asyncio.create_task(ledger.run(call, slow_handler))
        await worker_started.wait()

        async def fast_handler(eff):
            return {"should": "not run"}

        with pytest.raises(EffectTimeoutError) as exc_info:
            await ledger.run(call, fast_handler, run_options=short_timeout)

        assert "0.05s" in str(exc_info.value)

        task_a.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_a

    async def test_concurrent_begin_same_idem_key_only_one_creates(
        self, store: Any
    ) -> None:
        """Multiple concurrent begin() calls with same key should result in one effect."""
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"concurrent": "upsert_race"})

        results = await asyncio.gather(*[ledger.begin(call) for _ in range(10)])

        fresh_count = sum(1 for r in results if r.idempotency_status == "fresh")
        replayed_count = sum(1 for r in results if r.idempotency_status == "replayed")

        assert fresh_count == 1
        assert replayed_count == 9

        effect_ids = {r.effect.id for r in results}
        assert len(effect_ids) == 1

    async def test_concurrent_run_same_call_only_one_executes(self, store: Any) -> None:
        """Multiple concurrent run() calls should execute handler exactly once."""
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"concurrent": "run_race"})

        execution_count = 0
        execution_lock = asyncio.Lock()

        async def counting_handler(eff):
            nonlocal execution_count
            async with execution_lock:
                execution_count += 1
            await asyncio.sleep(0.05)
            return {"executed_by": execution_count}

        results = await asyncio.gather(
            *[ledger.run(call, counting_handler) for _ in range(5)]
        )

        assert all(r == {"executed_by": 1} for r in results)
        assert execution_count == 1

    async def test_stale_claim_requires_time_threshold(self, store: Any) -> None:
        """claim_for_processing should fail if effect is not stale enough."""
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"stale": "threshold_test"})

        begin_result = await ledger.begin(call)
        effect = begin_result.effect

        # Try to claim with a very long threshold - should fail since effect is fresh
        claimed = await store.claim_for_processing(
            effect.id,
            EffectStatus.PROCESSING,
            stale_threshold_ms=60000,  # 60 seconds
        )

        assert claimed is False

        # Effect should still be in PROCESSING
        current = await store.find_by_id(effect.id)
        assert current is not None
        assert current.status == EffectStatus.PROCESSING

    async def test_claim_from_ready_status(self, store: Any) -> None:
        """claim_for_processing from READY should work without threshold."""
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"claim": "from_ready"})

        begin_result = await ledger.begin(call)
        effect = begin_result.effect

        # Move to REQUIRES_APPROVAL then READY
        await store.transition(
            effect.id, EffectStatus.PROCESSING, EffectStatus.REQUIRES_APPROVAL
        )
        await store.transition(
            effect.id, EffectStatus.REQUIRES_APPROVAL, EffectStatus.READY
        )

        # Claim from READY
        claimed = await store.claim_for_processing(effect.id, EffectStatus.READY)
        assert claimed is True

        # Should now be PROCESSING
        current = await store.find_by_id(effect.id)
        assert current is not None
        assert current.status == EffectStatus.PROCESSING

    async def test_effect_disappears_during_wait_raises_invariant(
        self, store: MemoryStore
    ) -> None:
        """If effect disappears mid-wait, should raise invariant error."""
        from agent_ledger.errors import EffectLedgerInvariantError

        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"disappear": "during_wait"})

        worker_started = asyncio.Event()

        async def slow_handler(eff):
            worker_started.set()
            await asyncio.sleep(10)
            return {"done": True}

        task = asyncio.create_task(ledger.run(call, slow_handler))
        await worker_started.wait()

        effects = await store.list_effects()
        assert len(effects) == 1

        call_count = 0
        original_find = store.find_by_idem_key

        async def find_returns_none_after_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                return None
            return await original_find(*args, **kwargs)

        short_opts = RunOptions(
            concurrency=ConcurrencyOptions(
                effect_timeout_s=0.1,
                initial_interval_s=0.01,
                max_interval_s=0.02,
            )
        )

        async def second_handler(e):
            return {"second": True}

        with (
            patch.object(store, "find_by_idem_key", find_returns_none_after_first),
            pytest.raises(EffectLedgerInvariantError, match="disappeared"),
        ):
            await ledger.run(call, second_handler, run_options=short_opts)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# -----------------------------------------------------------------------------
# Store Behavior
# -----------------------------------------------------------------------------


class TestStore:
    async def test_upsert_does_not_overwrite_terminal_status(self, store: Any) -> None:
        input1 = UpsertEffectInput(
            idem_key="test-terminal-protection",
            workflow_id="wf1",
            call_id="c1",
            tool="test.tool",
            status=EffectStatus.PROCESSING,
            args_canonical='{"key": "value"}',
            resource_id_canonical="test.tool",
        )
        result1 = await store.upsert(input1)
        assert result1.created is True
        effect_id = result1.effect.id

        await store.transition(
            effect_id,
            EffectStatus.PROCESSING,
            EffectStatus.SUCCEEDED,
            result={"original": "result"},
        )

        input2 = UpsertEffectInput(
            idem_key="test-terminal-protection",
            workflow_id="wf1",
            call_id="c1",
            tool="test.tool",
            status=EffectStatus.PROCESSING,
            args_canonical='{"key": "value"}',
            resource_id_canonical="test.tool",
            result={"new": "result"},
        )
        result2 = await store.upsert(input2)

        assert result2.created is False
        assert result2.effect.status == EffectStatus.SUCCEEDED
        assert result2.effect.result == {"original": "result"}

    async def test_transition_fails_on_status_mismatch(self, store: Any) -> None:
        input_data = UpsertEffectInput(
            idem_key="test-cas",
            workflow_id="wf1",
            call_id="c1",
            tool="test.tool",
            status=EffectStatus.PROCESSING,
            args_canonical='{"key": "value"}',
            resource_id_canonical="test.tool",
        )
        result = await store.upsert(input_data)
        effect_id = result.effect.id

        success = await store.transition(
            effect_id,
            EffectStatus.READY,
            EffectStatus.SUCCEEDED,
            result={"data": 123},
        )

        assert success is False

        effect = await store.find_by_id(effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.PROCESSING
        assert effect.result is None

    async def test_transition_fails_from_terminal_status(self, store: Any) -> None:
        input_data = UpsertEffectInput(
            idem_key="test-terminal-transition",
            workflow_id="wf1",
            call_id="c1",
            tool="test.tool",
            status=EffectStatus.PROCESSING,
            args_canonical='{"key": "value"}',
            resource_id_canonical="test.tool",
        )
        result = await store.upsert(input_data)
        effect_id = result.effect.id

        await store.transition(
            effect_id,
            EffectStatus.PROCESSING,
            EffectStatus.SUCCEEDED,
            result={"done": True},
        )

        success = await store.transition(
            effect_id,
            EffectStatus.SUCCEEDED,
            EffectStatus.FAILED,
            error={"message": "oops"},
        )

        assert success is False

        effect = await store.find_by_id(effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.SUCCEEDED


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


class TestValidation:
    async def test_rejects_non_json_serializable_args(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))

        with pytest.raises(EffectLedgerValidationError) as exc_info:
            await ledger.begin(make_call(args={"func": lambda x: x}))

        assert exc_info.value.field == "args"
        assert "JSON-serializable" in str(exc_info.value)

    async def test_rejects_args_exceeding_size_limit(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store, max_args_size_bytes=100))

        with pytest.raises(EffectLedgerValidationError) as exc_info:
            await ledger.begin(make_call(args={"data": "x" * 200}))

        assert exc_info.value.field == "args"
        assert "exceed maximum size" in str(exc_info.value)

    async def test_rejects_non_dict_args(self, store: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ToolCall(
                workflow_id="test",
                tool="test.tool",
                args=["not", "a", "dict"],  # type: ignore
            )

        assert "args" in str(exc_info.value)
        assert "dictionary" in str(exc_info.value).lower()

    async def test_empty_args_dict(self, ledger: EffectLedger[None]) -> None:
        call1 = make_call(args={})
        call2 = make_call(args={})

        result1 = await ledger.begin(call1)
        result2 = await ledger.begin(call2)

        assert result1.effect.idem_key == result2.effect.idem_key
        assert result1.idempotency_status == "fresh"
        assert result2.idempotency_status == "replayed"

    async def test_unicode_in_workflow_id_and_tool(
        self, ledger: EffectLedger[None]
    ) -> None:
        call = ToolCall(
            workflow_id="工作流-123",
            tool="工具.发送消息",
            args={"message": "Hello 世界"},
        )

        async def handler(eff):
            return {"sent": True}

        result = await ledger.run(call, handler)
        assert result == {"sent": True}

        result2 = await ledger.run(call, handler)
        assert result2 == {"sent": True}

    async def test_args_at_size_limit(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store, max_args_size_bytes=500))

        call = make_call(args={"data": "x" * 400})

        async def handler(eff):
            return {"ok": True}

        result = await ledger.run(call, handler)
        assert result == {"ok": True}


# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    async def test_store_error_wraps_backend_exceptions(self) -> None:
        store = MemoryStore()
        store._id_to_idem_key["some-id"] = "some-idem-key"

        with (
            patch.object(
                store._cache, "get", side_effect=RuntimeError("Cache corrupted")
            ),
            pytest.raises(EffectStoreError) as exc_info,
        ):
            await store.find_by_id("some-id")

        assert exc_info.value.operation == "find_by_id"
        assert exc_info.value.effect_id == "some-id"
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "Cache corrupted" in str(exc_info.value.__cause__)

    async def test_store_error_str_includes_context(self) -> None:
        err = EffectStoreError(
            "Test error",
            operation="upsert",
            idem_key="test-key",
            effect_id="test-id",
        )

        err_str = str(err)
        assert "Test error" in err_str
        assert "operation=upsert" in err_str
        assert "idem_key=test-key" in err_str
        assert "effect_id=test-id" in err_str

    async def test_store_error_not_double_wrapped(self) -> None:
        store = MemoryStore()
        store._id_to_idem_key["some-id"] = "some-idem-key"

        original_store_error = EffectStoreError(
            "Already wrapped",
            operation="inner_op",
        )

        with (
            patch.object(store._cache, "get", side_effect=original_store_error),
            pytest.raises(EffectStoreError) as exc_info,
        ):
            await store.find_by_id("some-id")

        assert exc_info.value is original_store_error
        assert exc_info.value.operation == "inner_op"

    async def test_store_error_from_transition(self) -> None:
        store = MemoryStore()
        ledger = EffectLedger(EffectLedgerOptions(store=store))

        begin_result = await ledger.begin(make_call())
        effect_id = begin_result.effect.id

        with (
            patch.object(store._cache, "get", side_effect=OSError("Disk full")),
            pytest.raises(EffectStoreError) as exc_info,
        ):
            await store.transition(
                effect_id,
                EffectStatus.PROCESSING,
                EffectStatus.SUCCEEDED,
                result={"ok": True},
            )

        assert exc_info.value.operation == "transition"
        assert exc_info.value.effect_id == effect_id
        assert "Disk full" in str(exc_info.value.__cause__)

    async def test_handler_raising_effect_failed_error_not_wrapped(
        self, store: Any
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"error": "effect_failed"})

        async def handler(eff):
            raise EffectFailedError(
                "test-key", {"code": "INNER", "message": "Inner error"}
            )

        with pytest.raises(EffectFailedError) as exc_info:
            await ledger.run(call, handler)

        assert exc_info.value.code == "INNER"
        assert "Inner error" in str(exc_info.value)

    async def test_handler_raising_effect_denied_error_not_wrapped(
        self, store: Any
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"error": "effect_denied"})

        async def handler(eff):
            raise EffectDeniedError("test-key", "Access denied")

        with pytest.raises(EffectDeniedError) as exc_info:
            await ledger.run(call, handler)

        assert exc_info.value.reason == "Access denied"

    async def test_generic_handler_error_recorded_in_effect(self, store: Any) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"error": "generic"})

        async def handler(eff):
            raise RuntimeError("Something broke")

        with pytest.raises(RuntimeError, match="Something broke"):
            await ledger.run(call, handler)

        begin_result = await ledger.begin(call)
        assert begin_result.effect.status == EffectStatus.FAILED
        assert begin_result.effect.error is not None
        assert "Something broke" in begin_result.effect.error.message

    async def test_store_error_during_commit_propagates(self) -> None:
        store = MemoryStore()
        ledger = EffectLedger(EffectLedgerOptions(store=store))

        async def handler(eff):
            return {"success": True}

        async def failing_transition(*args, **kwargs):
            raise EffectStoreError(
                "Connection lost",
                operation="transition",
            )

        with (
            patch.object(store, "transition", failing_transition),
            pytest.raises(EffectStoreError) as exc_info,
        ):
            call = make_call(args={"commit": "will_fail"})
            await ledger.run(call, handler)

        assert exc_info.value.operation == "transition"


# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------


class TestHooks:
    async def test_on_approval_required_fires_on_fresh_creation(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"hooks": "fresh_creation"})

        hook_calls: list[Any] = []

        async def on_approval(effect: Any) -> None:
            hook_calls.append(effect)

        hooks = LedgerHooks(
            requires_approval=lambda _: True,
            on_approval_required=on_approval,
        )

        async def handler(eff: Any) -> dict[str, bool]:
            return {"done": True}

        waiter_started = asyncio.Event()

        async def run_with_approval() -> Any:
            waiter_started.set()
            return await ledger.run(
                call,
                handler,
                hooks=hooks,
            )

        task = asyncio.create_task(run_with_approval())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        assert len(hook_calls) == 1
        assert hook_calls[0].tool == "test.tool"

        effect = (await store.list_effects())[0]
        await ledger.approve(effect.idem_key)

        result = await task
        assert result == {"done": True}

    async def test_on_approval_required_does_not_fire_on_replay(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"hooks": "replay_test"})

        hook_calls: list[Any] = []

        async def on_approval(effect: Any) -> None:
            hook_calls.append(effect)

        hooks = LedgerHooks(
            requires_approval=lambda _: True,
            on_approval_required=on_approval,
        )

        async def handler(eff: Any) -> dict[str, bool]:
            return {"done": True}

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        worker_started = asyncio.Event()

        async def run_as_second_worker() -> Any:
            worker_started.set()
            return await ledger.run(
                call,
                handler,
                hooks=hooks,
            )

        task = asyncio.create_task(run_as_second_worker())
        await worker_started.wait()
        await asyncio.sleep(0.02)

        assert len(hook_calls) == 0

        await ledger.approve(begin_result.effect.idem_key)
        result = await task
        assert result == {"done": True}

    async def test_hook_error_does_not_abort_run(self, store: MemoryStore) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"hooks": "error_handling"})

        async def failing_hook(effect: Any) -> None:
            raise RuntimeError("Hook failed!")

        hooks = LedgerHooks(
            requires_approval=lambda _: True,
            on_approval_required=failing_hook,
        )

        async def handler(eff: Any) -> dict[str, bool]:
            return {"done": True}

        waiter_started = asyncio.Event()

        async def run_with_failing_hook() -> Any:
            waiter_started.set()
            return await ledger.run(
                call,
                handler,
                hooks=hooks,
            )

        task = asyncio.create_task(run_with_failing_hook())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        effect = (await store.list_effects())[0]
        assert effect.status == EffectStatus.REQUIRES_APPROVAL

        await ledger.approve(effect.idem_key)

        result = await task
        assert result == {"done": True}

    async def test_idem_key_convenience_method(self, store: MemoryStore) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))
        call = make_call(args={"idem_key": "test"})

        from agent_ledger import compute_idem_key

        expected_key = compute_idem_key(call)
        actual_key = ledger.idem_key(call)

        assert actual_key == expected_key

        begin_result = await ledger.begin(call)
        assert begin_result.effect.idem_key == actual_key

    def test_hooks_validates_callable(self) -> None:
        from pydantic import ValidationError

        from agent_ledger import LedgerHooks

        with pytest.raises(ValidationError) as exc_info:
            LedgerHooks(on_approval_required="not a callable")

        assert "on_approval_required must be callable" in str(exc_info.value)

    def test_hooks_accepts_valid_callable(self) -> None:
        from agent_ledger import LedgerHooks

        async def valid_hook(effect: Any) -> None:
            pass

        hooks = LedgerHooks(on_approval_required=valid_hook)
        assert hooks.on_approval_required is valid_hook

    def test_hooks_accepts_none(self) -> None:
        from agent_ledger import LedgerHooks

        hooks = LedgerHooks(on_approval_required=None)
        assert hooks.on_approval_required is None

        hooks_default = LedgerHooks()
        assert hooks_default.on_approval_required is None


class TestPolicyHook:
    async def test_requires_approval_hook_triggers_approval_flow(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        def policy(call: ToolCall) -> bool:
            return call.tool == "dangerous.tool"

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"executed": True}

        safe_call = make_call(tool="safe.tool", args={"policy": "safe"})
        result = await ledger.run(safe_call, handler, hooks=hooks)
        assert result == {"executed": True}

        safe_effect = await store.find_by_idem_key(ledger.idem_key(safe_call))
        assert safe_effect is not None
        assert safe_effect.status == EffectStatus.SUCCEEDED

    async def test_requires_approval_hook_blocks_dangerous_tool(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        def policy(call: ToolCall) -> bool:
            return call.tool == "dangerous.tool"

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"executed": True}

        dangerous_call = make_call(tool="dangerous.tool", args={"policy": "dangerous"})

        waiter_started = asyncio.Event()

        async def run_dangerous() -> Any:
            waiter_started.set()
            return await ledger.run(dangerous_call, handler, hooks=hooks)

        task = asyncio.create_task(run_dangerous())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        dangerous_effect = await store.find_by_idem_key(ledger.idem_key(dangerous_call))
        assert dangerous_effect is not None
        assert dangerous_effect.status == EffectStatus.REQUIRES_APPROVAL

        await ledger.approve(dangerous_effect.idem_key)
        result = await task
        assert result == {"executed": True}

    async def test_requires_approval_hook_checks_args(self, store: MemoryStore) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        def policy(call: ToolCall) -> bool:
            return call.args.get("amount", 0) > 1000

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"charged": True}

        small_call = make_call(tool="stripe.charge", args={"amount": 500})
        result = await ledger.run(small_call, handler, hooks=hooks)
        assert result == {"charged": True}

        small_effect = await store.find_by_idem_key(ledger.idem_key(small_call))
        assert small_effect is not None
        assert small_effect.status == EffectStatus.SUCCEEDED

    async def test_requires_approval_hook_large_amount_requires_approval(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        def policy(call: ToolCall) -> bool:
            return call.args.get("amount", 0) > 1000

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"charged": True}

        large_call = make_call(tool="stripe.charge", args={"amount": 5000})

        waiter_started = asyncio.Event()

        async def run_large() -> Any:
            waiter_started.set()
            return await ledger.run(large_call, handler, hooks=hooks)

        task = asyncio.create_task(run_large())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        large_effect = await store.find_by_idem_key(ledger.idem_key(large_call))
        assert large_effect is not None
        assert large_effect.status == EffectStatus.REQUIRES_APPROVAL

        await ledger.approve(large_effect.idem_key)
        result = await task
        assert result == {"charged": True}

    async def test_requires_approval_hook_takes_precedence_over_run_options(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        def policy(call: ToolCall) -> bool:
            return False

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"executed": True}

        call = make_call(args={"no_approval": "test"})
        result = await ledger.run(
            call,
            handler,
            hooks=hooks,
        )

        assert result == {"executed": True}

        effect = await store.find_by_idem_key(ledger.idem_key(call))
        assert effect is not None
        assert effect.status == EffectStatus.SUCCEEDED

    async def test_requires_approval_hook_with_on_approval_required(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        notification_calls: list[Any] = []

        def policy(call: ToolCall) -> bool:
            return call.tool == "notify.tool"

        async def on_approval(effect: Any) -> None:
            notification_calls.append(effect)

        hooks = LedgerHooks(
            requires_approval=policy,
            on_approval_required=on_approval,
        )

        async def handler(eff: Any) -> dict[str, bool]:
            return {"done": True}

        call = make_call(tool="notify.tool", args={"combined": "test"})

        waiter_started = asyncio.Event()

        async def run_with_hooks() -> Any:
            waiter_started.set()
            return await ledger.run(call, handler, hooks=hooks)

        task = asyncio.create_task(run_with_hooks())
        await waiter_started.wait()
        await asyncio.sleep(0.02)

        assert len(notification_calls) == 1
        assert notification_calls[0].tool == "notify.tool"

        effect = (await store.list_effects())[0]
        await ledger.approve(effect.idem_key)

        result = await task
        assert result == {"done": True}

    def test_requires_approval_hook_validates_callable(self) -> None:
        from pydantic import ValidationError

        from agent_ledger import LedgerHooks

        with pytest.raises(ValidationError) as exc_info:
            LedgerHooks(requires_approval="not a callable")

        assert "requires_approval must be callable" in str(exc_info.value)

    def test_requires_approval_hook_accepts_lambda(self) -> None:
        from agent_ledger import LedgerHooks

        hooks = LedgerHooks(requires_approval=lambda call: call.tool == "test")
        assert hooks.requires_approval is not None
        assert callable(hooks.requires_approval)

    async def test_requires_approval_hook_not_called_on_replay(
        self, store: MemoryStore
    ) -> None:
        from agent_ledger import LedgerHooks

        ledger = EffectLedger(EffectLedgerOptions(store=store))

        policy_calls: list[ToolCall] = []

        def policy(call: ToolCall) -> bool:
            policy_calls.append(call)
            return False

        hooks = LedgerHooks(requires_approval=policy)

        async def handler(eff: Any) -> dict[str, bool]:
            return {"done": True}

        call = make_call(args={"replay": "test"})

        await ledger.run(call, handler, hooks=hooks)
        assert len(policy_calls) == 1

        await ledger.run(call, handler, hooks=hooks)
        assert len(policy_calls) == 1


# -----------------------------------------------------------------------------
# Config Merging: _merge_options / _get_concurrency_field
# -----------------------------------------------------------------------------


class TestConfigMerging:
    def test_per_call_explicit_none_overrides_instance_default(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(approval_timeout_s=60.0)
        )
        per_call = RunOptions(concurrency=ConcurrencyOptions(approval_timeout_s=None))

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.approval_timeout_s is None

    def test_per_call_value_overrides_instance_default(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(approval_timeout_s=60.0)
        )
        per_call = RunOptions(concurrency=ConcurrencyOptions(approval_timeout_s=120.0))

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.approval_timeout_s == 120.0

    def test_unset_per_call_uses_instance_default(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(approval_timeout_s=60.0)
        )
        per_call = RunOptions(concurrency=ConcurrencyOptions())

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.approval_timeout_s == 60.0

    def test_unset_both_uses_global_default(self) -> None:
        from agent_ledger.ledger import DEFAULT_CONCURRENCY, _merge_options

        instance_defaults = RunOptions(concurrency=ConcurrencyOptions())
        per_call = RunOptions(concurrency=ConcurrencyOptions())

        merged = _merge_options(instance_defaults, per_call)

        assert (
            merged.concurrency.approval_timeout_s
            == DEFAULT_CONCURRENCY.approval_timeout_s
        )

    def test_no_options_uses_global_default(self) -> None:
        from agent_ledger.ledger import DEFAULT_CONCURRENCY, _merge_options

        merged = _merge_options(None, None)

        assert (
            merged.concurrency.approval_timeout_s
            == DEFAULT_CONCURRENCY.approval_timeout_s
        )
        assert (
            merged.concurrency.effect_timeout_s == DEFAULT_CONCURRENCY.effect_timeout_s
        )

    def test_instance_explicit_none_preserved_when_per_call_unset(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(approval_timeout_s=None)
        )
        per_call = RunOptions(concurrency=ConcurrencyOptions())

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.approval_timeout_s is None

    def test_all_concurrency_fields_merge_correctly(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(
                effect_timeout_s=10.0,
                initial_interval_s=0.1,
            )
        )
        per_call = RunOptions(
            concurrency=ConcurrencyOptions(
                effect_timeout_s=20.0,
                max_interval_s=5.0,
            )
        )

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.effect_timeout_s == 20.0
        assert merged.concurrency.initial_interval_s == 0.1
        assert merged.concurrency.max_interval_s == 5.0

    def test_each_concurrency_field_merges_independently(self) -> None:
        from agent_ledger.ledger import DEFAULT_CONCURRENCY, _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(
                effect_timeout_s=10.0,
                approval_timeout_s=30.0,
                initial_interval_s=0.2,
            )
        )
        per_call = RunOptions(
            concurrency=ConcurrencyOptions(
                approval_timeout_s=60.0,
                max_interval_s=2.0,
                backoff_multiplier=2.0,
            )
        )

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.effect_timeout_s == 10.0
        assert merged.concurrency.approval_timeout_s == 60.0
        assert merged.concurrency.initial_interval_s == 0.2
        assert merged.concurrency.max_interval_s == 2.0
        assert merged.concurrency.backoff_multiplier == 2.0
        assert merged.concurrency.jitter_factor == DEFAULT_CONCURRENCY.jitter_factor

    def test_stale_options_per_call_overrides_instance(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(stale=StaleOptions(after_ms=5000))
        per_call = RunOptions(stale=StaleOptions(after_ms=10000))

        merged = _merge_options(instance_defaults, per_call)

        assert merged.stale.after_ms == 10000

    def test_stale_options_instance_used_when_per_call_unset(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(stale=StaleOptions(after_ms=5000))
        per_call = RunOptions()

        merged = _merge_options(instance_defaults, per_call)

        assert merged.stale.after_ms == 5000

    def test_stale_options_default_when_both_unset(self) -> None:
        from agent_ledger.ledger import DEFAULT_STALE, _merge_options

        merged = _merge_options(None, None)

        assert merged.stale.after_ms == DEFAULT_STALE.after_ms

    def test_mixed_concurrency_and_stale_options(self) -> None:
        from agent_ledger.ledger import _merge_options

        instance_defaults = RunOptions(
            concurrency=ConcurrencyOptions(effect_timeout_s=15.0),
            stale=StaleOptions(after_ms=3000),
        )
        per_call = RunOptions(
            concurrency=ConcurrencyOptions(approval_timeout_s=45.0),
            stale=StaleOptions(after_ms=6000),
        )

        merged = _merge_options(instance_defaults, per_call)

        assert merged.concurrency.effect_timeout_s == 15.0
        assert merged.concurrency.approval_timeout_s == 45.0
        assert merged.stale.after_ms == 6000


class TestConfigMergingPublicAPI:
    async def test_run_options_override_ledger_defaults(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(
            EffectLedgerOptions(
                store=store,
                defaults=LedgerDefaults(
                    run=RunOptions(
                        concurrency=ConcurrencyOptions(effect_timeout_s=100.0)
                    )
                ),
            )
        )

        call = make_call(args={"config_test": "run_options_override"})

        async def handler(effect: Any) -> str:
            return "done"

        result = await ledger.run(
            call,
            handler,
            run_options=RunOptions(
                concurrency=ConcurrencyOptions(effect_timeout_s=50.0)
            ),
        )

        assert result == "done"

    async def test_ledger_defaults_applied_when_no_run_options(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(
            EffectLedgerOptions(
                store=store,
                defaults=LedgerDefaults(
                    run=RunOptions(
                        concurrency=ConcurrencyOptions(effect_timeout_s=100.0)
                    )
                ),
            )
        )

        call = make_call(args={"config_test": "ledger_defaults"})

        async def handler(effect: Any) -> str:
            return "done"

        result = await ledger.run(call, handler)

        assert result == "done"

    async def test_approval_timeout_none_allows_indefinite_wait(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(
            EffectLedgerOptions(
                store=store,
                defaults=LedgerDefaults(
                    run=RunOptions(
                        concurrency=ConcurrencyOptions(approval_timeout_s=60.0)
                    )
                ),
            )
        )

        call = make_call(args={"config_test": "indefinite_approval"})

        async def handler(effect: Any) -> str:
            return "approved"

        hooks = LedgerHooks(requires_approval=lambda _: True)

        async def run_and_approve() -> str:
            task = asyncio.create_task(
                ledger.run(
                    call,
                    handler,
                    hooks=hooks,
                    run_options=RunOptions(
                        concurrency=ConcurrencyOptions(approval_timeout_s=None)
                    ),
                )
            )
            await asyncio.sleep(0.05)
            effect = (await store.list_effects())[0]
            await ledger.approve(effect.idem_key)
            return await task

        result = await run_and_approve()
        assert result == "approved"

    async def test_effect_timeout_triggers_on_stale_processing(
        self, store: MemoryStore
    ) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))

        call = make_call(args={"config_test": "effect_timeout"})

        begin_result = await ledger.begin(call)
        assert begin_result.effect.status == EffectStatus.PROCESSING

        async def handler(effect: Any) -> str:
            return "done"

        with pytest.raises(EffectTimeoutError):
            await ledger.run(
                call,
                handler,
                run_options=RunOptions(
                    concurrency=ConcurrencyOptions(effect_timeout_s=0.01)
                ),
            )

    async def test_stale_options_via_run_options(self, store: MemoryStore) -> None:
        ledger = EffectLedger(EffectLedgerOptions(store=store))

        call = make_call(args={"config_test": "stale_options"})

        async def handler(effect: Any) -> str:
            return "done"

        result = await ledger.run(
            call,
            handler,
            run_options=RunOptions(stale=StaleOptions(after_ms=5000)),
        )

        assert result == "done"

    async def test_stale_options_via_ledger_defaults(self, store: MemoryStore) -> None:
        ledger = EffectLedger(
            EffectLedgerOptions(
                store=store,
                defaults=LedgerDefaults(
                    run=RunOptions(stale=StaleOptions(after_ms=5000))
                ),
            )
        )

        call = make_call(args={"config_test": "stale_defaults"})

        async def handler(effect: Any) -> str:
            return "done"

        result = await ledger.run(call, handler)

        assert result == "done"
