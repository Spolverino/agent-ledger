import pytest

from effect_ledger import (
    EffectDeniedError,
    EffectFailedError,
    EffectLedger,
    EffectLedgerOptions,
    EffectTimeoutError,
    MemoryStore,
    RunOptions,
    ToolCall,
    ConcurrencyOptions,
)


def make_call(**overrides) -> ToolCall:
    return ToolCall(
        workflow_id=overrides.get("workflow_id", "test-workflow"),
        tool=overrides.get("tool", "test.tool"),
        args=overrides.get("args", {"key": "value"}),
        call_id=overrides.get("call_id"),
        resource=overrides.get("resource"),
        idempotency_keys=overrides.get("idempotency_keys"),
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def ledger(store: MemoryStore) -> EffectLedger[None]:
    return EffectLedger(EffectLedgerOptions(store=store))


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

        from effect_ledger import CommitSucceeded
        await ledger.commit(begin_result.effect.id, CommitSucceeded(result="done"))

        result = await ledger.begin(call)

        assert result.idempotency_status == "replayed"
        assert result.cached is True
        assert result.cached_result == "done"

    async def test_increments_dedup_count_on_replays(
        self, store: MemoryStore, ledger: EffectLedger[None]
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

        from effect_ledger import CommitSucceeded
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

        from effect_ledger import CommitFailed, EffectError
        await ledger.commit(
            begin_result.effect.id,
            CommitFailed(error=EffectError(code="ERR_TEST", message="Something went wrong")),
        )

        updated = await ledger.get_effect(begin_result.effect.id)
        assert updated is not None
        assert updated.status.value == "failed"
        assert updated.error is not None
        assert updated.error.code == "ERR_TEST"


class TestRun:
    async def test_executes_handler_and_commits_success(
        self, store: MemoryStore, ledger: EffectLedger[None]
    ) -> None:
        async def handler(effect):
            return {"executed": True}

        result = await ledger.run(make_call(), handler)

        assert result == {"executed": True}
        assert store.size == 1

        effects = store.list_effects()
        assert effects[0].status.value == "succeeded"

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
        self, store: MemoryStore, ledger: EffectLedger[None]
    ) -> None:
        call = make_call()

        async def handler(effect):
            raise ValueError("Handler failed")

        with pytest.raises(ValueError, match="Handler failed"):
            await ledger.run(call, handler)

        effects = store.list_effects()
        assert effects[0].status.value == "failed"
        assert effects[0].error is not None
        assert effects[0].error.message == "Handler failed"

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


class TestIdempotencyKeyComputation:
    async def test_generates_same_key_for_same_tool_call(
        self, ledger: EffectLedger[None]
    ) -> None:
        call1 = make_call(args={"a": 1, "b": 2})
        call2 = make_call(args={"b": 2, "a": 1})

        e1 = await ledger.begin(call1)
        e2 = await ledger.begin(call2)

        assert e1.effect.idem_key == e2.effect.idem_key

    async def test_generates_different_keys_for_different_args(
        self, ledger: EffectLedger[None]
    ) -> None:
        e1 = await ledger.begin(make_call(args={"x": 1}))
        e2 = await ledger.begin(make_call(args={"x": 2}))

        assert e1.effect.idem_key != e2.effect.idem_key

    async def test_uses_resource_descriptor_for_key_when_provided(
        self, ledger: EffectLedger[None]
    ) -> None:
        from effect_ledger import ResourceDescriptor

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

    async def test_uses_idempotency_keys_subset_when_provided(
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


class TestFindByIdemKey:
    async def test_finds_effect_by_idempotency_key(
        self, ledger: EffectLedger[None]
    ) -> None:
        begin_result = await ledger.begin(make_call())

        found = await ledger.find_by_idem_key(begin_result.effect.idem_key)

        assert found is not None
        assert found.id == begin_result.effect.id

    async def test_returns_none_for_unknown_key(
        self, ledger: EffectLedger[None]
    ) -> None:
        found = await ledger.find_by_idem_key("unknown-key")
        assert found is None


class TestApprovalFlow:
    async def test_requests_approval_and_wait_for_it(
        self, store: MemoryStore, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approval": "test1"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)

        updated = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert updated is not None
        assert updated.status.value == "requires_approval"

    async def test_approves_effect_and_transitions_to_ready(
        self, store: MemoryStore, ledger: EffectLedger[None]
    ) -> None:
        call = make_call(args={"approval": "test2"})

        begin_result = await ledger.begin(call)
        await ledger.request_approval(begin_result.effect.idem_key)
        await ledger.approve(begin_result.effect.idem_key)

        updated = await store.find_by_idem_key(begin_result.effect.idem_key)
        assert updated is not None
        assert updated.status.value == "ready"

    async def test_denies_effect_with_reason(
        self, store: MemoryStore, ledger: EffectLedger[None]
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
