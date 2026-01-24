# Vanilla OpenAI Function Calling + agent-ledger

The simplest integration - no framework, just OpenAI SDK with idempotent tools.

## Setup

```bash
cd examples/openai-functions
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python agent.py
```

## Key Pattern

Wrap tool execution with `ledger.run()`:

```python
async def execute_tool(name: str, args: dict) -> str:
    result = await ledger.run(
        ToolCall(workflow_id=WORKFLOW_ID, tool=name, args=args),
        handler=lambda _: tool_functions[name](**args),
    )
    return json.dumps(result)
```

This ensures each unique `(workflow_id, tool, args)` executes exactly once.
