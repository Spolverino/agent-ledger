# CrewAI + agent-ledger

Idempotent tool execution for CrewAI agents.

## Setup

```bash
cd examples/crewai
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python agent.py
```

## Key Pattern

Wrap tool execution with the `@idempotent` decorator:

```python
@tool("Charge Customer")
@idempotent("stripe.charge")
def charge_customer(amount_cents: int) -> str:
    # This only executes once per unique args
    return stripe.charge(amount_cents)
```

Each unique `(workflow_id, tool, args)` executes exactly once.
