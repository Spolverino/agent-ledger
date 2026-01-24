"""
LangChain ReAct Agent with Idempotent Tools

A real LangChain agent that uses agent-ledger to prevent duplicate
side effects when the LLM retries or the agent crashes.

Usage:
    export OPENAI_API_KEY=sk-...
    python agent.py
"""

import asyncio
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent_ledger import EffectLedger, EffectLedgerOptions, MemoryStore, ToolCall

# --- Setup ---

store = MemoryStore()  # Use PostgresStore in production
ledger = EffectLedger(EffectLedgerOptions(store=store))


# --- Tools with idempotency ---


@tool
async def charge_customer(amount_cents: int, currency: str = "usd") -> dict:
    """Charge a customer's credit card. Amount is in cents (e.g., 5000 = $50.00)."""

    async def _handler(effect):
        print(f"  💳 Charging ${amount_cents / 100:.2f} {currency.upper()}...")
        # In production: return await stripe.PaymentIntent.create(...)
        return {"charge_id": "ch_xxx", "amount": amount_cents, "status": "succeeded"}

    return await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="stripe.charge",
            args={"amount_cents": amount_cents, "currency": currency},
        ),
        handler=_handler,
    )


@tool
async def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a recipient."""

    async def _handler(effect):
        print(f"  📧 Sending email to {to}: {subject}")
        # In production: return await sendgrid.send(...)
        return {"message_id": "msg_xxx", "status": "sent"}

    return await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="email.send",
            args={"to": to, "subject": subject, "body": body},
        ),
        handler=_handler,
    )


@tool
async def create_ticket(title: str, description: str, priority: str = "medium") -> dict:
    """Create a support ticket."""

    async def _handler(effect):
        print(f"  🎫 Creating ticket: {title}")
        # In production: return await zendesk.tickets.create(...)
        return {"ticket_id": "TKT-123", "status": "open"}

    return await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="tickets.create",
            args={"title": title, "description": description, "priority": priority},
        ),
        handler=_handler,
    )


# --- Agent ---

WORKFLOW_ID = "order-42"  # Scope for idempotency - use order ID, session ID, webhook idempotency key, etc.


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, [charge_customer, send_email, create_ticket])

    query = """
    Process this order:
    - Customer: alice@example.com
    - Amount: $75.00

    1. Charge the customer
    2. Send a confirmation email
    3. Create a support ticket for order tracking
    """

    print("=" * 60)
    print("USER REQUEST")
    print("=" * 60)
    print(query.strip())
    print()

    print("=" * 60)
    print("AGENT EXECUTION")
    print("=" * 60)
    response = await agent.ainvoke({"messages": [("user", query)]})

    print()
    print("=" * 60)
    print("AGENT RESPONSE")
    print("=" * 60)
    print(response["messages"][-1].content)

    print()
    print("=" * 60)
    print("LEDGER (audit trail)")
    print("=" * 60)
    for effect in await store.list_effects():
        print(f"  {effect.tool}: {effect.args_canonical} → {effect.result}")


if __name__ == "__main__":
    asyncio.run(main())
