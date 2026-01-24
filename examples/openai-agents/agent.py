"""
OpenAI Agents SDK with Idempotent Tools

Uses the official OpenAI Agents SDK with agent-ledger for idempotency.

Usage:
    export OPENAI_API_KEY=sk-...
    python agent.py
"""

import asyncio
import os
from typing import Annotated

from agents import Agent, Runner, function_tool

from agent_ledger import EffectLedger, EffectLedgerOptions, MemoryStore, ToolCall

# --- Setup ---

store = MemoryStore()
ledger = EffectLedger(EffectLedgerOptions(store=store))

WORKFLOW_ID = "order-42"


# --- Tools with idempotency ---


@function_tool
async def charge_customer(
    amount_cents: Annotated[int, "Amount in cents (e.g., 5000 = $50.00)"],
    currency: Annotated[str, "Currency code"] = "usd",
) -> str:
    """Charge a customer's credit card."""

    async def _handler(effect):
        print(f"  💳 Charging ${amount_cents/100:.2f} {currency.upper()}...")
        return {"charge_id": "ch_xxx", "amount": amount_cents, "status": "succeeded"}

    result = await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="stripe.charge",
            args={"amount_cents": amount_cents, "currency": currency},
        ),
        handler=_handler,
    )
    return f"Charged ${amount_cents/100:.2f}. Charge ID: {result['charge_id']}"


@function_tool
async def send_email(
    to: Annotated[str, "Recipient email address"],
    subject: Annotated[str, "Email subject"],
    body: Annotated[str, "Email body"],
) -> str:
    """Send an email to a recipient."""

    async def _handler(effect):
        print(f"  📧 Sending email to {to}: {subject}")
        return {"message_id": "msg_xxx", "status": "sent"}

    result = await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="email.send",
            args={"to": to, "subject": subject, "body": body},
        ),
        handler=_handler,
    )
    return f"Email sent to {to}. Message ID: {result['message_id']}"


@function_tool
async def create_ticket(
    title: Annotated[str, "Ticket title"],
    description: Annotated[str, "Ticket description"],
) -> str:
    """Create a support ticket."""

    async def _handler(effect):
        print(f"  🎫 Creating ticket: {title}")
        return {"ticket_id": "TKT-123", "status": "open"}

    result = await ledger.run(
        ToolCall(
            workflow_id=WORKFLOW_ID,
            tool="tickets.create",
            args={"title": title, "description": description},
        ),
        handler=_handler,
    )
    return f"Ticket created: {result['ticket_id']}"


# --- Agent ---

agent = Agent(
    name="OrderProcessor",
    instructions="You process customer orders by charging cards, sending emails, and creating tickets.",
    tools=[charge_customer, send_email, create_ticket],
)


# --- Main ---


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    query = """
    Process order #42 for alice@example.com:
    1. Charge $75.00
    2. Send confirmation email
    3. Create a support ticket
    """

    print("=" * 60)
    print("USER REQUEST")
    print("=" * 60)
    print(query.strip())
    print()

    print("=" * 60)
    print("AGENT EXECUTION")
    print("=" * 60)

    result = await Runner.run(agent, input=query)

    print()
    print("=" * 60)
    print("AGENT RESPONSE")
    print("=" * 60)
    print(result.final_output)

    print()
    print("=" * 60)
    print("LEDGER (audit trail)")
    print("=" * 60)
    for effect in await store.list_effects():
        print(f"  {effect.tool}: {effect.args_canonical} → {effect.result}")


if __name__ == "__main__":
    asyncio.run(main())
