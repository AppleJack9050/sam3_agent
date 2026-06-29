"""LLM-assisted prompt adaptation via the **Claude Agent SDK**.

This replaces the old LangChain `ChatAnthropic(...).with_structured_output(...)`
call. We drive the SDK's one-shot ``query()`` with a JSON-schema
``output_format`` so Claude returns a typed decision (``prompts``, ``reasoning``).

Authentication uses your **Claude subscription**, not a metered API key:
the SDK shells out to the Claude Code CLI (`claude`), which uses the OAuth
credentials created by ``claude login`` / ``claude setup-token``. As long as
``ANTHROPIC_API_KEY`` is *not* set in the environment, billing goes to your
Claude Pro/Max plan. See the README "Install" section.

The Claude Agent SDK is imported lazily inside the async call so that importing
this module (and running the deterministic fallback) never requires the SDK or
the CLI to be installed.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import List, Sequence, Tuple

from pydantic import BaseModel, Field

from .config import AgentConfig


class PromptSetDecision(BaseModel):
    prompts: List[str] = Field(description="Next exclude-mode prompt set.")
    reasoning: str = Field(description="Brief reasoning.")


_SYSTEM_PROMPT = (
    "You orchestrate a SAM 3 segment-by-exclusion agent for glacier-terminus "
    "scenes. The agent queries SAM 3 for each prompt in a list of *exclude* "
    "classes (sky, clouds, far mountains, man-made objects, etc.), unions the "
    "masks, and inverts to keep the foreground. Given the last prompt set and "
    "the reasons it failed quality checks, propose the next prompt set. Keep "
    "prompts as short noun phrases. Return only the structured decision."
)


def _run_coro(coro):
    """Run an async coroutine to completion from synchronous code.

    Always executes on a dedicated thread with its own event loop, so this is
    safe whether or not the caller is already inside a running event loop
    (e.g. a FastAPI ``async def`` endpoint), where a bare ``asyncio.run`` would
    raise "cannot be called from a running event loop".
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


async def _aquery_decision(
    cfg: AgentConfig,
    attempt: int,
    last_prompts: Sequence[str],
    reasons: Sequence[str],
    hints: Sequence[Tuple[str, ...]],
) -> PromptSetDecision:
    # Imported here (not at module top) so the deterministic fallback works
    # even when claude-agent-sdk / the Claude Code CLI are not installed.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    user_prompt = (
        f"Preserve target: {cfg.preserve_label}\n"
        f"Attempt: {attempt}\n"
        f"Last exclude prompts: {json.dumps(list(last_prompts))}\n"
        f"Quality failures: {json.dumps(list(reasons))}\n"
        f"Hint prompt sets: {json.dumps([list(h) for h in hints])}\n"
        "Propose the next exclude prompt set as short noun phrases."
    )

    options = ClaudeAgentOptions(
        system_prompt=_SYSTEM_PROMPT,
        model=cfg.llm_model,
        max_turns=1,            # one-shot LLM call, no agentic tool loop
        allowed_tools=[],       # no file/bash/web tools
        permission_mode="dontAsk",
        setting_sources=None,   # ignore project/user settings files
        max_budget_usd=cfg.llm_max_budget_usd,
        output_format={
            "type": "json_schema",
            "schema": PromptSetDecision.model_json_schema(),
        },
    )

    structured = None
    text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(
                    f"Claude Agent SDK error: {message.errors or message.subtype}"
                )
            structured = message.structured_output

    if structured is None:
        # No schema-validated payload came back; try to parse the raw text.
        structured = json.loads(text)
    return PromptSetDecision.model_validate(structured)


def decide_next_prompts(
    cfg: AgentConfig,
    attempt: int,
    last_prompts: Sequence[str],
    reasons: Sequence[str],
    hints: Sequence[Tuple[str, ...]],
) -> PromptSetDecision:
    """Ask Claude (via the Agent SDK, on your subscription) for the next prompt
    set. Raises on any SDK/CLI/auth failure — callers fall back deterministically.
    """
    return _run_coro(_aquery_decision(cfg, attempt, last_prompts, reasons, hints))
