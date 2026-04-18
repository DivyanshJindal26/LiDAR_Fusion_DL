"""
FastAPI router for the /chat endpoint.
Proxies to OpenRouter (OpenAI-compatible API) so the API key never reaches the browser.
Handles the full tool-use agentic loop server-side before returning a final response.
"""
import os
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

router = APIRouter()

# OpenRouter is OpenAI-compatible
_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:3000"),
        "X-Title": "LiDAR Fusion Demo",
    },
)

MODEL = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-3-mini-beta")


def _content_to_text(content: Any) -> str:
    """Convert mixed content payloads into plain text for provider compatibility."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Accept common text block shapes; ignore tool blocks.
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize incoming chat history into OpenAI-compatible message objects."""
    normalized: list[dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        if role == "tool":
            tool_call_id = m.get("tool_call_id")
            if not tool_call_id:
                continue
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _content_to_text(m.get("content")),
                }
            )
            continue

        item: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(m.get("content")),
        }

        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            tc_norm = []
            for tc in m["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
                fn_name = fn.get("name") if fn else tc.get("name")
                fn_args = fn.get("arguments") if fn else tc.get("input")
                if fn_name is None:
                    continue
                if isinstance(fn_args, dict):
                    fn_args = json.dumps(fn_args)
                elif fn_args is None:
                    fn_args = "{}"
                tc_norm.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args,
                        },
                    }
                )
            if tc_norm:
                item["tool_calls"] = tc_norm

        normalized.append(item)

    return normalized


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    scene_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    done: bool = True


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    system = req.scene_context.get("system", "") if req.scene_context else ""
    tools = req.scene_context.get("tools") if req.scene_context else None

    messages = _normalize_messages(req.messages)
    if system:
        messages = [{"role": "system", "content": system}] + messages

    # Agentic loop: keep calling until no more tool_calls
    max_rounds = 5
    for _ in range(max_rounds):
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await _client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            # Final text response
            return ChatResponse(content=_content_to_text(msg.content), done=True)

        # Return the tool call info to the frontend to execute
        tool_calls_out = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments or "{}"),
            }
            for tc in msg.tool_calls
        ]
        return ChatResponse(
            content=_content_to_text(msg.content),
            tool_calls=tool_calls_out,
            done=False,
        )

    return ChatResponse(content="Max tool-use rounds reached.", done=True)
