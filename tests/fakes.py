"""Test doubles for LLM clients."""

from __future__ import annotations

from app.llm.base import LLMClient, LLMRequest


class SequenceClient(LLMClient):
    """Returns prepared responses (or raises prepared exceptions) in order."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, responses: list[str | Exception]) -> None:
        super().__init__("fake-1")
        self._responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if not self._responses:
            raise AssertionError("SequenceClient ran out of prepared responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
