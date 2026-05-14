"""
ICL Structured Memory system for continual learning benchmark.

Adds 4 writable memory sections (schema_map, format_quirks, drift_log,
verified_joins) as optional fields on the response schema. The LLM
appends short facts to any section as part of its structured response.
Memory persists across questions within a run and gets injected at the
start of each new question. Drift detection flags schema sections as
UNVERIFIED.
"""

import re
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, create_model, model_validator

from src.errors import ProviderRefusalError
from src.interface import ContinualLearningSystem, Observation, Query, Response
from src.registry import register_system
from src.systems.utils import (
    ProviderTurnClient,
    TokenBudgetTracker,
    completion_with_structured_output,
    resolve_context_token_limit,
)

_DEFAULT_MODEL = "gpt-5.4"

_MEMORY_SECTIONS = ["schema_map", "format_quirks", "drift_log", "verified_joins"]

_MEMORY_FIELD_NAMES = {
    "schema_map": "schema_map_add",
    "format_quirks": "format_quirks_add",
    "drift_log": "drift_log_add",
    "verified_joins": "verified_joins_add",
}

_MEMORY_FIELD_DESCRIPTIONS = {
    "schema_map_add": (
        "Optional: one short fact about tables, columns, joins, or data types you "
        "just discovered (e.g. 'items_g2 has columns prc INT, ttl TEXT'). "
        "Leave null if nothing new."
    ),
    "format_quirks_add": (
        "Optional: one short fact about a non-obvious encoding you just verified "
        "(e.g. 'items_g2.prc is in cents — divide by 100 for dollars'). "
        "Leave null if nothing new."
    ),
    "drift_log_add": (
        "Optional: one short fact about a schema change, renamed table, or stale "
        "column you detected. Leave null if nothing new."
    ),
    "verified_joins_add": (
        "Optional: one short fact about a join key that produced correct results "
        "(e.g. 'fdbk_g2 joins items_g2 via ref_id'). Leave null if nothing new."
    ),
}

_RE_INCORRECT = re.compile(r"\bquestion\s+\d+:\s+incorrect\b")
_RE_CORRECT = re.compile(r"\bquestion\s+\d+:\s+correct\b")


class _MemorySchemaBase(BaseModel):
    _task_field_names: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        task_fields = cls._task_field_names
        if not task_fields:
            return data
        if any(k in data for k in task_fields):
            return data
        nested = _find_nested_dict_with_keys(data, task_fields)
        if nested is None:
            return data
        result = {k: v for k, v in nested.items() if k in cls.model_fields}
        for memory_field in _MEMORY_FIELD_NAMES.values():
            if memory_field not in result and memory_field in data:
                result[memory_field] = data[memory_field]
        return result


def _find_nested_dict_with_keys(obj: Any, expected) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if any(k in obj for k in expected):
        return obj
    for v in obj.values():
        found = _find_nested_dict_with_keys(v, expected)
        if found is not None:
            return found
    return None


def _build_memory_schema(task_schema: type[BaseModel]) -> type[BaseModel]:
    """Wrap task schema with 4 optional memory_*_add fields."""
    field_defs: dict = {}
    for name, field_info in task_schema.model_fields.items():
        field_defs[name] = (field_info.annotation, field_info)
    for field_name, description in _MEMORY_FIELD_DESCRIPTIONS.items():
        field_defs[field_name] = (
            Optional[str],
            Field(default=None, description=description),
        )
    wrapped = create_model(
        f"{task_schema.__name__}WithMemory",
        __base__=_MemorySchemaBase,
        **field_defs,
    )
    wrapped._task_field_names = frozenset(task_schema.model_fields.keys())
    return wrapped


@register_system("icl_structured_memory")
class IclStructuredMemory(ContinualLearningSystem):
    parallel_safe = True

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int | None = None,
        name: str = "icl_structured_memory",
        reserve_tokens: int = 500,
        provider_mode: str = "auto",
    ):
        self._name = name
        self.model = model
        self.max_tokens = resolve_context_token_limit(model, max_tokens)
        self.reserve_tokens = reserve_tokens
        self._provider_client = ProviderTurnClient(
            model=model,
            system_prompt="",
            provider_mode=provider_mode,
        )

        self.messages: list[dict[str, str]] = []
        self._token_budget = TokenBudgetTracker()
        self.interaction_count: int = 0

        self._memory: dict[str, list[str]] = {s: [] for s in _MEMORY_SECTIONS}
        self._drift_detected: bool = False
        self._instance_index: int = 0
        self._at_instance_boundary: bool = True

    # --- memory rendering / mutation ---

    def _memory_block(self) -> str:
        lines = ["=== STRUCTURED MEMORY ==="]
        for section in _MEMORY_SECTIONS:
            entries = self._memory[section]
            status = (
                " [UNVERIFIED — re-explore]"
                if (self._drift_detected and section in ("schema_map", "format_quirks"))
                else ""
            )
            lines.append(f"\n[{section.upper()}{status}]")
            if entries:
                lines.extend(f"  • {e}" for e in entries)
            else:
                lines.append("  • EMPTY")
        lines.append("=========================")
        return "\n".join(lines)

    def _append_memory(self, section: str, entry: str) -> None:
        if section in self._memory:
            entry = entry.strip()
            if entry and entry not in self._memory[section]:
                self._memory[section].append(entry)

    def _mark_unverified(self, sections: list[str]) -> None:
        for section in sections:
            self._memory[section] = [
                e if e.startswith("[UNVERIFIED]") else f"[UNVERIFIED] {e}"
                for e in self._memory[section]
            ]

    def _parse_observation_for_memory(self, content: str) -> None:
        lowered = content.lower()
        if "migration" in lowered or "schema changed" in lowered:
            self._drift_detected = True
            self._mark_unverified(["schema_map", "format_quirks"])
        if "no such column" in lowered or "no such table" in lowered:
            self._append_memory("drift_log", content[:120])
            self._mark_unverified(["schema_map", "verified_joins"])
        if _RE_INCORRECT.search(lowered):
            self._append_memory("drift_log", content[:120])
        elif _RE_CORRECT.search(lowered):
            self._append_memory("verified_joins", content[:120])

    def _apply_model_memory_writes(self, action: BaseModel) -> dict[str, int]:
        """Extract memory_*_add fields from the model's response and append them."""
        added = {s: 0 for s in _MEMORY_SECTIONS}
        for section, field_name in _MEMORY_FIELD_NAMES.items():
            value = getattr(action, field_name, None)
            if value:
                before = len(self._memory[section])
                self._append_memory(section, value)
                if len(self._memory[section]) > before:
                    added[section] = 1
        return added

    # --- system interface ---

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1

        if self._at_instance_boundary:
            has_content = any(self._memory[s] for s in _MEMORY_SECTIONS)
            if has_content:
                self.messages.append({"role": "user", "content": self._memory_block()})
            self._at_instance_boundary = False

        query_content = query.prompt if query.prompt else "(no content)"
        self.messages.append({"role": "user", "content": query_content})

        llm_messages = list(self.messages)
        llm_schema = _build_memory_schema(query.response_schema)

        usage_events = []
        try:
            if self._provider_client.state.provider == "litellm":
                wrapped_action, usage_event = completion_with_structured_output(
                    model=self.model,
                    messages=llm_messages,
                    response_schema=llm_schema,
                )
                usage_events = [usage_event]
                assistant_record = wrapped_action.model_dump_json()
            else:
                provider_result = self._provider_client.respond_structured(
                    messages=llm_messages,
                    response_schema=llm_schema,
                )
                wrapped_action = provider_result.action
                usage_events = provider_result.usage_events
                assistant_record = provider_result.assistant_record
        except ProviderRefusalError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        for usage_event in usage_events:
            self.record_usage_event(usage_event)

        memory_writes = self._apply_model_memory_writes(wrapped_action)

        task_fields = query.response_schema.model_fields.keys()
        task_data = {k: getattr(wrapped_action, k) for k in task_fields}
        task_action = query.response_schema(**task_data)

        self.messages.append({"role": "assistant", "content": assistant_record})

        return Response(
            action=task_action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "icl_structured_memory",
                "model": self.model,
                "drift_detected": self._drift_detected,
                "memory_facts": {s: len(v) for s, v in self._memory.items()},
                "memory_writes_this_turn": memory_writes,
            },
        )

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        content = observation.content.strip()
        if content:
            self._parse_observation_for_memory(content)
            self.messages.append({"role": "user", "content": f"FEEDBACK: {content}"})
        if observation.instance_complete:
            self._instance_index += 1
            self._at_instance_boundary = True

    def reset(self) -> None:
        self.messages = []
        self._token_budget.reset()
        self.interaction_count = 0
        self._memory = {s: [] for s in _MEMORY_SECTIONS}
        self._drift_detected = False
        self._instance_index = 0
        self._at_instance_boundary = True
        self._provider_client.reset()

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "icl_structured_memory",
            "memory": self._memory,
            "drift_detected": self._drift_detected,
            "instance_index": self._instance_index,
            "interaction_count": self.interaction_count,
            "model": self.model,
        }
