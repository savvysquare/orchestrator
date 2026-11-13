from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


# your routes and other code go below

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}
#!/usr/bin/env python3
"""
Multi-AI Orchestrator with Graceful Degradation (LangGraph)
===========================================================

Flow:

    START -> planner -> router -> executor -> verifier -> (router | finalizer) -> END
                          ^           |            |
                          |           v            v
                          +------ fallback_handler (retry on next model / degrade)
                          |           |
                          |           v  (limits hit or no models left)
                          +------ human_handoff  <-- interrupt(); resume with Command(resume=...)

Install:
    pip install -U langgraph langchain-core langchain-anthropic langchain-openai pydantic

API keys (set at least one; two or more gives real fallback across providers):
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export OPENROUTER_API_KEY=...        # optional third provider (OpenAI-compatible)

Optional model overrides (model names change often, so adjust to what your accounts offer):
    ANTHROPIC_STRONG_MODEL, ANTHROPIC_FAST_MODEL,
    OPENAI_STRONG_MODEL, OPENAI_FAST_MODEL, OPENROUTER_FAST_MODEL

Run:
    python multi_ai_orchestrator.py
    python multi_ai_orchestrator.py --goal "Your goal here"
    python multi_ai_orchestrator.py --simulate-fail-step 2          # forces a human handoff on step 2
    python multi_ai_orchestrator.py --simulate-fail-step 2 --auto-resume
"""



import argparse
import ast
import importlib.util
import json
import logging
import operator
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================================
# 0. CONFIGURATION (all limits live here so nothing can loop forever)
# ============================================================================
MAX_PLAN_STEPS = 6              # hard cap on planner output
MAX_FALLBACKS_PER_STEP = 3      # failures tolerated on one step before a human is asked
MAX_TOTAL_FALLBACKS = 10        # failures tolerated across the whole run
CIRCUIT_BREAKER_THRESHOLD = 2   # consecutive exceptions before a model is skipped for later steps
VERIFY_PASS_SCORE = 0.6         # minimum verifier score to accept a step
REQUEST_TIMEOUT_S = 90
RECURSION_LIMIT = 150           # LangGraph's own safety net


# ============================================================================
# 1. SHARED STATE
# ============================================================================
class OrchestratorState(TypedDict, total=False):
    # --- Required fields -----------------------------------------------------
    user_goal: str
    plan: list[dict]                       # ordered steps (PlanStep as dicts)
    current_step: int                      # index into plan
    results: dict[str, dict]               # step_id -> {title, output, model, status, score}
    errors: Annotated[list[str], operator.add]   # append-only error log
    fallback_count: int                    # total fallbacks used in this run
    final_output: str
    handoff_instructions: str              # plain-language instructions for the human
    status: str                            # planning|running|degrading|needs_human|completed|partial|aborted

    # --- Per-step scratch space (reset whenever a step completes) -------------
    selected_model: str
    tried_models: list[str]
    model_failures: dict[str, int]         # consecutive exception count (circuit breaker)
    last_output: str
    last_error: str
    last_error_kind: str                   # "exception" | "verification" | ""
    verifier_feedback: str
    degraded: bool
    human_action: str


def initial_state(goal: str) -> OrchestratorState:
    return OrchestratorState(
        user_goal=goal, plan=[], current_step=0, results={}, errors=[],
        fallback_count=0, final_output="", handoff_instructions="", status="planning",
        selected_model="", tried_models=[], model_failures={}, last_output="",
        last_error="", last_error_kind="", verifier_feedback="", degraded=False,
        human_action="",
    )


# ============================================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# ============================================================================
StepKind = Literal["research", "analysis", "draft", "tool_use", "general"]


class PlanStep(BaseModel):
    title: str = Field(description="Short step title")
    instruction: str = Field(description="Precise instruction for the specialist executing this step")
    kind: StepKind = Field(default="general", description="Type of specialist needed")
    complexity: Literal["low", "high"] = Field(
        default="high", description="'low' = simple/cheap model is fine; 'high' = needs a strong model")
    success_criteria: str = Field(description="How a verifier can tell this step succeeded")
    tool: str = Field(default="", description="Name of a tool to call first, or empty string for none")
    tool_input: str = Field(default="", description="Input for the tool, or empty string")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description=f"Ordered steps, at most {MAX_PLAN_STEPS}")


class Verdict(BaseModel):
    passed: bool = Field(description="True if the output satisfies the step's success criteria")
    score: float = Field(description="Quality score from 0.0 to 1.0")
    issues: list[str] = Field(default_factory=list, description="Concrete problems found")
    feedback: str = Field(default="", description="One or two sentences on how to improve the output")


# ============================================================================
# 3. MODEL REGISTRY  (add a new model/provider = add one ModelSpec)
# ============================================================================
@dataclass(frozen=True)
class ModelSpec:
    name: str                 # unique id used in state and logs
    provider: str             # "anthropic" | "openai" | "openrouter"
    model_id: str
    tier: int                 # 0 = strong/expensive, 1 = fast/cheap
    api_key_env: str
    base_url: str | None = None

    @property
    def available(self) -> bool:
        pkg = {"anthropic": "langchain_anthropic"}.get(self.provider, "langchain_openai")
        return bool(os.getenv(self.api_key_env)) and importlib.util.find_spec(pkg) is not None


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# Order matters: within a tier, earlier entries are preferred, and providers alternate
# so the first fallback after a failure is usually a *different* provider.
MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("anthropic-strong", "anthropic", _env("ANTHROPIC_STRONG_MODEL", "claude-sonnet-5-5"), 0, "ANTHROPIC_API_KEY"),
    ModelSpec("openai-strong", "openai", _env("OPENAI_STRONG_MODEL", "gpt-4.1"), 0, "OPENAI_API_KEY"),
    ModelSpec("anthropic-fast", "anthropic", _env("ANTHROPIC_FAST_MODEL", "claude-haiku-5-5"), 1, "ANTHROPIC_API_KEY"),
    ModelSpec("openai-fast", "openai", _env("OPENAI_FAST_MODEL", "gpt-4.1-mini"), 1, "OPENAI_API_KEY"),
    ModelSpec("openrouter-fast", "openrouter", _env("OPENROUTER_FAST_MODEL", "google/gemini-2.0-flash-001"), 1,
              "OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1"),
]
SPEC_BY_NAME = {m.name: m for m in MODEL_REGISTRY}
_MODEL_CACHE: dict[str, Any] = {}


def get_chat_model(spec: ModelSpec):
    """Lazily build (and cache) a LangChain chat model. Imports are lazy so a missing
    provider package never breaks the other providers."""
    if spec.name in _MODEL_CACHE:
        return _MODEL_CACHE[spec.name]
    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=spec.model_id, max_tokens=4096, timeout=REQUEST_TIMEOUT_S, max_retries=1)
    else:  # openai and any OpenAI-compatible endpoint (OpenRouter, vLLM, Together, ...)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=spec.model_id, api_key=os.getenv(spec.api_key_env), base_url=spec.base_url,
                         timeout=REQUEST_TIMEOUT_S, max_retries=1)
    _MODEL_CACHE[spec.name] = llm
    return llm


def rank_models(complexity: str) -> list[ModelSpec]:
    """Preference order of *available* models for a step.
    high -> strong tier first, then fast tier (degradation = cheaper/faster).
    low  -> fast tier first (cheap), then strong tier as a quality escalation."""
    avail = [m for m in MODEL_REGISTRY if m.available]
    return sorted(avail, key=lambda m: m.tier if complexity == "high" else -m.tier)  # stable sort


def _text(response) -> str:
    c = response.content
    if isinstance(c, str):
        return c.strip()
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in c
                   if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")).strip()


def call_text(spec: ModelSpec, messages: list) -> str:
    return _text(get_chat_model(spec).invoke(messages))


def call_structured(spec: ModelSpec, schema: type[BaseModel], messages: list):
    """Structured output via tool/function calling, which works across providers."""
    llm = get_chat_model(spec)
    kwargs = {"method": "function_calling"} if spec.provider in ("openai", "openrouter") else {}
    out = llm.with_structured_output(schema, **kwargs).invoke(messages)
    if out is None:
        raise ValueError("Model returned no structured output")
    return out


# ============================================================================
# 4. TOOLS  (add a tool = decorate one function)
# ============================================================================
TOOLS: dict[str, tuple[Callable[[str], str], str]] = {}


def register_tool(name: str, description: str):
    def deco(fn: Callable[[str], str]):
        TOOLS[name] = (fn, description)
        return fn
    return deco


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _safe_eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("Unsupported expression")


@register_tool("calculator", "Evaluate an arithmetic expression, e.g. '(1200 * 0.15) / 12'")
def calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression.strip(), mode="eval").body))


@register_tool("current_time", "Return the current UTC date and time (input ignored)")
def current_time(_: str) -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_tool(name: str, tool_input: str) -> str:
    if name not in TOOLS:
        raise KeyError(f"Unknown tool '{name}'")
    return TOOLS[name][0](tool_input)


# Specialist system prompts (add a new kind here + to StepKind to extend)
SPECIALISTS: dict[str, str] = {
    "research": "You are a meticulous research specialist. Give accurate, well-organized findings. "
                "State clearly when you are unsure or lack current data; never invent sources or figures.",
    "analysis": "You are a rigorous analyst. Reason step by step, show key numbers and assumptions, "
                "and finish with clear conclusions.",
    "draft": "You are an excellent writer. Produce polished, ready-to-use text that follows the instruction exactly.",
    "tool_use": "You are an assistant that interprets tool output. Use the provided tool result faithfully "
                "and explain what it means for the task.",
    "general": "You are a capable, concise assistant. Complete the instruction precisely.",
}


# ============================================================================
# 5. HELPERS
# ============================================================================
def _clip(text: str, n: int = 1200) -> str:
    return text if len(text) <= n else text[:n] + " …[truncated]"


def _context_from_results(state: OrchestratorState) -> str:
    """Condensed summary of earlier step results, passed to later steps."""
    parts = []
    for step in state["plan"][: state["current_step"]]:
        r = state["results"].get(step["id"])
        if r and r.get("output"):
            parts.append(f"### {step['title']} ({r['status']})\n{_clip(r['output'])}")
    return "\n\n".join(parts) or "(none yet)"


def _advance(state: OrchestratorState) -> dict:
    """Move to the next step and reset all per-step scratch fields."""
    return {"current_step": state["current_step"] + 1, "selected_model": "", "tried_models": [],
            "last_output": "", "last_error": "", "last_error_kind": "", "verifier_feedback": "",
            "degraded": False, "status": "running"}


def _prepare_handoff(state: OrchestratorState, reason: str) -> dict:
    """Build plain-language instructions and flag the run as needing a human.
    Stored in state *before* the interrupt so it is checkpointed and inspectable."""
    idx, plan = state["current_step"], state["plan"]
    step = plan[idx]
    instructions = (
        f"Step {idx + 1} of {len(plan)} (\"{step['title']}\") could not be completed automatically.\n"
        f"Reason: {reason}\n\n"
        "How to continue, reply with one of:\n"
        "  1. provide_result: do this step yourself, e.g. {\"action\": \"provide_result\", \"content\": \"...\"}\n"
        "  2. retry:          try the automated models again, e.g. {\"action\": \"retry\"}\n"
        "  3. skip:           continue without this step, e.g. {\"action\": \"skip\"}\n"
        "  4. abort:          stop and package what exists, e.g. {\"action\": \"abort\"}\n"
        f"\nStep instruction: {step['instruction']}\nSuccess criteria: {step.get('success_criteria', '')}"
    )
    return {"status": "needs_human", "handoff_instructions": instructions}


# ============================================================================
# 6. NODES
# ============================================================================
def planner_node(state: OrchestratorState) -> dict:
    """Break the goal into ordered steps using the strongest available model, trying fallbacks."""
    tool_list = "\n".join(f"- {n}: {d}" for n, (_, d) in TOOLS.items())
    messages = [
        SystemMessage(content=(
            f"You are a planning agent. Break the user's goal into 2-{MAX_PLAN_STEPS} ordered, self-contained "
            "steps. Mark each step 'low' complexity if a small, fast model can do it. Only use a tool when it "
            f"clearly helps. Available tools:\n{tool_list}")),
        HumanMessage(content=state["user_goal"]),
    ]
    errors: list[str] = []
    for spec in rank_models("high"):
        try:
            plan: Plan = call_structured(spec, Plan, messages)
            steps = []
            for i, s in enumerate(plan.steps[:MAX_PLAN_STEPS], start=1):
                d = s.model_dump()
                d["id"] = f"s{i}"
                if d["tool"] and d["tool"] not in TOOLS:
                    d["tool"], d["tool_input"] = "", ""       # ignore hallucinated tools
                steps.append(d)
            if not steps:
                raise ValueError("Planner returned zero steps")
            log.info("Planner (%s) produced %d steps", spec.name, len(steps))
            return {"plan": steps, "current_step": 0, "status": "running", "errors": errors}
        except Exception as e:  # noqa: BLE001
            errors.append(f"[planner] {spec.name} failed: {type(e).__name__}: {e}")
            log.warning(errors[-1])
    # Degraded plan: treat the whole goal as one step rather than failing outright.
    fallback = [{"id": "s1", "title": "Complete the goal", "instruction": state["user_goal"], "kind": "general",
                 "complexity": "high", "success_criteria": "The goal is fully and accurately addressed.",
                 "tool": "", "tool_input": ""}]
    errors.append("[planner] all models failed; using single-step fallback plan")
    return {"plan": fallback, "current_step": 0, "status": "running", "errors": errors}


def router_node(state: OrchestratorState) -> dict:
    """Pick the best available model for the current step: not yet tried on this step,
    and not tripped by the circuit breaker."""
    step = state["plan"][state["current_step"]]
    tried = list(state.get("tried_models", []))
    failures = state.get("model_failures", {})
    candidates = [m for m in rank_models(step.get("complexity", "high"))
                  if m.name not in tried and failures.get(m.name, 0) < CIRCUIT_BREAKER_THRESHOLD]
    if not candidates:
        return _prepare_handoff(state, "No remaining healthy models to try for this step.")
    chosen = candidates[0]
    log.info("Router: step %d -> %s (attempt %d)", state["current_step"] + 1, chosen.name, len(tried) + 1)
    return {"selected_model": chosen.name, "tried_models": tried + [chosen.name], "status": "running"}


def _maybe_simulate_failure(step_index: int) -> None:
    """Demo hook: SIMULATE_FAIL_STEP=2 makes every model fail on step 2 (1-based)."""
    if os.getenv("SIMULATE_FAIL_STEP") == str(step_index + 1):
        raise RuntimeError("Simulated provider outage")


def executor_node(state: OrchestratorState) -> dict:
    """Run the current step with the routed model. Never raises: failures are recorded
    in state so the graph can degrade gracefully."""
    idx = state["current_step"]
    step = state["plan"][idx]
    spec = SPEC_BY_NAME[state["selected_model"]]
    new_errors: list[str] = []
    try:
        _maybe_simulate_failure(idx)

        # Optional tool call. If the tool fails we continue without it (alternative approach).
        tool_block = ""
        if step.get("tool"):
            try:
                tool_block = f"\n\nTool '{step['tool']}' result:\n{run_tool(step['tool'], step.get('tool_input') or step['instruction'])}"
            except Exception as e:  # noqa: BLE001
                new_errors.append(f"[step {idx + 1}] tool '{step['tool']}' failed: {e}")
                tool_block = f"\n\n(Tool '{step['tool']}' failed: {e}. Do your best without it.)"

        extra = ""
        if state.get("verifier_feedback"):
            extra += f"\n\nA previous attempt was rejected. Reviewer feedback: {state['verifier_feedback']}"
        if state.get("degraded"):
            extra += "\n\nPrevious attempts failed. Be concise and focus on the core deliverable."

        messages = [
            SystemMessage(content=SPECIALISTS.get(step["kind"], SPECIALISTS["general"])),
            HumanMessage(content=(
                f"Overall goal: {state['user_goal']}\n\n"
                f"Completed so far:\n{_context_from_results(state)}\n\n"
                f"Your step ({idx + 1}/{len(state['plan'])}): {step['title']}\n"
                f"Instruction: {step['instruction']}\n"
                f"Success criteria: {step['success_criteria']}{tool_block}{extra}")),
        ]
        text = call_text(spec, messages)
        if not text:
            raise ValueError("Model returned empty output")
        return {"last_output": text, "last_error": "", "last_error_kind": "", "errors": new_errors}
    except Exception as e:  # noqa: BLE001
        msg = f"[step {idx + 1}] {spec.name} failed: {type(e).__name__}: {e}"
        log.warning(msg)
        return {"last_error": msg, "last_error_kind": "exception", "errors": new_errors + [msg]}


def verifier_node(state: OrchestratorState) -> dict:
    """Check the output with a cheap model (structured verdict). On pass, record the result
    and advance. If the verifier itself is unavailable, fall back to a basic heuristic."""
    idx = state["current_step"]
    step = state["plan"][idx]
    output = state["last_output"]
    verdict: Verdict | None = None
    new_errors: list[str] = []

    if len(output.strip()) < 20:
        verdict = Verdict(passed=False, score=0.0, issues=["Output is empty or too short"],
                          feedback="Produce a complete answer.")
    else:
        messages = [
            SystemMessage(content="You are a strict but fair QA reviewer. Judge whether the output satisfies "
                                  "the instruction and success criteria. Do not penalize reasonable brevity."),
            HumanMessage(content=(f"Instruction: {step['instruction']}\nSuccess criteria: {step['success_criteria']}\n\n"
                                  f"Output to review:\n{_clip(output, 6000)}")),
        ]
        for spec in rank_models("low")[:3]:
            try:
                verdict = call_structured(spec, Verdict, messages)
                break
            except Exception as e:  # noqa: BLE001
                new_errors.append(f"[verifier] {spec.name} failed: {type(e).__name__}: {e}")
        if verdict is None:  # every verifier failed: accept with a warning rather than block progress
            new_errors.append(f"[step {idx + 1}] verifier unavailable; accepted by heuristic")
            verdict = Verdict(passed=True, score=VERIFY_PASS_SCORE, feedback="Unverified (heuristic accept)")

    if verdict.passed and verdict.score >= VERIFY_PASS_SCORE:
        failures = dict(state.get("model_failures", {}))
        failures[state["selected_model"]] = 0   # success closes the circuit breaker
        results = dict(state["results"])
        results[step["id"]] = {"title": step["title"], "output": output, "model": state["selected_model"],
                               "status": "ok", "score": verdict.score}
        return {"results": results, "model_failures": failures, "errors": new_errors, **_advance(state)}

    msg = (f"[step {idx + 1}] verification failed ({state['selected_model']}, score={verdict.score:.2f}): "
           f"{'; '.join(verdict.issues) or verdict.feedback}")
    log.warning(msg)
    return {"last_error": msg, "last_error_kind": "verification", "verifier_feedback": verdict.feedback,
            "errors": new_errors + [msg]}


def fallback_handler_node(state: OrchestratorState) -> dict:
    """Degradation: count the failure, update the circuit breaker, then either send the step
    back to the router (next model / simpler prompt) or escalate to a human when limits are hit."""
    failed = state["selected_model"]
    total = state.get("fallback_count", 0) + 1
    updates: dict[str, Any] = {"fallback_count": total, "degraded": True, "status": "degrading"}

    if state.get("last_error_kind") == "exception":
        failures = dict(state.get("model_failures", {}))
        failures[failed] = failures.get(failed, 0) + 1
        updates["model_failures"] = failures

    step_failures = len(state.get("tried_models", []))
    if step_failures > MAX_FALLBACKS_PER_STEP or total >= MAX_TOTAL_FALLBACKS:
        reason = (f"Fallback limit reached ({step_failures} failed attempts on this step, {total} total). "
                  f"Last problem: {state.get('last_error', 'unknown')}")
        updates.update(_prepare_handoff(state, reason))
    log.info("Fallback #%d after %s failure on %s -> %s", total, state.get("last_error_kind"), failed, updates["status"])
    return updates


def human_handoff_node(state: OrchestratorState) -> dict:
    """Pause the graph. The payload is returned to the caller; resume later with
    Command(resume={"action": ..., "content": ...}).

    NOTE: on resume LangGraph re-runs this node from the top, so everything before
    interrupt() must be free of side effects."""
    idx = state["current_step"]
    step = state["plan"][idx]
    decision = interrupt({
        "message": state["handoff_instructions"],
        "step": {"number": idx + 1, "total": len(state["plan"]), "title": step["title"]},
        "partial_results": {sid: {"title": r["title"], "status": r["status"], "output": _clip(r["output"], 2000)}
                            for sid, r in state["results"].items()},
        "last_attempt_output": _clip(state.get("last_output", ""), 2000),
        "recent_errors": state.get("errors", [])[-5:],
        "resume_options": ["provide_result", "retry", "skip", "abort"],
    })

    # Accept a bare string as "provide_result" for convenience.
    if isinstance(decision, str):
        decision = {"action": "provide_result", "content": decision}
    action = (decision or {}).get("action", "abort") if isinstance(decision, dict) else "abort"
    content = (decision or {}).get("content", "") if isinstance(decision, dict) else ""
    if action == "provide_result" and not content.strip():
        action = "abort"

    results = dict(state["results"])
    base = {"human_action": action, "handoff_instructions": ""}
    partial = state.get("last_output", "")

    if action == "provide_result":
        results[step["id"]] = {"title": step["title"], "output": content, "model": "human", "status": "human", "score": 1.0}
        return {"results": results, **base, **_advance(state)}
    if action == "skip":
        if partial:
            results[step["id"]] = {"title": step["title"], "output": partial, "model": state.get("selected_model", ""),
                                   "status": "unverified", "score": 0.0}
        else:
            results[step["id"]] = {"title": step["title"], "output": "", "model": "", "status": "skipped", "score": 0.0}
        return {"results": results, **base, **_advance(state)}
    if action == "retry":
        # Human-approved retry: give this step a fresh budget and re-enable all models.
        return {**base, "tried_models": [], "model_failures": {}, "degraded": False, "last_error": "",
                "last_error_kind": "", "status": "running", "fallback_count": 0}
    # abort (also the default for anything unrecognized)
    if partial:
        results[step["id"]] = {"title": step["title"], "output": partial, "model": state.get("selected_model", ""),
                               "status": "unverified", "score": 0.0}
    return {"results": results, "status": "aborted", "human_action": "abort", "handoff_instructions": ""}


def finalizer_node(state: OrchestratorState) -> dict:
    """Package the complete result, or the best partial result plus handoff notes."""
    plan, results = state["plan"], state["results"]
    good = {"ok", "human"}
    notes: list[str] = []
    sections: list[str] = []
    for i, step in enumerate(plan, start=1):
        r = results.get(step["id"])
        if r is None:
            notes.append(f"Step {i} (\"{step['title']}\") was not run.")
            continue
        if r["status"] == "unverified":
            notes.append(f"Step {i} (\"{step['title']}\") is an unverified partial result: please review.")
        elif r["status"] == "skipped":
            notes.append(f"Step {i} (\"{step['title']}\") was skipped and has no output.")
        elif r["status"] == "human":
            notes.append(f"Step {i} (\"{step['title']}\") was completed by a human.")
        if r["output"]:
            sections.append(f"## {i}. {step['title']} [{r['status']}]\n\n{r['output']}")

    complete = all(results.get(s["id"], {}).get("status") in good | {"ok"} for s in plan)
    if state.get("status") == "aborted":
        status = "aborted"
    else:
        status = "completed" if complete else "partial"

    body = "\n\n".join(sections) or "(no output was produced)"
    # For fully successful runs, ask a strong model for a cohesive final deliverable (best-effort).
    if complete and len(plan) > 1:
        messages = [
            SystemMessage(content="Combine the step results into one cohesive final deliverable that fully answers "
                                  "the goal. Remove redundancy; keep all important facts and numbers."),
            HumanMessage(content=f"Goal: {state['user_goal']}\n\nStep results:\n{body}"),
        ]
        for spec in rank_models("high")[:2]:
            try:
                body = call_text(spec, messages) or body
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Finalizer synthesis with %s failed: %s", spec.name, e)

    output = f"# Result for: {state['user_goal']}\n\nStatus: {status}\n\n{body}"
    if notes:
        output += "\n\n---\n## Handoff notes\n" + "\n".join(f"- {n}" for n in notes)
    return {"final_output": output, "handoff_instructions": "\n".join(notes), "status": status}


# ============================================================================
# 7. CONDITIONAL EDGES
# ============================================================================
def route_after_router(state: OrchestratorState) -> str:
    return "human_handoff" if state.get("status") == "needs_human" else "executor"


def route_after_executor(state: OrchestratorState) -> str:
    return "fallback_handler" if state.get("last_error") else "verifier"


def route_after_verifier(state: OrchestratorState) -> str:
    if state.get("last_error"):                         # failed verification -> degrade/retry
        return "fallback_handler"
    if state["current_step"] >= len(state["plan"]):     # all steps done
        return "finalizer"
    return "router"                                     # next step


def route_after_handler(state: OrchestratorState) -> str:
    return "human_handoff" if state.get("status") == "needs_human" else "router"


def route_after_handoff(state: OrchestratorState) -> str:
    action = state.get("human_action")
    if action == "abort":
        return "finalizer"
    if action == "retry":
        return "router"
    return "finalizer" if state["current_step"] >= len(state["plan"]) else "router"


# ============================================================================
# 8. GRAPH ASSEMBLY
# ============================================================================
def build_graph(checkpointer=None):
    g = StateGraph(OrchestratorState)
    g.add_node("planner", planner_node)
    g.add_node("router", router_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("fallback_handler", fallback_handler_node)
    g.add_node("human_handoff", human_handoff_node)
    g.add_node("finalizer", finalizer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "router")
    g.add_conditional_edges("router", route_after_router, ["executor", "human_handoff"])
    g.add_conditional_edges("executor", route_after_executor, ["verifier", "fallback_handler"])
    g.add_conditional_edges("verifier", route_after_verifier, ["fallback_handler", "router", "finalizer"])
    g.add_conditional_edges("fallback_handler", route_after_handler, ["router", "human_handoff"])
    g.add_conditional_edges("human_handoff", route_after_handoff, ["router", "finalizer"])
    g.add_edge("finalizer", END)
    return g.compile(checkpointer=checkpointer)


def make_checkpointer():
    """InMemorySaver for demos (state is lost when the process exits).

    PRODUCTION (Postgres) - pip install langgraph-checkpoint-postgres "psycopg[binary,pool]"

        from langgraph.checkpoint.postgres import PostgresSaver
        DB_URI = "postgresql://user:pass@host:5432/dbname?sslmode=disable"
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()               # run once to create tables
            graph = build_graph(checkpointer)
            ...                                # invoke / resume as below, inside this block

    With a persistent saver, a paused run can be resumed hours later, even from a different
    process, using the same thread_id.
    """
    return InMemorySaver()


# ============================================================================
# 9. DEMO RUNNER
# ============================================================================
def pending_interrupts(graph, config) -> list[Any]:
    snap = graph.get_state(config)
    return [i for t in snap.tasks for i in t.interrupts]


def run_until_pause(graph, graph_input, config) -> None:
    """Stream node-by-node progress. Returns when the graph finishes OR pauses on an interrupt."""
    for chunk in graph.stream(graph_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue
            detail = ""
            if isinstance(update, dict):
                detail = f"status={update.get('status', '')}".strip()
            print(f"  -> {node:<17} {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-AI orchestrator demo")
    parser.add_argument("--goal", default=(
        "Evaluate whether a small cafe selling 120 coffees/day at $4.50 each, with $2,100 in monthly "
        "fixed costs and 35% ingredient cost, is profitable. Calculate the monthly profit, explain the key "
        "risks, and draft a short 3-sentence recommendation for the owner."))
    parser.add_argument("--simulate-fail-step", type=int, default=0,
                        help="Force every model to fail on this step (1-based) to demo degradation + human handoff")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Resume interrupts automatically with canned input instead of prompting")
    parser.add_argument("--thread-id", default=None)
    args = parser.parse_args()

    if args.simulate_fail_step:
        os.environ["SIMULATE_FAIL_STEP"] = str(args.simulate_fail_step)

    available = [m.name for m in MODEL_REGISTRY if m.available]
    if not available:
        raise SystemExit("No models available. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY / OPENROUTER_API_KEY "
                         "and install the matching langchain packages.")
    print(f"Available models: {', '.join(available)}\n")

    graph = build_graph(make_checkpointer())
    config = {"configurable": {"thread_id": args.thread_id or str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}

    print(f"GOAL: {args.goal}\n")
    run_until_pause(graph, initial_state(args.goal), config)

    # Keep resuming for as long as the graph is paused on a human handoff.
    while (pending := pending_interrupts(graph, config)):
        payload = pending[0].value
        print("\n" + "=" * 70 + "\nHUMAN HANDOFF REQUIRED\n" + "=" * 70)
        print(payload["message"])
        if payload["partial_results"]:
            print("\nPartial results so far:")
            for r in payload["partial_results"].values():
                print(f"  - {r['title']} [{r['status']}]")
        print()

        if args.auto_resume:
            # Simulates a human completing the step by hand.
            resume = {"action": "provide_result", "content": "(Result supplied by a human reviewer for this step.)"}
            print(f"[auto-resume] {json.dumps(resume)}")
            os.environ.pop("SIMULATE_FAIL_STEP", None)
        else:
            raw = input("Your reply (JSON, or plain text to supply the result, 'skip'/'retry'/'abort'): ").strip()
            if raw in ("skip", "retry", "abort"):
                resume = {"action": raw}
            else:
                try:
                    resume = json.loads(raw)
                except json.JSONDecodeError:
                    resume = {"action": "provide_result", "content": raw}
            if resume.get("action") == "retry":
                os.environ.pop("SIMULATE_FAIL_STEP", None)

        print("\nResuming...\n")
        run_until_pause(graph, Command(resume=resume), config)

    final = graph.get_state(config).values
    print("\n" + "=" * 70 + f"\nFINAL STATUS: {final['status']}   (fallbacks used: {final['fallback_count']})\n" + "=" * 70)
    print(final["final_output"])
    if final.get("errors"):
        print(f"\n{len(final['errors'])} error(s) were handled along the way; last 3:")
        for e in final["errors"][-3:]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
