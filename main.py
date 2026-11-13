from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
from multi_ai_orchestrator import build_graph, make_checkpointer, initial_state, Command

app = FastAPI(title="Multi-AI Orchestrator")

# One graph instance (InMemorySaver is fine for testing; state is lost on restart)
graph = build_graph(make_checkpointer())

class StartRequest(BaseModel):
    goal: str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    action: str  # "provide_result", "retry", "skip", "abort"
    content: Optional[str] = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-AI Orchestrator API is running"}

@app.post("/start")
def start(req: StartRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # Run until it finishes or hits a human handoff
    result = graph.invoke(initial_state(req.goal), config)
    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "final_output": result.get("final_output"),
        "handoff_instructions": result.get("handoff_instructions"),
        "errors": result.get("errors", [])[-3:],
    }

@app.post("/resume")
def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    resume_value = {"action": req.action, "content": req.content or ""}
    result = graph.invoke(Command(resume=resume_value), config)
    return {
        "thread_id": req.thread_id,
        "status": result.get("status"),
        "final_output": result.get("final_output"),
        "handoff_instructions": result.get("handoff_instructions"),
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
