import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import json

from src.main import run_hedge_fund
from src.web.chat_graph import chat_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# Mount static files for HTML/CSS/JS
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# In-memory store for session contexts
session_contexts = {}

class RunHedgeFundRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    initial_cash: float
    model_name: str
    model_provider: str
    selected_analysts: list[str]

@app.post("/api/run")
def api_run_hedge_fund(req: RunHedgeFundRequest):
    try:
        portfolio = {
            "cash": req.initial_cash,
            "margin_requirement": 0.0,
            "margin_used": 0.0,
            "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for t in req.tickers},
            "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in req.tickers},
        }

        result = run_hedge_fund(
            tickers=req.tickers,
            start_date=req.start_date,
            end_date=req.end_date,
            portfolio=portfolio,
            show_reasoning=True,
            selected_analysts=req.selected_analysts,
            model_name=req.model_name,
            model_provider=req.model_provider,
        )
        
        session_id = str(uuid.uuid4())
        session_contexts[session_id] = {
            "data": {
                "analyst_signals": result["analyst_signals"],
                "decisions": result["decisions"],
                "portfolio": portfolio,
            },
            "metadata": {
                "model_name": req.model_name,
                "model_provider": req.model_provider,
            }
        }
        
        return {"session_id": session_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if req.session_id not in session_contexts:
        raise HTTPException(status_code=404, detail="Session not found. Please run analysis first.")
        
    ctx = session_contexts[req.session_id]
    
    config = {"configurable": {"thread_id": req.session_id}}
    
    response = chat_agent.invoke(
        {
            "messages": [HumanMessage(content=req.message)],
            "data": ctx["data"],
            "metadata": ctx["metadata"],
        },
        config=config
    )
    
    ai_msg = response["messages"][-1]
    return {"reply": ai_msg.content}
