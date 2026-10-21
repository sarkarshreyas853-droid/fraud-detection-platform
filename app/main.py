"""FastAPI scoring service.   Run:  uvicorn app.main:app --port 8000"""
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from app.scoring import Scorer

state = {}


@asynccontextmanager
async def lifespan(app):
    state["scorer"] = Scorer()
    yield


app = FastAPI(title="Fraud Scoring Service", version="0.1", lifespan=lifespan)


class Transaction(BaseModel):
    tx_id: Optional[int] = None
    step: int = Field(..., ge=0, description="hour of simulated time")
    type: str = Field(..., description="TRANSFER or CASH_OUT")
    amount: float = Field(..., ge=0)
    nameDest: str
    label: Optional[int] = Field(None, description="true label, only used when replaying known data")


@app.get("/health")
def health():
    s = state["scorer"]
    return {"status": "ok", "model": s.card["run"], "features": s.features, "threshold": s.threshold}


@app.get("/stats")
def stats():
    return state["scorer"].stats


@app.post("/score")
def score(tx: Transaction):
    result = state["scorer"].score(tx.model_dump())
    record = {**result, "label": tx.label, "amount": tx.amount, "type": tx.type, "nameDest": tx.nameDest}
    with open(config.SCORED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return result
