import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import analyze_text, get_demo_feed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Pnyx Lens", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


class OverlayResponse(BaseModel):
    analysis: dict


class FeedResponse(BaseModel):
    posts: list[dict]


@app.get("/feed/demo", response_model=FeedResponse)
async def feed_demo():
    return FeedResponse(posts=get_demo_feed())


@app.post("/overlay/analyze", response_model=OverlayResponse)
async def overlay_analyze(req: AnalyzeRequest):
    return OverlayResponse(analysis=analyze_text(req.text))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "product": "Pnyx Lens",
        "keys": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
    }
