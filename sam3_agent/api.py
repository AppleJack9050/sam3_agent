"""FastAPI wrapper for single and batch segmentation jobs."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .agent import SAM3Agent
from .config import AgentConfig

app = FastAPI(title="SAM3 Segmentation Agent")


class RunResponse(BaseModel):
    output_paths: dict
    coverage: float
    attempts: int
    mode: str
    final_prompts: List[str]
    quality_ok: bool
    quality_reasons: List[str]
    history: List[dict]


@app.post("/segment", response_model=RunResponse)
async def segment(
    file: UploadFile = File(...),
    exemplar: Optional[UploadFile] = File(None),
    mode: str = Form("exclude"),
    preserve: Optional[str] = Form(None),
    exclude_prompts: Optional[str] = Form(None),  # comma-separated
    output_format: str = Form("png"),
    use_llm: bool = Form(False),
):
    tmp = Path(tempfile.mkdtemp(prefix="sam3_"))
    try:
        in_path = tmp / (file.filename or "input.png")
        with in_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        exemplar_path: Optional[str] = None
        if exemplar is not None:
            ex_path = tmp / (exemplar.filename or "exemplar.png")
            with ex_path.open("wb") as f:
                shutil.copyfileobj(exemplar.file, f)
            exemplar_path = str(ex_path)

        kwargs = dict(
            mode=mode,
            output_format=output_format,
            use_llm=use_llm,
            exemplar_image=exemplar_path,
        )
        if exclude_prompts:
            kwargs["exclude_prompts"] = tuple(p.strip() for p in exclude_prompts.split(",") if p.strip())
        if preserve:
            if mode == "include":
                kwargs["target"] = preserve
            else:
                kwargs["preserve_label"] = preserve

        out_path = tmp / f"out_{uuid.uuid4().hex}"
        cfg = AgentConfig(**kwargs)
        agent = SAM3Agent(cfg)
        result = agent.run(in_path, out_path)
        return RunResponse(
            output_paths=result.output_paths,
            coverage=result.coverage,
            attempts=result.attempts,
            mode=result.mode,
            final_prompts=result.final_prompts,
            quality_ok=result.quality.ok,
            quality_reasons=result.quality.reasons,
            history=result.history,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


class BatchItem(BaseModel):
    input_path: str
    output_path: str


class BatchRequest(BaseModel):
    items: List[BatchItem]
    mode: str = "exclude"
    preserve: Optional[str] = None
    exclude_prompts: Optional[List[str]] = None
    output_format: str = "png"
    use_llm: bool = False
    exemplar_image: Optional[str] = None


@app.post("/segment/batch")
def segment_batch(req: BatchRequest):
    kwargs = dict(
        mode=req.mode,
        output_format=req.output_format,
        use_llm=req.use_llm,
        exemplar_image=req.exemplar_image,
    )
    if req.exclude_prompts:
        kwargs["exclude_prompts"] = tuple(req.exclude_prompts)
    if req.preserve:
        if req.mode == "include":
            kwargs["target"] = req.preserve
        else:
            kwargs["preserve_label"] = req.preserve

    cfg = AgentConfig(**kwargs)
    agent = SAM3Agent(cfg)
    pairs = [(i.input_path, i.output_path) for i in req.items]
    results = agent.run_batch(pairs)
    return [
        {
            "output_paths": r.output_paths,
            "coverage": r.coverage,
            "attempts": r.attempts,
            "mode": r.mode,
            "final_prompts": r.final_prompts,
            "quality_ok": r.quality.ok,
            "history": r.history,
        }
        for r in results
    ]


@app.get("/healthz")
def healthz():
    return {"ok": True}
