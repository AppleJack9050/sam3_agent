"""SAM3Agent: thin facade over the deterministic pipeline in `pipeline.py`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .config import AgentConfig
from .pipeline import run_pipeline
from .quality import QualityReport
from .sam3_inference import SAM3Predictor


@dataclass
class AgentResult:
    output_paths: dict
    coverage: float
    attempts: int
    final_prompts: List[str]
    mode: str
    quality: QualityReport
    history: list = field(default_factory=list)


class SAM3Agent:
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        predictor: Optional[SAM3Predictor] = None,
    ):
        self.cfg = config or AgentConfig()
        self.predictor = predictor or SAM3Predictor(self.cfg.model_name, self.cfg.device)

    def run(self, image_path: str | Path, out_path: str | Path) -> AgentResult:
        final = run_pipeline(
            self.predictor, str(image_path), str(out_path), self.cfg,
        )
        quality: QualityReport = final["quality"]
        return AgentResult(
            output_paths=final["output_paths"],
            coverage=quality.coverage,
            attempts=final["attempt"],
            final_prompts=list(final.get("prompts", [])),
            mode=self.cfg.mode,
            quality=quality,
            history=final.get("history", []),
        )

    def run_batch(self, pairs: List[Tuple[str, str]]) -> List[AgentResult]:
        return [self.run(p, o) for p, o in pairs]
