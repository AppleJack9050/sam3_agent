"""SAM 3 wrapper.

Uses the real `facebookresearch/sam3` Python API:

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.5)

    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(prompt="glacier", state=state)
    # state["masks"]  -> (N, 1, H, W) bool at original resolution
    # state["scores"] -> (N,) float

A `MockSAM3Predictor` covers the same interface for CPU-only tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    mask: np.ndarray  # HxW bool
    score: float
    label: str


class SAM3Predictor:
    def __init__(self, model_name: str = "facebook/sam3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        if self._processor is not None:
            return

        if self.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA backend requested but torch.cuda.is_available() is False. "
                    "Install a CUDA 12.8+ PyTorch build with Blackwell (sm_120) support: "
                    "pip install --extra-index-url https://download.pytorch.org/whl/cu128 "
                    "'torch>=2.7' torchvision"
                )
            cap = torch.cuda.get_device_capability(0)
            if cap < (12, 0):
                import warnings

                warnings.warn(
                    f"Detected CUDA device capability sm_{cap[0]}{cap[1]}; "
                    "expected sm_120 (Blackwell / RTX 5090).",
                    stacklevel=2,
                )

        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "SAM 3 is not installed. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam3.git\n"
                "  pip install einops hydra-core pycocotools psutil av decord timm ftfy regex matplotlib scipy\n"
                f"Original error: {e}"
            ) from e

        self._model = build_sam3_image_model(device=self.device, load_from_HF=True)
        self._processor = Sam3Processor(self._model, device=self.device, confidence_threshold=0.0)

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        exemplar_image: Optional[np.ndarray] = None,
        score_threshold: float = 0.5,
    ) -> List[Detection]:
        """Run SAM 3 with an open-vocabulary text prompt.

        Note: exemplar (visual-prompt) support is not wired through the
        image-model processor; if an exemplar is provided it is ignored with
        a warning. Use a more specific `text_prompt` to disambiguate instead.
        """
        from PIL import Image

        self.load()

        if exemplar_image is not None:
            import warnings

            warnings.warn(
                "Exemplar prompts are not supported by the SAM 3 image processor; "
                "ignoring exemplar and falling back to text-only prompting.",
                stacklevel=2,
            )

        pil = Image.fromarray(image) if not isinstance(image, Image.Image) else image

        import torch

        # SAM 3 checkpoints ship in bf16; wrap in autocast so float32 inputs
        # from the processor match the model's parameter dtype.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        )

        # Let the processor keep all predictions; we apply score_threshold ourselves.
        self._processor.confidence_threshold = 0.0
        with autocast_ctx:
            state = self._processor.set_image(pil)
            state = self._processor.set_text_prompt(prompt=text_prompt, state=state)

        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None or len(masks) == 0:
            return []

        masks_np = masks.detach().to(torch.bool).cpu().numpy()
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        scores_np = scores.detach().float().cpu().numpy()

        detections: List[Detection] = []
        for m, s in zip(masks_np, scores_np):
            if float(s) >= score_threshold:
                detections.append(
                    Detection(mask=m.astype(bool), score=float(s), label=text_prompt)
                )
        return detections


class MockSAM3Predictor(SAM3Predictor):
    """Deterministic mock for tests / demos without GPU.

    Dispatches on the text prompt so exclude-mode tests can simulate SAM 3
    returning different masks for different classes:
      * "sky" / "clouds" / "blue sky"   → bright blue pixels
      * "glacier" / "ice" / default     → bright blue-white pixels (legacy)
      * "distant mountains" / "horizon" → dark top-half pixels
      * "buildings" / "vehicles" ...    → thin high-aspect blobs (none by default)
    """

    def load(self):
        self._model = "mock"
        self._processor = "mock"

    def predict(self, image, text_prompt, exemplar_image=None, score_threshold=0.5):
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H, W = image.shape[:2]
        prompt = (text_prompt or "").lower()

        if "sky" in prompt or "cloud" in prompt:
            # Blue / pale-blue sky
            lower = np.array([95, 20, 140])
            upper = np.array([135, 255, 255])
            mask = cv2.inRange(hsv, lower, upper) > 0
        elif "distant" in prompt or "horizon" in prompt or "mountain" in prompt:
            # Upper-half low-saturation gray/brown
            mask = np.zeros((H, W), dtype=bool)
            upper_half = hsv[: H // 2]
            gray = (upper_half[..., 1] < 40) & (upper_half[..., 2] < 180)
            mask[: H // 2] = gray
        else:
            # Default: bright blue-white "ice"
            lower = np.array([85, 0, 160])
            upper = np.array([130, 80, 255])
            mask = cv2.inRange(hsv, lower, upper) > 0

        if not mask.any():
            return []
        return [Detection(mask=mask, score=0.9, label=text_prompt)]
