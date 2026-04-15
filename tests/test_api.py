from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from sam3_agent import api as api_module
from sam3_agent.sam3_inference import MockSAM3Predictor


def _png_bytes(arr) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_segment_endpoint_exclude_mode(monkeypatch, sky_and_ground_image):
    monkeypatch.setattr(api_module, "SAM3Agent", _agent_factory_with_mock())

    client = TestClient(api_module.app)
    resp = client.post(
        "/segment",
        files={"file": ("sg.png", _png_bytes(sky_and_ground_image), "image/png")},
        data={
            "mode": "exclude",
            "exclude_prompts": "sky,clouds",
            "output_format": "png",
            "use_llm": "false",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "exclude"
    assert body["attempts"] >= 1
    assert body["final_prompts"]
    assert Path(body["output_paths"]["image"]).exists()
    assert isinstance(body["history"], list) and body["history"]


def test_segment_endpoint_include_mode(monkeypatch, ice_image):
    monkeypatch.setattr(api_module, "SAM3Agent", _agent_factory_with_mock())

    client = TestClient(api_module.app)
    resp = client.post(
        "/segment",
        files={"file": ("ice.png", _png_bytes(ice_image), "image/png")},
        data={
            "mode": "include",
            "preserve": "glacier",
            "output_format": "png",
            "use_llm": "false",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "include"
    assert body["final_prompts"] == ["glacier"]


def test_healthz():
    client = TestClient(api_module.app)
    assert client.get("/healthz").json() == {"ok": True}


def _agent_factory_with_mock():
    from sam3_agent.agent import SAM3Agent as _RealAgent

    def factory(cfg):
        return _RealAgent(cfg, predictor=MockSAM3Predictor())

    return factory
