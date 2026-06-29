from pathlib import Path

from sam3_agent import AgentConfig, SAM3Agent
from sam3_agent import llm
from sam3_agent.sam3_inference import MockSAM3Predictor


def test_exclude_mode_blacks_out_sky(sky_and_ground_image_path, tmp_path: Path):
    cfg = AgentConfig(
        mode="exclude",
        exclude_prompts=("sky", "clouds"),
        min_keep_coverage=0.1,
        max_keep_coverage=0.95,
        max_retries=1,
        largest_component_only=True,
    )
    agent = SAM3Agent(cfg, predictor=MockSAM3Predictor())
    result = agent.run(sky_and_ground_image_path, tmp_path / "out")

    assert Path(result.output_paths["image"]).exists()
    assert result.mode == "exclude"
    # Sky (top half, 128/256 rows) was excluded; keep coverage must be < 0.6.
    assert result.coverage < 0.6
    # And it should cover most of the bottom half — > 0.3.
    assert result.coverage > 0.3
    assert result.quality.ok, result.quality.reasons
    assert result.attempts == 1


def test_include_mode_preserves_legacy_behavior(ice_image_path, tmp_path: Path):
    cfg = AgentConfig(
        mode="include",
        target="glacier",
        min_keep_coverage=0.001,
        max_keep_coverage=0.95,
        max_retries=2,
        largest_component_only=False,
        morph_close_px=1,
    )
    agent = SAM3Agent(cfg, predictor=MockSAM3Predictor())
    result = agent.run(ice_image_path, tmp_path / "out")

    assert Path(result.output_paths["image"]).exists()
    assert result.mode == "include"
    assert result.attempts >= 1
    assert result.quality.ok, result.quality.reasons


def test_pipeline_gives_up_after_max_retries(sky_and_ground_image_path, tmp_path: Path):
    # Force QA failure by constraining keep coverage to an impossible band.
    cfg = AgentConfig(
        mode="exclude",
        exclude_prompts=("sky",),
        min_keep_coverage=0.99,
        max_keep_coverage=1.0,
        max_retries=2,
    )
    agent = SAM3Agent(cfg, predictor=MockSAM3Predictor())
    result = agent.run(sky_and_ground_image_path, tmp_path / "out")

    assert result.attempts == cfg.max_retries
    assert not result.quality.ok
    assert Path(result.output_paths["image"]).exists()


def test_use_llm_adapt_path_invokes_sdk_decision(
    monkeypatch, sky_and_ground_image_path, tmp_path: Path
):
    """When use_llm is on and the first attempt fails QA, the pipeline asks the
    Claude Agent SDK adapter (here stubbed) for the next prompt set and retries
    with it. No network / subscription call is made — the SDK call is replaced.
    """
    calls = []

    def fake_decide(cfg, attempt, last_prompts, reasons, hints):
        calls.append({"attempt": attempt, "prompts": list(last_prompts), "reasons": list(reasons)})
        return llm.PromptSetDecision(prompts=["sky", "clouds"], reasoning="stub")

    monkeypatch.setattr(llm, "decide_next_prompts", fake_decide)

    cfg = AgentConfig(
        mode="exclude",
        # First attempt keeps almost everything (no real exclusions) → fails the
        # max-coverage gate → triggers the LLM adapt step.
        exclude_prompts=("an object that does not appear in this scene",),
        use_llm=True,
        min_keep_coverage=0.1,
        max_keep_coverage=0.6,
        max_retries=2,
    )
    agent = SAM3Agent(cfg, predictor=MockSAM3Predictor())
    result = agent.run(sky_and_ground_image_path, tmp_path / "out")

    # The SDK adapter was consulted exactly once (after the failed attempt 1)...
    assert len(calls) == 1
    assert calls[0]["attempt"] == 1
    # ...and its proposed prompts drove the successful second attempt.
    assert result.final_prompts == ["sky", "clouds"]
    assert result.attempts == 2
    assert result.quality.ok, result.quality.reasons
