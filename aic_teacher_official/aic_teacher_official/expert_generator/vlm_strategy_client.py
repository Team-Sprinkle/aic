"""OpenAI-backed VLM strategy provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aic_teacher_official.vlm_planner import _image_items, load_openai_api_key
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot
from aic_teacher_official.expert_generator.vlm_strategy import (
    ExpertMode,
    VLMStrategy,
    build_strategy_prompt,
    parse_vlm_strategy,
    save_strategy_debug,
)


class OpenAIVLMStrategyProvider:
    def __init__(self, *, model: str = "gpt-5-mini"):
        self.model = model

    def strategy_for_scene(
        self,
        snapshot: SceneSnapshot,
        *,
        mode: ExpertMode,
        output_dir: str | Path | None = None,
    ) -> VLMStrategy:
        scene_summary = snapshot.to_dict()
        prompt = build_strategy_prompt(scene_summary, mode=mode)
        if not snapshot.camera_images:
            raise RuntimeError("GPT-5-mini strategy requires live validated camera images; snapshot.camera_images is empty.")
        api_key = load_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for GPT-5-mini strategy generation.")
        raw_response = ""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            content: list[dict[str, Any]] = [
                {"type": "input_text", "text": prompt},
                *_image_items([Path(path) for path in snapshot.camera_images[:8]]),
            ]
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
            )
            raw_response = response.output_text
            strategy = parse_vlm_strategy(raw_response, expected_mode=mode.value)
            if output_dir is not None:
                save_strategy_debug(output_dir, prompt=prompt, raw_response=raw_response, strategy=strategy)
            return strategy
        except Exception as ex:
            if output_dir is not None:
                save_strategy_debug(output_dir, prompt=prompt, raw_response=raw_response, error=f"{type(ex).__name__}: {ex}")
            raise
