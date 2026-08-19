"""Tests for LangGraph Object Detection Pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from free_detection import ObjectDetectionPipeline, build_detection_graph, DetectionState
from free_detection.detection_pipeline import RoundResult


def test_graph_build():
    """Verify that build_detection_graph produces a valid compilable StateGraph."""
    graph = build_detection_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_json_repair_parsing():
    """Verify that parse_detections uses json_repair to handle broken/malformed JSON."""
    from free_detection.agent.parser import parse_detections

    # 1. Single quotes, trailing commas, unquoted keys
    malformed1 = "{detections: [{'label': 'person', 'bbox_2d': [10, 20, 100, 200],},]}"
    res1 = parse_detections(malformed1)
    assert len(res1) == 1
    assert res1[0]["label"] == "person"
    assert res1[0]["bbox_2d"] == [10, 20, 100, 200]

    # 2. Text with thinking blocks and markdown fences
    malformed2 = "<think>Searching...</think>```json\n[{label: \"car\", bbox_2d: [0, 0, 50, 50]}\n```"
    res2 = parse_detections(malformed2)
    assert len(res2) == 1
    assert res2[0]["label"] == "car"

    # 3. Text wrapped inside <answer> tags with missing end bracket
    malformed3 = "<answer>[{\"label\": \"dog\", \"bbox_2d\": [100, 100, 300, 300]}"
    res3 = parse_detections(malformed3)
    assert len(res3) == 1
    assert res3[0]["label"] == "dog"


def test_agent_package_exports():
    """Verify clean imports from free_detection.agent and nodes."""
    from free_detection.agent import (
        ObjectDetectionPipeline,
        DetectionState,
        RoundResult,
        build_detection_graph,
        parse_detections,
        validate_detections,
        draw_grid,
        render_detections,
        pil_to_data_uri,
        render_detector_prompt,
        render_judge_prompt,
    )
    from free_detection.agent.nodes import (
        node_preprocess,
        node_detector,
        node_crop_verify,
        node_judge,
        node_prepare_next_round,
        route_judge_decision,
        node_finalize,
    )
    assert ObjectDetectionPipeline is not None
    assert DetectionState is not None
    assert RoundResult is not None
    assert callable(build_detection_graph)
    assert callable(node_preprocess)
    assert callable(node_detector)
    assert callable(node_crop_verify)
    assert callable(node_judge)
    assert callable(node_prepare_next_round)
    assert callable(route_judge_decision)
    assert callable(node_finalize)


def test_pipeline_execution_single_round():
    """Test full single-round execution where judge score satisfies the threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "test_sample.jpg"
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        img.save(img_file)

        mock_client = MagicMock()

        detector_resp = MagicMock()
        detector_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='<answer>[{"label": "person", "bbox_2d": [100, 100, 500, 500]}]</answer>'
                )
            )
        ]

        judge_resp = MagicMock()
        judge_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='<score>9</score><feedback>Great bounding box!</feedback><actions>NONE</actions>'
                )
            )
        ]

        mock_client.chat.completions.create.side_effect = [detector_resp, judge_resp]

        pipeline = ObjectDetectionPipeline(
            client=mock_client,
            detector_model="test-detector",
            judge_model="test-judge",
            max_rounds=2,
            score_threshold=8,
        )

        assert hasattr(pipeline, "graph")
        assert pipeline.graph is not None

        progress_calls = []

        def on_progress(round_res, annotated):
            progress_calls.append((round_res.round, round_res.score))

        best, history = pipeline.run(
            image_path=str(img_file),
            categories=["person", "car"],
            category_definitions="- person: human\n- car: vehicle",
            show_plot=False,
            output_dir=str(tmp_path / "output"),
            progress_callback=on_progress,
        )

        assert best["score"] == 9
        assert best["round"] == 1
        assert len(best["detections"]) == 1
        assert best["detections"][0]["label"] == "person"
        assert len(history) == 1
        assert isinstance(history[0], RoundResult)
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 9)

        out_dir = tmp_path / "output"
        assert (out_dir / "best_annotated.jpg").is_file()
        assert (out_dir / "best_detections.json").is_file()
        assert (out_dir / "history.json").is_file()


def test_pipeline_execution_multi_round():
    """Test multi-round loop where round 1 has low score and round 2 reaches threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "test_sample2.jpg"
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        img.save(img_file)

        mock_client = MagicMock()

        # Round 1: detector -> score 5
        d1 = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<answer>[{"label": "car", "bbox_2d": [10, 10, 50, 50]}]</answer>'
                    )
                )
            ]
        )
        j1 = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<score>5</score><feedback>Missed person</feedback><actions>ADD person at [100,100,500,500]</actions>'
                    )
                )
            ]
        )

        # Round 2: detector -> score 9
        d2 = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<answer>[{"label": "car", "bbox_2d": [10, 10, 50, 50]}, {"label": "person", "bbox_2d": [100, 100, 500, 500]}]</answer>'
                    )
                )
            ]
        )
        j2 = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<score>9</score><feedback>All objects found</feedback><actions>NONE</actions>'
                    )
                )
            ]
        )

        mock_client.chat.completions.create.side_effect = [d1, j1, d2, j2]

        pipeline = ObjectDetectionPipeline(
            client=mock_client,
            detector_model="test-detector",
            judge_model="test-judge",
            max_rounds=3,
            score_threshold=8,
        )

        best, history = pipeline.run(
            image_path=str(img_file),
            categories=["person", "car"],
            category_definitions="- person: human\n- car: vehicle",
            show_plot=False,
            output_dir=str(tmp_path / "output2"),
        )

        assert len(history) == 2
        assert history[0].round == 1
        assert history[0].score == 5
        assert history[1].round == 2
        assert history[1].score == 9
        assert best["score"] == 9
        assert best["round"] == 2
        assert len(best["detections"]) == 2


def test_pipeline_execution_tiled():
    """Test detector execution with tiling and NMS enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "test_tiled.jpg"
        img = Image.new("RGB", (1000, 1000), color=(200, 200, 200))
        img.save(img_file)

        mock_client = MagicMock()

        # Mock responses for 4 tiles + 1 judge call
        tile_resp = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<answer>[{"label": "dog", "bbox_2d": [100, 100, 200, 200]}]</answer>'
                    )
                )
            ]
        )
        judge_resp = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='<score>10</score><feedback>Flawless detection</feedback><actions>NONE</actions>'
                    )
                )
            ]
        )

        def mock_create(**kwargs):
            if kwargs.get("model") == "test-judge":
                return judge_resp
            return tile_resp

        mock_client.chat.completions.create.side_effect = mock_create

        pipeline = ObjectDetectionPipeline(
            client=mock_client,
            detector_model="test-detector",
            judge_model="test-judge",
            max_rounds=1,
            score_threshold=8,
            preprocessing_config={
                "tiling_enabled": True,
                "tile_size": 512,
                "tile_overlap": 0.1,
            },
        )

        best, history = pipeline.run(
            image_path=str(img_file),
            categories=["dog"],
            category_definitions="- dog: canine",
            show_plot=False,
            output_dir=str(tmp_path / "output_tiled"),
        )

        assert best["score"] == 10
        assert len(history) == 1
        assert len(best["detections"]) >= 1
