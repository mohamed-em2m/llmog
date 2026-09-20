"""Unit tests for the whole-image classify task (no server needed)."""

import json

from image_classification.inputs import collect_classify_images
from image_classification.inference import normalize_classify_response
from image_classification.outputs import build_class_index
from image_classification.prompt import parse_categories, render_image_classify_prompt


def test_parse_categories():
    assert parse_categories("a, b ,c") == ["a", "b", "c"]
    assert parse_categories("") == []


def test_normalize_single():
    preds = normalize_classify_response(
        {"class": "Stain", "confidence": 4, "reasoning": "x"}, "single"
    )
    assert len(preds) == 1
    assert preds[0]["class"] == "stain"
    # 1-5 scale normalized to 0-100
    assert preds[0]["confidence"] == 80.0


def test_normalize_top_k_order():
    preds = normalize_classify_response(
        [
            {"class": "b", "confidence": 70},
            {"class": "a", "confidence": 90},
        ],
        "top_k",
        top_k=2,
    )
    assert [p["class"] for p in preds] == ["a", "b"]


def test_normalize_multi_threshold():
    preds = normalize_classify_response(
        {
            "predictions": [
                {"class": "a", "confidence": 90},
                {"class": "b", "confidence": 10},
            ]
        },
        "multi",
        multi_threshold=50.0,
    )
    assert [p["class"] for p in preds] == ["a"]


def test_build_class_index_appends_new():
    idx = build_class_index(
        ["red", "blue"],
        [[{"class": "red", "confidence": 90}, {"class": "green", "confidence": 80}]],
    )
    assert idx == {"red": 0, "blue": 1, "green": 2}


def test_render_prompt_strict_single(tmp_path=None):
    text = render_image_classify_prompt(
        ["hole", "stain"], "strict", "", "single", 3, 50.0
    )
    assert "hole, stain" in text
    assert "strict" in text.lower()


def test_collect_images_dedupes(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = collect_classify_images([str(a), str(a)], str(tmp_path), ".jpg,.png")
    names = sorted(p.name for p in out)
    assert names == ["a.jpg", "b.png"]


def test_classify_config_validation():
    from schemes import PipelineConfig

    cfg = PipelineConfig(task="classify", images=["a.jpg"])
    assert cfg.task == "classify"
    assert cfg.output_format == "csv"
    try:
        PipelineConfig(task="classify")
    except Exception as exc:
        assert "--image" in str(exc) or "--input_folder" in str(exc)
    else:
        raise AssertionError("classify without inputs should fail")


def test_outputs_csv_and_yolo(tmp_path):
    from image_classification.outputs import (
        write_predictions_csv,
        write_yolo_cls,
    )

    rows = [
        {
            "filename": "a.jpg",
            "path": "x/a.jpg",
            "predicted_class": "red",
            "confidence": "90.0",
            "reasoning": "r",
            "top_k_json": "[]",
            "all_scores_json": json.dumps([{"class": "red", "confidence": 90}]),
            "status": "ok",
            "error": "",
        }
    ]
    idx = {"red": 0, "blue": 1}
    csv_path = write_predictions_csv(rows, tmp_path / "predictions.csv")
    assert csv_path.is_file()
    written = write_yolo_cls(rows, idx, tmp_path / "labels")
    assert written and (tmp_path / "labels" / "a.txt").read_text().startswith("0 ")
