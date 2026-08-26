"""Extended coverage for detection_viewer, viewer_utils, schemes, servers, tab_server, image_preprocessing."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# detection_viewer
# ---------------------------------------------------------------------------
class TestDetectionViewer:
    def test_load_image_variants(self):
        from detection_viewer import _load_image

        # PIL
        im = Image.new("RGB", (10, 10), (255, 0, 0))
        assert _load_image(im) is im

        # numpy
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        out = _load_image(arr)
        assert isinstance(out, Image.Image)
        assert out.size == (5, 5)

        # Path
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            im.save(p)
            out2 = _load_image(p)
            assert isinstance(out2, Image.Image)

        # str path
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "y.jpg"
            im.save(p)
            out3 = _load_image(str(p))
            assert isinstance(out3, Image.Image)

    def test_save_image_to_cache_dedup(self, tmp_path, monkeypatch):
        from detection_viewer import _save_image_to_cache, _WEBP_URL_CACHE

        _WEBP_URL_CACHE.clear()
        im = Image.new("RGB", (32, 32), (0, 255, 0))

        # mock save_pil_to_cache to avoid gradio file system
        monkeypatch.setattr(
            "gradio.processing_utils.save_pil_to_cache",
            lambda img, cache_dir, format="webp": str(Path(cache_dir) / "fake.webp"),
        )
        url1 = _save_image_to_cache(im, str(tmp_path))
        assert url1.endswith("fake.webp")
        # second call with same object id should hit identity cache
        url2 = _save_image_to_cache(im, str(tmp_path))
        assert url1 == url2

        # different object same content should hit content hash cache
        im2 = Image.new("RGB", (32, 32), (0, 255, 0))
        url3 = _save_image_to_cache(im2, str(tmp_path))
        assert url3 == url1

        # cache size bounded
        assert len(_WEBP_URL_CACHE) <= 256

    def test_detection_viewer_postprocess(self, tmp_path, monkeypatch):
        from detection_viewer import DetectionViewer

        monkeypatch.setattr(
            "gradio.processing_utils.save_pil_to_cache",
            lambda img, cache_dir, format="webp": str(Path(cache_dir) / "a.webp"),
        )
        # Need to set GRADIO_CACHE attribute after creation
        viewer = DetectionViewer()
        viewer.GRADIO_CACHE = str(tmp_path)

        # None -> None
        assert viewer._process(None) is None
        assert viewer.postprocess(None) is None

        # string passthrough
        assert viewer.postprocess("already") == "already"

        im = Image.new("RGB", (20, 20), (100, 100, 100))
        # 2-tuple
        payload = viewer._process(
            (
                im,
                [
                    {
                        "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
                        "label": "cat",
                        "score": 0.9,
                    }
                ],
            )
        )
        data = json.loads(payload)
        assert "image" in data
        assert data["annotations"][0]["label"] == "cat"
        assert data["annotations"][0]["score"] == 0.9
        assert data["annotations"][0]["bbox"]["x"] == 1

        # default label when missing
        payload2 = viewer._process(
            (im, [{"bbox": {"x": 0, "y": 0, "width": 1, "height": 1}}])
        )
        data2 = json.loads(payload2)
        assert data2["annotations"][0]["label"].startswith("Detection")

        # with keypoints -> default Person
        payload3 = viewer._process((im, [{"keypoints": [{"x": 5, "y": 5}]}]))
        data3 = json.loads(payload3)
        assert data3["annotations"][0]["label"].startswith("Person")

        # 3-tuple with score_threshold config
        payload4 = viewer._process((im, [], {"score_threshold": (0.2, 0.8)}))
        data4 = json.loads(payload4)
        assert data4["scoreThresholdMin"] == 0.2
        assert data4["scoreThresholdMax"] == 0.8

        # color palette cycles
        many = [{"label": f"a{i}"} for i in range(20)]
        payload5 = viewer._process((im, many))
        data5 = json.loads(payload5)
        assert len(data5["annotations"]) == 20

    def test_process_example(self, tmp_path, monkeypatch):
        from detection_viewer import DetectionViewer

        monkeypatch.setattr(
            "gradio.processing_utils.save_pil_to_cache",
            lambda img, cache_dir, format="webp": str(Path(cache_dir) / "ex.webp"),
        )
        viewer = DetectionViewer()
        viewer.GRADIO_CACHE = str(tmp_path)
        im = Image.new("RGB", (10, 10))
        html = viewer.process_example((im, []))
        assert "img" in html and "ex.webp" in html
        assert viewer.process_example(None) is None

    def test_api_info(self):
        from detection_viewer import DetectionViewer

        viewer = DetectionViewer()
        info = viewer.api_info()
        assert info["type"] == "string"
        assert "bbox" in info["description"]


# ---------------------------------------------------------------------------
# viewer_utils
# ---------------------------------------------------------------------------
class TestViewerUtils:
    def test_pipeline_detections_to_annotations(self):
        from interface.viewer_utils import pipeline_detections_to_annotations

        assert pipeline_detections_to_annotations([], (100, 100)) == []
        assert (
            pipeline_detections_to_annotations(
                [{"label": "x", "bbox_2d": [0, 0, 10, 10]}], (0, 0)
            )
            == []
        )

        dets = [
            {"label": "person", "bbox_2d": [0, 0, 100, 100], "score": 9},
            {"label": "car", "bbox_2d": [100, 100, 200, 200], "score": 0.5},
            {"label": "bad", "bbox_2d": [0, 0]},  # invalid
            {
                "label": "inv",
                "bbox_2d": [200, 200, 100, 100],
                "score": 50,
            },  # inverted + score 50 -> 0.5
        ]
        out = pipeline_detections_to_annotations(dets, (1000, 1000))
        assert len(out) == 3  # bad skipped
        assert out[0]["bbox"]["x"] == pytest.approx(0)
        assert out[0]["score"] == pytest.approx(0.9)  # 9 -> 0.9
        assert out[1]["score"] == pytest.approx(0.5)
        # inverted should be normalized
        assert out[2]["bbox"]["x"] == pytest.approx(100)
        assert out[2]["score"] == pytest.approx(0.5)

        # clamp test
        out2 = pipeline_detections_to_annotations(
            [{"bbox_2d": [-100, -100, 2000, 2000]}], (100, 100)
        )
        assert out2[0]["bbox"]["width"] == pytest.approx(100)

        # zero area skipped
        out3 = pipeline_detections_to_annotations(
            [{"bbox_2d": [10, 10, 10, 10]}], (100, 100)
        )
        assert out3 == []

    def test_region_results_to_annotations(self):
        from interface.viewer_utils import region_results_to_annotations

        assert region_results_to_annotations([], []) == []
        regions = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            {"x1": 60, "y1": 60, "x2": 30, "y2": 30},
            {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        ]
        labels = ["cat", "dog"]
        out = region_results_to_annotations(regions, labels, confidences=[90, 0.8])
        assert len(out) == 2
        assert out[0]["label"] == "cat"
        assert out[0]["score"] == pytest.approx(0.9)
        assert out[1]["score"] == pytest.approx(0.8)
        # swapped coords normalized
        assert out[1]["bbox"]["x"] == 30

    def test_realtime_boxes_to_annotations(self):
        from interface.viewer_utils import realtime_boxes_to_annotations

        assert realtime_boxes_to_annotations([]) == []
        boxes = [
            [10, 20, 50, 80, "person", 1],
            [0, 0, 10, 10],  # no label
            None,
            [0, 0, 0, 0, "x"],  # zero area
        ]
        out = realtime_boxes_to_annotations(boxes)
        assert len(out) == 2
        assert out[0]["label"] == "person #1"
        assert out[1]["label"].startswith("obj")

    def test_build_viewer_payload(self):
        from interface.viewer_utils import (
            build_viewer_payload,
            detections_to_viewer_payload,
        )

        assert build_viewer_payload(None, []) is None
        im = Image.new("RGB", (10, 10))
        assert build_viewer_payload(im, None) == (im, [])
        assert detections_to_viewer_payload(None, []) is None
        payload = detections_to_viewer_payload(
            im, [{"label": "x", "bbox_2d": [0, 0, 100, 100]}]
        )
        assert payload[0] is im
        assert len(payload[1]) == 1

    def test_build_prep_config(self):
        from interface.viewer_utils import build_prep_config

        cfg = build_prep_config(False)
        assert cfg["resolution_enabled"] is False
        assert cfg["som_enabled"] is False

        cfg2 = build_prep_config(
            True,
            prep_short_edge=800,
            prep_contrast_method="clahe",
            prep_grid_line_color="custom",
            prep_grid_line_color_custom="#123456",
            prep_tile_overlap=20,
            prep_crop_padding=15,
            prep_custom_resize_enabled=True,
            prep_custom_resize_width=640,
            prep_custom_resize_height=480,
        )
        assert cfg2["target_short_edge"] == 800
        assert cfg2["contrast_method"] == "clahe"
        assert cfg2["grid_line_color"] == "#123456"
        assert cfg2["tile_overlap"] == pytest.approx(0.2)
        assert cfg2["crop_padding"] == pytest.approx(0.15)
        assert cfg2["custom_resize"] is True
        assert cfg2["custom_resize_width"] == 640


# ---------------------------------------------------------------------------
# schemes PipelineConfig
# ---------------------------------------------------------------------------
class TestPipelineConfig:
    def test_valid_free_detection(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(images=["a.jpg"], task="free_detection")
        assert cfg.images == ["a.jpg"]
        cfg2 = PipelineConfig(images="a.jpg", task="free_detection")
        assert cfg2.images == ["a.jpg"]

    def test_valid_auto_label(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(task="auto_label", train_image="imgs/")
        assert cfg.train_image == "imgs/"

    def test_missing_images_fails(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="free_detection.*--image"):
            PipelineConfig(task="free_detection", images=[])

    def test_missing_train_image_fails(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="train_image"):
            PipelineConfig(task="auto_label", train_image=None)

    def test_overlap_validator(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="prep_tile_overlap"):
            PipelineConfig(images=["a.jpg"], prep_tile_overlap=0.9)
        with pytest.raises(Exception, match="prep_tile_overlap"):
            PipelineConfig(images=["a.jpg"], prep_tile_overlap=-0.1)

    def test_gpu_mem_validator(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="gpu_memory_utilization"):
            PipelineConfig(images=["a.jpg"], gpu_memory_utilization=1.5)
        with pytest.raises(Exception, match="gpu_memory_utilization"):
            PipelineConfig(images=["a.jpg"], gpu_memory_utilization=0)

    def test_indices_validator(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="start_index"):
            PipelineConfig(images=["a.jpg"], start_index=-1)
        with pytest.raises(Exception, match="end_index"):
            PipelineConfig(images=["a.jpg"], start_index=5, end_index=3)

    def test_vllm_and_llama_props(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(images=["a.jpg"], model="my/model", tensor_parallel_size=2)
        v = cfg.vllm
        assert v["--tensor-parallel-size"] == 2
        assert v["--model"] == "my/model"
        llama_cpp = cfg.llama_cpp
        assert "-m" in llama_cpp
        assert llama_cpp["-m"] == "my/model"


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------
class TestServers:
    def test_llama_detailed_status_and_health(self):
        from servers.llama_server_manager import LlamaServerManager

        mgr = LlamaServerManager(model="test", port=8123)
        mgr.process = MagicMock()
        mgr.process.pid = 1234
        mgr.process.poll.return_value = None
        mgr._started_at = time.time() - 65
        mgr._last_health_latency_ms = 12.3
        mgr.logs = ["a\n", "b\n"]

        with patch("servers.llama_server_manager.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "ok"}
            assert mgr.is_healthy() is True
            assert mgr._last_health_latency_ms is not None

        details = mgr.get_detailed_status()
        assert details["pid"] == 1234
        assert details["port"] == 8123
        assert details["uptime_s"] >= 60
        assert details["log_lines"] == 2

        # dead process
        mgr.process.poll.return_value = 1
        assert mgr.is_healthy() is False

    def test_vllm_detailed_status(self):
        from servers.vllm_server_manager import VllmServerManager

        mgr = VllmServerManager(model="test", port=8124)
        mgr.process = MagicMock()
        mgr.process.pid = 999
        mgr.process.poll.return_value = None
        mgr._started_at = time.time() - 3700
        mgr.logs = []
        details = mgr.get_detailed_status()
        assert details["pid"] == 999
        assert details["uptime_s"] >= 3600

    def test_logs_timestamped(self):
        from servers.llama_server_manager import LlamaServerManager

        mgr = LlamaServerManager(model="x", port=8125)
        # Simulate monitor adding timestamped line
        mgr.logs = ["[12:00:01] hello\n", "[12:00:02] world\n"]
        assert "[12:00:01]" in mgr.get_logs()


# ---------------------------------------------------------------------------
# tab_server helpers
# ---------------------------------------------------------------------------
class TestTabServer:
    def test_fmt_uptime(self):
        from interface.tab_server import _fmt_uptime

        assert _fmt_uptime(5) == "5s"
        assert _fmt_uptime(65) == "1m 5s"
        assert _fmt_uptime(3665) == "1h 1m"

    def test_render_status_html(self):
        from interface.tab_server import _render_status_html

        html = _render_status_html(None, None)
        assert "No local server" in html

        details = {
            "pid": 123,
            "port": 8080,
            "model": "a/b/c/model.gguf",
            "url": "http://localhost:8080",
            "uptime_s": 90,
            "health_latency_ms": 15.2,
            "log_lines": 10,
        }
        html2 = _render_status_html(details, True)
        assert "123" in html2
        assert "8080" in html2
        assert "Healthy" in html2

        html3 = _render_status_html(details, False)
        assert "Starting" in html3

    def test_get_server_status_and_logs_no_server(self):
        from interface import tab_server
        from interface.state import state

        with state.server_lock:
            old = state.server_manager
            state.server_manager = None
            logs, badge, html = tab_server.get_server_status_and_logs()
            assert "No server" in logs
            assert "STOPPED" in badge
            assert "No local server" in html
            state.server_manager = old

    def test_get_server_status_and_logs_healthy(self):
        from interface import tab_server
        from interface.state import state

        mgr = MagicMock()
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None
        mgr.process.pid = 111
        mgr.port = 8080
        mgr.model = "test/model"
        mgr.server_url = "http://localhost:8080"
        mgr.get_logs.return_value = "\n".join([f"line {i}" for i in range(200)])
        mgr.is_healthy.return_value = True
        mgr.get_detailed_status.return_value = {
            "pid": 111,
            "port": 8080,
            "host": "0.0.0.0",
            "url": "http://localhost:8080",
            "model": "test/model",
            "uptime_s": 10,
            "health_latency_ms": 5.0,
            "log_lines": 200,
        }
        with state.server_lock:
            old = state.server_manager
            state.server_manager = mgr
            logs, badge, html = tab_server.get_server_status_and_logs()
            assert "healthy" in logs.lower()
            assert "RUNNING" in badge
            assert "111" in html
            state.server_manager = old

    def test_clear_and_download_logs(self, tmp_path):
        from interface import tab_server
        from interface.state import state
        from servers.llama_server_manager import LlamaServerManager

        mgr = LlamaServerManager(model="x", port=8126)
        mgr.logs = ["a\n", "b\n"]
        mgr._started_at = time.time()
        with state.server_lock:
            old = state.server_manager
            state.server_manager = mgr
            # clear
            logs, html = tab_server.clear_server_logs()
            assert "[UI] Logs cleared" in logs
            # download
            path = tab_server.download_server_logs()
            assert Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            assert "Logs cleared" in content or "LLM Server Logs" in content
            Path(path).unlink(missing_ok=True)
            state.server_manager = old

        # no server
        with state.server_lock:
            old = state.server_manager
            state.server_manager = None
            path2 = tab_server.download_server_logs()
            assert Path(path2).exists()
            Path(path2).unlink(missing_ok=True)
            state.server_manager = old


# ---------------------------------------------------------------------------
# image_preprocessing
# ---------------------------------------------------------------------------
class TestImagePreprocessing:
    def test_preprocess_vlm_conditioning_smoke(self):
        from free_detection.image_preprocessing import preprocess_vlm_conditioning

        im = Image.new("RGB", (64, 64), (128, 128, 128))
        out = preprocess_vlm_conditioning(
            im, clahe_enabled=False, white_balance_enabled=False
        )
        assert isinstance(out, Image.Image)
        assert out.size == (64, 64)

        out2 = preprocess_vlm_conditioning(
            im, clahe_enabled=True, white_balance_enabled=True
        )
        assert isinstance(out2, Image.Image)

    def test_tiling_helpers(self):
        from free_detection.image_preprocessing import (
            get_image_tiles,
            apply_nms,
            calculate_iou,
        )
        from PIL import Image

        # get_image_tiles returns list[dict] for 1000x1000 with 512/0.1
        im = Image.new("RGB", (1000, 1000))
        tiles = get_image_tiles(im, tile_size=512, overlap_pct=0.1)
        assert len(tiles) >= 4
        for t in tiles:
            assert isinstance(t, dict)
            assert "tile_image" in t
            assert isinstance(t["tile_image"], Image.Image)
            assert "tile_x" in t and "tile_y" in t

        # calculate_iou smoke
        assert calculate_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)
        assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

        # NMS smoke: overlapping detections
        dets = [
            {"label": "a", "bbox_2d": [0, 0, 100, 100], "score": 0.9},
            {"label": "a", "bbox_2d": [10, 10, 110, 110], "score": 0.8},
            {"label": "a", "bbox_2d": [200, 200, 300, 300], "score": 0.95},
        ]
        kept = apply_nms(dets, iou_threshold=0.5)
        assert isinstance(kept, list)
        assert len(kept) <= 3

    def test_gray_world_wb(self):
        from free_detection.image_preprocessing import _apply_gray_world_wb

        arr = np.full((10, 10, 3), 100, dtype=np.uint8)
        arr[:, :, 0] = 150  # more red
        out = _apply_gray_world_wb(arr)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# free_detection agent parser edge
# ---------------------------------------------------------------------------
class TestParserEdge:
    def test_empty_and_invalid(self):
        from free_detection.agent.parser import parse_detections

        # empty string may raise ValueError or return [] depending on implementation;
        # ensure it doesn't crash with valid return or raises correctly
        try:
            res = parse_detections("")
            assert isinstance(res, list)
        except ValueError:
            pass
        try:
            res2 = parse_detections("no json here")
            assert isinstance(res2, list)
        except ValueError:
            pass
        with pytest.raises((TypeError, AttributeError, ValueError)):
            parse_detections(None)  # type: ignore

    def test_score_normalization(self):
        from interface.viewer_utils import pipeline_detections_to_annotations

        # score 10 -> 1.0, 100 -> 1.0
        out = pipeline_detections_to_annotations(
            [{"bbox_2d": [0, 0, 10, 10], "score": 10}], (100, 100)
        )
        assert out[0]["score"] == 1.0
        out2 = pipeline_detections_to_annotations(
            [{"bbox_2d": [0, 0, 10, 10], "score": 100}], (100, 100)
        )
        assert out2[0]["score"] == 1.0
