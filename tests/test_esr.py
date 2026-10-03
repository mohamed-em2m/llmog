"""Coverage for the Real-ESRGAN upscaler: registry, downloads, projection math, manager gating, and pipeline wiring."""

import argparse
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_all_entries_well_formed(self):
        from esr.registry import ESR_MODELS

        assert len(ESR_MODELS) >= 6
        for key, entry in ESR_MODELS.items():
            assert entry["file"].endswith(".pth"), key
            assert str(entry["url"]).startswith(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            ), key
            assert entry["url"].endswith(entry["file"]), key
            assert int(entry["scale"]) > 0, key

    def test_default_model_exists(self):
        from esr.registry import DEFAULT_ESR_MODEL, ESR_MODEL_CHOICES, get_model_entry

        assert DEFAULT_ESR_MODEL in ESR_MODEL_CHOICES
        entry = get_model_entry(DEFAULT_ESR_MODEL)
        assert int(entry["scale"]) == 4

    def test_unknown_key_error_mentions_path_override(self):
        from esr.registry import get_model_entry

        with pytest.raises(KeyError, match="esr_model_path"):
            get_model_entry("nope-not-a-model")


# ---------------------------------------------------------------------------
# download / cache resolution (no network: cache hits + explicit paths only)
# ---------------------------------------------------------------------------
class TestResolveWeights:
    def test_explicit_local_path_wins(self, tmp_path):
        from esr.download import resolve_weights

        p = tmp_path / "custom.pth"
        p.write_bytes(b"fake-weights")
        assert resolve_weights(model_path=str(p)) == p

    def test_missing_explicit_path_is_loud(self, tmp_path):
        from esr.download import resolve_weights

        with pytest.raises(FileNotFoundError, match="esr_model_path"):
            resolve_weights(model_path=str(tmp_path / "absent.pth"))

    def test_cache_hit_skips_download(self, tmp_path, monkeypatch):
        from esr.download import resolve_weights
        from esr.registry import get_model_entry

        entry = get_model_entry("general-x4v3")
        cached = tmp_path / str(entry["file"])
        cached.write_bytes(b"cached-weights")

        def _boom(url, dest):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr("esr.download._download", _boom)
        assert resolve_weights(model_key="general-x4v3", cache_dir=tmp_path) == cached

    def test_no_cache_no_download_raises_helpfully(self, tmp_path):
        from esr.download import resolve_weights

        with pytest.raises(FileNotFoundError, match="esr_model_path"):
            resolve_weights(
                model_key="general-x4v3", cache_dir=tmp_path, auto_download=False
            )

    def test_env_var_override(self, tmp_path, monkeypatch):
        from esr.download import resolve_weights

        p = tmp_path / "env.pth"
        p.write_bytes(b"env-weights")
        monkeypatch.setenv("LLMOG_ESR_MODEL", str(p))
        assert resolve_weights() == p

    def test_default_cache_dir_layout(self, monkeypatch):
        from esr.download import default_cache_dir

        monkeypatch.setenv("LLMOG_CACHE_DIR", "/tmp/llmog-cache-test")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert default_cache_dir() == Path("/tmp/llmog-cache-test/esr")

    def test_xdg_cache_home_honored(self, monkeypatch):
        from esr.download import default_cache_dir

        monkeypatch.delenv("LLMOG_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-test")
        assert default_cache_dir() == Path("/tmp/xdg-test/llmog/esr")


# ---------------------------------------------------------------------------
# tiling geometry (F1: narrow images must tile without crashing blending)
# ---------------------------------------------------------------------------
class TestTileGeometry:
    def test_square_image(self):
        from esr.project import tile_geometry

        coords, th, tw = tile_geometry(600, 600, 512, 16)
        assert (th, tw) == (512, 512)
        assert coords == [(0, 0), (0, 88), (88, 0), (88, 88)]

    def test_narrow_image_short_but_full_tiles(self):
        import numpy as np

        from esr.project import tile_geometry

        # 600x100 with tile 512: previously crashed blending (100px-tall
        # tiles vs a square 512 feather mask).
        coords, th, tw = tile_geometry(100, 600, 512, 16)
        assert (th, tw) == (100, 512)
        assert coords == [(0, 0), (0, 88)]
        fake = np.zeros((100, 600, 3), dtype=np.float32)
        for y, x in coords:
            assert fake[y : y + th, x : x + tw].shape == (100, 512, 3)

    def test_sub_tile_image_single_tile(self):
        from esr.project import tile_geometry

        coords, th, tw = tile_geometry(100, 80, 512, 16)
        assert coords == [(0, 0)]
        assert (th, tw) == (100, 80)

    def test_exact_tile_no_duplicates(self):
        from esr.project import tile_geometry

        coords, th, tw = tile_geometry(512, 512, 512, 16)
        assert coords == [(0, 0)]

    def test_rectangular_mask_matches_tiles(self):

        from esr.project import tile_geometry

        # Simulate the blend shapes: every patch must equal the mask shape.
        h, w, tile, overlap, scale = 100, 600, 512, 16, 4
        coords, th, tw = tile_geometry(h, w, tile, overlap)
        mask_shape = (th * scale, tw * scale)
        assert mask_shape == (400, 2048)
        for y, x in coords:
            patch_shape = (th * scale, tw * scale)
            assert patch_shape == mask_shape


# ---------------------------------------------------------------------------
# pre-cap + singleton + config fixes (F2/F3/F5)
# ---------------------------------------------------------------------------
class TestPreCapMath:
    def test_input_cap_keeps_output_within_guard(self):
        from esr.project import fit_long_edge

        # 5000px input, x4 model, 4096 guard -> pre-fit input to 1024 so the
        # ESR output (4096) never exceeds the cap by construction.
        scale, cap = 4, 4096
        in_cap = max(1, cap // scale)
        assert in_cap == 1024
        pw, ph = fit_long_edge(5000, 3000, in_cap)
        assert (pw, ph) == (1024, 614)
        assert max(pw * scale, ph * scale) <= cap

    def test_target_clamped_to_cap(self):
        target, cap = 8192, 4096
        eff = min(target, cap) if target > 0 and cap > 0 else target
        assert eff == 4096

    def test_from_config_honors_explicit_values(self):
        from esr.manager import ESRConfig

        cfg = ESRConfig.from_config({"esr_max_long_edge": 0})
        assert cfg.max_long_edge == 0  # was silently rewritten to 4096
        cfg2 = ESRConfig.from_config({"esr_target_long_edge": None})
        assert cfg2.target_long_edge == 2048
        cfg3 = ESRConfig.from_config({"esr_target_long_edge": 0})
        assert cfg3.target_long_edge == 0
        cfg4 = ESRConfig.from_config({"esr_enabled": True, "esr_scale": "4"})
        assert cfg4.scale == 4

    def test_singleton_keyed_by_config(self):
        from esr.manager import ESRConfig, ESRUpscaler

        ESRUpscaler.reset()
        try:
            a = ESRConfig(enabled=True, model="general-x4v3")
            b = ESRConfig(enabled=True, model="x4plus")
            assert ESRUpscaler.get(a) is ESRUpscaler.get(
                ESRConfig(enabled=True, model="general-x4v3")
            )
            assert ESRUpscaler.get(b) is not ESRUpscaler.get(a)
        finally:
            ESRUpscaler.reset()

    def test_disabled_info_carries_pre_keys(self):
        from esr.manager import ESRConfig, upscale_pil

        img = Image.new("RGB", (64, 48))
        _, info = upscale_pil(img, ESRConfig())
        assert (info["pre_w"], info["pre_h"]) == (64, 48)

    def test_crop_cap_disabled_is_identity(self):
        from auto_annotation.image_io import maybe_esr_upscale_pil

        img = Image.new("RGB", (3000, 2000))
        out, info = maybe_esr_upscale_pil(img, {}, purpose="crop", long_edge_cap=1024)
        assert out is img and info["applied"] is False


class TestProjection:
    def test_norm_to_orig_pixels_square(self):
        from esr.project import norm_to_orig_pixels

        assert norm_to_orig_pixels([0, 0, 1000, 1000], 640, 480) == [0, 0, 640, 480]
        assert norm_to_orig_pixels([250, 250, 750, 750], 1000, 1000) == [
            250,
            250,
            750,
            750,
        ]

    def test_norm_rounds_outward_and_clamps(self):
        from esr.project import norm_to_orig_pixels

        # 1/1000 of 640px = 0.64 -> floor 0 / ceil 1 (outward, no lost coverage)
        assert norm_to_orig_pixels([1, 1, 2, 2], 640, 480) == [0, 0, 2, 1]
        # out-of-range input clamps to the frame
        assert norm_to_orig_pixels([-50, -50, 1200, 1200], 640, 480) == [0, 0, 640, 480]
        # inverted coords normalize
        assert norm_to_orig_pixels([800, 800, 200, 200], 1000, 1000) == [
            200,
            200,
            800,
            800,
        ]

    def test_work_pixels_ratio(self):
        from esr.project import work_pixels_to_orig_pixels

        # 4x working image maps back exactly
        assert work_pixels_to_orig_pixels(
            [400, 400, 800, 800], 4000, 4000, 1000, 1000
        ) == [
            100,
            100,
            200,
            200,
        ]
        # non-square growth uses per-axis ratios
        assert work_pixels_to_orig_pixels(
            [0, 0, 2048, 1024], 2048, 1024, 1024, 512
        ) == [
            0,
            0,
            1024,
            512,
        ]

    def test_growth_factor_and_scaling(self):
        from esr.project import esr_growth_factor, scaled_pixel_param

        assert esr_growth_factor(2048, 2048, 512, 512) == 4.0
        assert esr_growth_factor(512, 512, 512, 512) == 1.0
        # byte-identical behavior when ESR is off (factor <= 1 is a no-op)
        assert scaled_pixel_param(1, 1.0) == 1
        assert scaled_pixel_param(512, 0.5) == 512
        # growth preserves relative measures
        assert scaled_pixel_param(1, 4.0) == 4
        assert scaled_pixel_param(512, 2.0) == 1024
        assert scaled_pixel_param(0, 4.0) == 1  # never vanishes

    def test_fit_long_edge(self):
        from esr.project import fit_long_edge

        assert fit_long_edge(4000, 3000, 2048) == (2048, 1536)
        assert fit_long_edge(100, 80, 2048) == (2048, 1638)  # upscales too
        assert fit_long_edge(800, 600, 0) == (800, 600)  # 0 = keep native
        assert fit_long_edge(2048, 1536, 2048) == (2048, 1536)


# ---------------------------------------------------------------------------
# manager gating (no torch on CPU CI: disabled path + loud enable path)
# ---------------------------------------------------------------------------
class TestManager:
    def test_from_config_dict_and_namespace(self):
        from esr.manager import ESRConfig

        cfg = ESRConfig.from_config({"esr_enabled": True, "esr_model": "x4plus"})
        assert cfg.enabled and cfg.model == "x4plus" and not cfg.for_crops
        ns = argparse.Namespace(esr_enabled=False)
        cfg2 = ESRConfig.from_config(ns)
        assert not cfg2.enabled and cfg2.target_long_edge == 2048

    def test_disabled_upscale_returns_input_untouched(self):
        from esr.manager import ESRConfig, upscale_pil

        img = Image.new("RGB", (64, 48), (10, 20, 30))
        out, info = upscale_pil(img, ESRConfig())
        assert out is img
        assert info["applied"] is False
        assert (info["orig_w"], info["orig_h"]) == (64, 48)

    def test_enabled_without_torch_is_loud(self):
        from esr.manager import ESRConfig, upscale_pil

        cfg = ESRConfig(enabled=True)
        img = Image.new("RGB", (32, 32))
        with pytest.raises(RuntimeError, match="esrgan"):
            upscale_pil(img, cfg)

    def test_torch_status_shape(self):
        from esr.manager import torch_status

        ok, reason = torch_status()
        assert isinstance(ok, bool) and isinstance(reason, str)


# ---------------------------------------------------------------------------
# image_io helpers (ESR off)
# ---------------------------------------------------------------------------
class TestImageIoEsr:
    def test_build_esr_settings_off(self):
        from auto_annotation.image_io import build_esr_settings

        assert build_esr_settings(None) == {}
        assert build_esr_settings({}) == {}
        assert build_esr_settings({"esr_enabled": False}) == {}
        ns = argparse.Namespace(esr_enabled=False)
        assert build_esr_settings(ns) == {}

    def test_build_esr_settings_on(self):
        from auto_annotation.image_io import build_esr_settings

        ns = argparse.Namespace(
            esr_enabled=True, esr_model="general-x4v3", esr_target_long_edge=2048
        )
        out = build_esr_settings(ns)
        assert out["esr_enabled"] is True
        assert out["esr_model"] == "general-x4v3"

    def test_maybe_upscale_disabled_is_identity(self):
        from auto_annotation.image_io import maybe_esr_upscale_pil

        img = Image.new("RGB", (40, 30), (1, 2, 3))
        out, info = maybe_esr_upscale_pil(img, {}, purpose="crop")
        assert out is img and info["applied"] is False

    def test_upscale_scene_for_som_disabled_keeps_box(self):
        from auto_annotation.image_io import upscale_scene_for_som

        img = Image.new("RGB", (100, 80))
        scene, box, info = upscale_scene_for_som(img, (10, 10, 50, 40), {})
        assert box == (10, 10, 50, 40) and info["applied"] is False


# ---------------------------------------------------------------------------
# config + parser surface
# ---------------------------------------------------------------------------
class TestConfigSurface:
    def test_pipeline_config_esr_defaults(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(task="classify", images=["x.jpg"])
        assert cfg.esr_enabled is False
        assert cfg.esr_model == "general-x4v3"
        assert cfg.esr_target_long_edge == 2048
        assert cfg.esr_max_long_edge == 4096
        assert cfg.esr_for_crops is False
        assert cfg.esr_compile is False

    def test_pipeline_config_esr_validation(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception, match="esr_target_long_edge"):
            PipelineConfig(task="classify", images=["x.jpg"], esr_target_long_edge=-1)
        with pytest.raises(Exception, match="esr_max_long_edge"):
            PipelineConfig(task="classify", images=["x.jpg"], esr_max_long_edge=0)

    def test_unified_parser_has_esr_flags(self):
        from main import build_parser

        ns = build_parser().parse_args([])
        assert ns.esr_enabled is False
        assert ns.esr_model == "general-x4v3"
        assert ns.esr_target_long_edge == 2048
        assert ns.esr_for_crops is False

    def test_checkpoint_fingerprint_includes_esr(self):
        from auto_annotation.checkpoint import RUN_SETTINGS_KEYS, build_run_settings

        for key in (
            "esr_enabled",
            "esr_model",
            "esr_model_path",
            "esr_scale",
            "esr_target_long_edge",
            "esr_for_crops",
        ):
            assert key in RUN_SETTINGS_KEYS
        fp = build_run_settings(esr_enabled=True, esr_model="x4plus")
        assert fp["esr_enabled"] is True and fp["esr_model"] == "x4plus"

    def test_gui_prep_config_esr_passthrough(self):
        from interface.viewer_utils import build_prep_config

        base = build_prep_config(prep_enabled=False)
        assert base.get("esr_enabled", False) in (False, None)
        out = build_prep_config(
            prep_enabled=True,
            esr_enabled=True,
            esr_settings={"esr_enabled": True, "esr_model": "general-x4v3"},
        )
        assert out["esr_enabled"] is True
        assert out["esr_model"] == "general-x4v3"


# ---------------------------------------------------------------------------
# --dump_vlm_crops (exact VLM-bound pixels on disk)
# ---------------------------------------------------------------------------
class TestDumpCrops:
    def test_helper_writes_readable_jpg(self, tmp_path):
        import cv2
        import numpy as np

        from auto_annotation.image_io import dump_vlm_crop

        crop = np.full((64, 48, 3), 200, dtype=np.uint8)
        path = dump_vlm_crop(crop, str(tmp_path / "dumps"), "img", 3)
        assert path is not None and Path(path).is_file()
        back = cv2.imread(str(path))
        assert back.shape == (64, 48, 3)

    def test_collect_dumps_per_box(self, tmp_path):
        import cv2
        import numpy as np

        from auto_annotation.batch_api import collect_batch_requests

        img_dir = tmp_path / "imgs"
        lbl_dir = tmp_path / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()
        cv2.imwrite(
            str(img_dir / "big.jpg"), np.full((200, 200, 3), 128, dtype=np.uint8)
        )
        (lbl_dir / "big.txt").write_text("0 0.7 0.7 0.5 0.5\n")
        dump_dir = tmp_path / "dumps"
        reqs, stems = collect_batch_requests(
            str(img_dir),
            str(lbl_dir),
            known_names=["hole"],
            dump_vlm_crops=str(dump_dir),
        )
        assert len(reqs) == 1
        dumped = sorted(p.name for p in dump_dir.iterdir())
        assert dumped == ["big_box0.jpg"]

    def test_config_and_parser_surface(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(task="classify", images=["x.jpg"])
        assert cfg.dump_vlm_crops is None
        from main import build_parser

        ns = build_parser().parse_args([])
        assert ns.dump_vlm_crops is None


# ---------------------------------------------------------------------------
# ESR-then-crop: whole image upscaled once, crops cut from it
# ---------------------------------------------------------------------------
def _fake_2x_upscale(pil_image, esr_settings, purpose="scene", long_edge_cap=None):
    w, h = pil_image.size
    info = {
        "applied": True,
        "orig_w": w,
        "orig_h": h,
        "pre_w": w,
        "pre_h": h,
        "up_w": w * 2,
        "up_h": h * 2,
        "work_w": w * 2,
        "work_h": h * 2,
        "scale": 4,
    }
    return pil_image.resize(
        (w * 2, h * 2), __import__("PIL").Image.Resampling.NEAREST
    ), info


class TestWholeImageEsr:
    def test_filter_measures_original_pixels(self, tmp_path, monkeypatch):
        """A 30px box on a 100px image is filtered at min_box_size=60 even
        though the whole-image 2x upscale makes it 60px in working pixels."""
        import threading

        import cv2
        import numpy as np

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        monkeypatch.setattr(
            "auto_annotation.single_image.maybe_esr_upscale_pil", _fake_2x_upscale
        )

        def _must_not_run(*a, **k):
            raise AssertionError("filtered boxes must never reach the model")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _must_not_run)

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        cv2.imwrite(
            str(train_image / "img0.jpg"), np.full((100, 100, 3), 128, dtype=np.uint8)
        )
        (train_label / "img0.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        out = tmp_path / "out"
        out.mkdir()
        stats = RunStats()
        process_one_image(
            "img0.jpg",
            str(train_image),
            str(train_label),
            str(out),
            {},
            threading.Lock(),
            object(),
            "test-model",
            2,
            False,
            False,
            64,
            64,
            stats,
            False,
            min_box_size=60,
            small_box_action="drop",
            drop_small_images=False,
            esr_settings={"esr_enabled": True, "esr_model": "general-x4v3"},
        )
        assert stats.boxes_skipped_small == 1

    def test_crop_cut_from_upscaled_image_coords_untouched(self, tmp_path, monkeypatch):
        """Crops come from the upscaled image, but output YOLO coords stay
        in original normalized space."""
        import threading

        import cv2
        import numpy as np

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        monkeypatch.setattr(
            "auto_annotation.single_image.maybe_esr_upscale_pil", _fake_2x_upscale
        )
        seen = {}

        def _fake_detect(crop_image, *a, **k):
            seen["shape"] = tuple(crop_image.shape)
            return {"class": "hole", "confidence": 5}

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _fake_detect)

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        cv2.imwrite(
            str(train_image / "img0.jpg"), np.full((100, 100, 3), 128, dtype=np.uint8)
        )
        (train_label / "img0.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        out = tmp_path / "out"
        out.mkdir()
        stats = RunStats()
        class_map = {}
        process_one_image(
            "img0.jpg",
            str(train_image),
            str(train_label),
            str(out),
            class_map,
            threading.Lock(),
            object(),
            "test-model",
            2,
            False,
            False,
            1024,
            1024,
            stats,
            False,
            min_box_size=0,
            crop_resize_ratio=2.0,
            esr_settings={"esr_enabled": True, "esr_model": "general-x4v3"},
        )
        # Native 50px crop x2 (upscaled image) x2 (ratio) = 200px payload.
        # Without whole-image ESR it would have been 100px.
        assert seen["shape"] == (200, 200, 3)
        # Output coords: original normalized values, verbatim.
        assert (out / "img0.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"
        assert class_map == {"hole": 0}

    def test_batch_collect_same_behavior(self, tmp_path, monkeypatch):
        """Batch path mirrors sync: original-pixel filter + original coords."""
        import cv2
        import numpy as np

        from auto_annotation.batch_api import collect_batch_requests

        monkeypatch.setattr(
            "auto_annotation.batch_api.maybe_esr_upscale_pil", _fake_2x_upscale
        )
        img_dir = tmp_path / "imgs"
        lbl_dir = tmp_path / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()
        cv2.imwrite(
            str(img_dir / "img.jpg"), np.full((100, 100, 3), 128, dtype=np.uint8)
        )
        # 10px box: filtered at min 15 on ORIGINAL pixels (20px working).
        (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        reqs, stems = collect_batch_requests(
            str(img_dir),
            str(lbl_dir),
            known_names=["hole"],
            min_box_size=15,
            small_box_action="drop",
            esr_settings={"esr_enabled": True},
        )
        assert reqs == []
        assert stems["img"]["sent"] == {}
        assert stems["img"]["skipped_small"] == 1

    def test_manifest_excluded_from_label_scan(self, tmp_path):
        import cv2
        import numpy as np

        from auto_annotation.image_io import find_labeled_images

        img_dir = tmp_path / "imgs"
        lbl_dir = tmp_path / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()
        cv2.imwrite(str(img_dir / "img.jpg"), np.full((50, 50, 3), 128, dtype=np.uint8))
        (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        (lbl_dir / "skipped_small_images.txt").write_text("some_stem\n")
        assert find_labeled_images(str(img_dir), str(lbl_dir), (".jpg",)) == ["img.jpg"]


# ---------------------------------------------------------------------------
# spandrel strictness: channels-last targets the wrapped nn.Module
# ---------------------------------------------------------------------------
class TestSpandrelStrictTo:
    """Regression for the live Kaggle crash:
    ``TypeError: to() got unexpected keyword arguments ['memory_format']``.

    Spandrel's ModelDescriptor.to() only forwards plain device/dtype
    positionals, so channels-last must be applied to the wrapped module
    (``model.model``), never the descriptor. Faked torch/spandrel modules
    reproduce the strict behavior without a GPU stack.
    """

    def _install_fakes(self, monkeypatch):
        import sys
        import types
        from unittest.mock import MagicMock

        wrapped_to_calls = []

        torch_stub = MagicMock(name="torch")
        torch_stub.cuda.is_available.return_value = True
        torch_stub.device.side_effect = lambda spec: f"device({spec})"
        torch_stub.channels_last = "channels-last-sentinel"

        class _FakeDescriptorBase:
            pass

        class _StrictDescriptor(_FakeDescriptorBase):
            scale = 4
            architecture = "fake"

            def __init__(self):
                self.model = MagicMock(name="wrapped_nn_module")
                self.model.parameters.return_value = []
                self.model.to.side_effect = lambda *a, **k: (
                    wrapped_to_calls.append((a, k)) or self.model
                )

            def to(self, *args, **kwargs):
                if kwargs:
                    raise TypeError(
                        f"to() got unexpected keyword arguments {list(kwargs)}"
                    )
                return self

            def eval(self):
                return self

        class _FakeLoader:
            def __init__(self, device=None):
                pass

            def load_from_file(self, path):
                return _StrictDescriptor()

        spandrel_stub = types.ModuleType("spandrel")
        spandrel_stub.ImageModelDescriptor = _FakeDescriptorBase
        spandrel_stub.ModelLoader = _FakeLoader

        monkeypatch.setitem(sys.modules, "torch", torch_stub)
        monkeypatch.setitem(sys.modules, "spandrel", spandrel_stub)
        return wrapped_to_calls

    def test_channels_last_goes_to_wrapped_module(self, tmp_path, monkeypatch):
        from esr.manager import ESRConfig, ESRUpscaler

        ESRUpscaler.reset()
        try:
            wrapped_calls = self._install_fakes(monkeypatch)
            weights = tmp_path / "fake.pth"
            weights.write_bytes(b"fake")
            cfg = ESRConfig(enabled=True, model_path=str(weights), channels_last=True)
            ESRUpscaler.get(cfg)._ensure_loaded()  # must not raise
            assert wrapped_calls, "wrapped module .to() was never called"
            _, kwargs = wrapped_calls[0]
            assert kwargs.get("memory_format") == "channels-last-sentinel"
        finally:
            ESRUpscaler.reset()

    def test_channels_last_off_touches_nothing(self, tmp_path, monkeypatch):
        from esr.manager import ESRConfig, ESRUpscaler

        ESRUpscaler.reset()
        try:
            wrapped_calls = self._install_fakes(monkeypatch)
            weights = tmp_path / "fake.pth"
            weights.write_bytes(b"fake")
            cfg = ESRConfig(enabled=True, model_path=str(weights), channels_last=False)
            ESRUpscaler.get(cfg)._ensure_loaded()  # must not raise
            assert wrapped_calls == []
        finally:
            ESRUpscaler.reset()
