"""Coverage for nested (grouped) YAML configs (rule-based sections).

Sections are pure organization: uniform families strip their prefix
(batch_/esr_/prep_), everything else resolves verbatim or via the
micro-alias table (see llmog/main.py). Flat keys keep working; a flat
top-level twin wins over nested; explicit CLI flags win over everything.
"""

import pytest


def _norm(raw):
    import sys

    sys.path.insert(0, "llmog")
    from main import _normalize_nested_config

    return _normalize_nested_config(raw)


class TestResolutionRules:
    def test_prefix_strip(self):
        n = _norm(
            {
                "esr": {"enabled": True, "target_long_edge": 1024},
                "prep": {"short_edge": 512},
                "batch": {"poll_interval": 30},
            }
        )
        assert n["esr_enabled"] is True
        assert n["esr_target_long_edge"] == 1024
        assert n["prep_short_edge"] == 512
        assert n["batch_poll_interval"] == 30

    def test_verbatim(self):
        n = _norm(
            {
                "server": {"base_url": "https://x/v1", "port": 1234},
                "llm": {"detector_model": "m", "max_rounds": 3},
                "sampling": {"shuffle": True, "batch_size": 7},
            }
        )
        assert n["base_url"] == "https://x/v1"
        assert n["port"] == 1234
        assert n["detector_model"] == "m"
        assert n["batch_size"] == 7

    def test_aliases(self):
        n = _norm(
            {
                "logging": {"level": "DEBUG", "file": "x.log"},
                "output": {"folder": "./o", "inplace": True},
                "sampling": {"start": 5, "end": 9},
                "sizing": {"size": 640},
                "reclass": {
                    "context": "full_som",
                    "resize_ratio": 1.5,
                    "dump_crops": "./d",
                },
                "batch": {"enabled": True, "window": "24h"},
                "server": {
                    "type": "external",
                    "url": "https://x",
                    "key": "k",
                    "workers": 3,
                },
                "classes": {"mode": "free", "defs": "d"},
                "classification": {
                    "mode": "top_k",
                    "format": "yolo",
                    "temperature": 0.5,
                    "max_tokens": 7,
                },
                "llm": {"retries": 5},
            }
        )
        assert (n["log_level"], n["log_file"]) == ("DEBUG", "x.log")
        assert (n["output_folder"], n["inplace_saving"]) == ("./o", True)
        assert (n["start_index"], n["end_index"]) == (5, 9)
        assert n["image_size"] == 640
        assert (n["recls_context"], n["crop_resize_ratio"]) == ("full_som", 1.5)
        assert n["dump_vlm_crops"] == "./d"
        assert (n["use_batch_api"], n["batch_completion_window"]) == (True, "24h")
        assert (n["server_type"], n["base_url"], n["api_key"]) == (
            "external",
            "https://x",
            "k",
        )
        assert n["max_workers"] == 3
        assert (n["class_mode"], n["class_definitions"]) == ("free", "d")
        assert (n["classification_mode"], n["output_format"]) == ("top_k", "yolo")
        assert (n["classification_temperature"], n["classification_max_tokens"]) == (
            0.5,
            7,
        )
        assert n["api_retries"] == 5

    def test_prefix_wins_over_verbatim_on_ties(self):
        # esr.batch_size must mean esr_batch_size, never staging batch_size.
        n = _norm({"esr": {"batch_size": 8}})
        assert n["esr_batch_size"] == 8
        assert "batch_size" not in n

    def test_non_section_mappings_pass_through(self):
        n = _norm({"serving_extra": {"ctx_size": 1}, "extra_body": {"a": 1}})
        assert n["serving_extra"] == {"ctx_size": 1}
        assert n["extra_body"] == {"a": 1}

    def test_none_section_skipped(self):
        assert _norm({"provider": None, "task": "classify"}) == {"task": "classify"}

    def test_flat_twin_wins_over_nested(self):
        n = _norm({"esr": {"enabled": True}, "esr_enabled": False})
        assert n["esr_enabled"] is False


class TestGuards:
    def test_batch_size_guard(self):
        with pytest.raises(ValueError, match="staging"):
            _norm({"batch": {"size": 10}})

    def test_unknown_section_key(self):
        with pytest.raises(ValueError, match="Unknown YAML key"):
            _norm({"nope": {"a": 1}})

    def test_flat_typo_suggests(self):
        with pytest.raises(ValueError, match="esr_enabled"):
            _norm({"esr_enabeld": True})

    def test_nested_typo_suggests(self):
        with pytest.raises(ValueError, match="Did you mean"):
            _norm({"esr": {"targt_long_edge": 1}})

    def test_unknown_nested_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            _norm({"esr": {"nope": 1}})

    def test_non_mapping_section(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _norm({"esr": "yes"})

    def test_section_collision(self):
        with pytest.raises(ValueError, match="collides"):
            _norm({"llm": {"model": "a"}, "server": {"model": "b"}})


class TestPrecedence:
    def test_nested_below_flat_below_cli(self, tmp_path):
        import sys

        sys.path.insert(0, "llmog")
        from main import parse_args

        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "task: classify\n"
            "images: [x.jpg]\n"
            "sampling:\n"
            "  seed: 7\n"
            "seed: 9\n"
            "logging:\n"
            "  level: DEBUG\n",
            encoding="utf-8",
        )
        cfg = parse_args(["--config", str(cfg_file)])
        assert cfg.seed == 9  # flat YAML twin beats nested
        assert cfg.log_level == "DEBUG"
        cfg2 = parse_args(["--config", str(cfg_file), "--seed", "11"])
        assert cfg2.seed == 11  # CLI beats YAML


class TestExamples:
    FILES = [
        "examples/config.example.yaml",
        "examples/esr_detection.example.yaml",
        "examples/auto_label_vllm.example.yaml",
        "examples/auto_label_external.example.yaml",
    ]

    def test_all_examples_load(self):
        import sys

        sys.path.insert(0, "llmog")
        from main import parse_args

        tasks = {
            "examples/config.example.yaml": "free_detection",
            "examples/esr_detection.example.yaml": "free_detection",
            "examples/auto_label_vllm.example.yaml": "auto_label",
            "examples/auto_label_external.example.yaml": "auto_label",
        }
        for path, task in tasks.items():
            cfg = parse_args(["--task", task, "--config", path])
            assert cfg.task == task, path

    def test_detection_values(self):
        import sys

        sys.path.insert(0, "llmog")
        from main import parse_args

        cfg = parse_args(
            ["--task", "free_detection", "--config", "examples/config.example.yaml"]
        )
        assert cfg.images == ["./assets/image.png"]
        assert cfg.max_rounds == 5 and cfg.score_threshold == 9
        assert cfg.prep_short_edge == 1024 and cfg.prep_grid_step == 50
        assert cfg.prep_grid_line_color == "blue"
        assert cfg.serving_extra == {"ctx_size": 20000, "parallel_slots": 1}
        assert cfg.esr_enabled is False

        esr = parse_args(
            [
                "--task",
                "free_detection",
                "--config",
                "examples/esr_detection.example.yaml",
            ]
        )
        assert esr.esr_enabled is True and esr.esr_target_long_edge == 2048
        assert esr.esr_overlap == 16 and esr.esr_channels_last is True
        assert esr.prep_tiling_enabled is True and esr.prep_tile_overlap == 0.2

    def test_autolabel_values(self):
        import sys

        sys.path.insert(0, "llmog")
        from main import parse_args

        vllm = parse_args(
            [
                "--task",
                "auto_label",
                "--config",
                "examples/auto_label_vllm.example.yaml",
            ]
        )
        assert vllm.server_type == "vllm" and vllm.max_workers == 10
        assert vllm.max_model_len == 20000 and vllm.trust_remote_code is True
        assert vllm.class_mode == "hybrid" and vllm.batch_size == 40

        ext = parse_args(
            [
                "--task",
                "auto_label",
                "--config",
                "examples/auto_label_external.example.yaml",
            ]
        )
        assert ext.server_type == "external" and ext.max_workers == 4
        assert ext.use_batch_api is True and ext.batch_mode == "auto"
        assert ext.batch_completion_window == "24h"
        assert ext.class_mode == "strict" and ext.model == "your-model-id"
