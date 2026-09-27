"""extra_body: YAML/CLI parsing and forwarding to the classify call."""

import pytest


class TestConfigParsing:
    def test_yaml_mapping_accepted(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(
            task="auto_label",
            train_image="imgs/",
            extra_body={"provider": {"order": ["x"]}},
        )
        assert cfg.extra_body == {"provider": {"order": ["x"]}}

    def test_defaults_to_none(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(task="auto_label", train_image="imgs/")
        assert cfg.extra_body is None

    def test_cli_json_string_accepted(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(
            task="auto_label",
            train_image="imgs/",
            extra_body='{"provider": {"order": ["x"]}}',
        )
        assert cfg.extra_body == {"provider": {"order": ["x"]}}

    def test_invalid_rejected(self):
        from schemes import PipelineConfig

        with pytest.raises(Exception):
            PipelineConfig(
                task="auto_label",
                train_image="imgs/",
                extra_body='["not", "a", "dict"]',
            )
        with pytest.raises(Exception):
            PipelineConfig(task="auto_label", train_image="imgs/", extra_body="{oops")
        with pytest.raises(Exception):
            PipelineConfig(task="auto_label", train_image="imgs/", extra_body=42)

    def test_standalone_cli_json_flag(self):
        from auto_annotation.cli import parse_args

        args = parse_args(
            [
                "--output_folder",
                "./out",
                "--model",
                "m",
                "--train_image",
                "imgs",
                "--train_label",
                "lbls",
                "--yaml_path",
                "data.yaml",
                "--extra_body",
                '{"a": 1}',
            ]
        )
        assert args.extra_body == {"a": 1}

    def test_standalone_cli_bad_json_errors(self):
        from auto_annotation.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--output_folder",
                    "./out",
                    "--model",
                    "m",
                    "--train_image",
                    "imgs",
                    "--train_label",
                    "lbls",
                    "--yaml_path",
                    "data.yaml",
                    "--extra_body",
                    "not-json",
                ]
            )


class TestForwarding:
    def _client(self, captured):
        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)

                class _Msg:
                    content = '{"class": "spot", "confidence": 5}'

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        return _Client()

    def test_detect_defect_sends_extra_body(self):
        import numpy as np

        from auto_annotation.image_io import detect_defect

        captured = {}
        out = detect_defect(
            np.zeros((16, 16, 3), dtype=np.uint8),
            self._client(captured),
            "m",
            ["spot"],
            extra_body={"provider": {"order": ["x"]}},
        )
        assert out["class"] == "spot"
        assert captured["extra_body"] == {"provider": {"order": ["x"]}}

    def test_detect_defect_omits_extra_body_when_unset(self):
        import numpy as np

        from auto_annotation.image_io import detect_defect

        captured = {}
        detect_defect(
            np.zeros((16, 16, 3), dtype=np.uint8),
            self._client(captured),
            "m",
            ["spot"],
        )
        assert "extra_body" not in captured

    def test_detect_defect_accepts_json_string(self):
        import numpy as np

        from auto_annotation.image_io import detect_defect

        captured = {}
        detect_defect(
            np.zeros((16, 16, 3), dtype=np.uint8),
            self._client(captured),
            "m",
            ["spot"],
            extra_body='{"a": 1}',
        )
        assert captured["extra_body"] == {"a": 1}

    def test_process_one_image_threads_it_through(self, tmp_path, monkeypatch):
        from PIL import Image

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats
        import threading

        seen = {}
        img_name = "img0.jpg"
        (tmp_path / "imgs").mkdir(exist_ok=True)
        (tmp_path / "lbls").mkdir(exist_ok=True)
        Image.new("RGB", (64, 64), (128, 128, 128)).save(tmp_path / "imgs" / img_name)
        (tmp_path / "lbls" / "img0.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        def _fake_detect(crop_image, client, model_name, known, **kwargs):
            seen.update(kwargs)
            return {"class": "spot", "confidence": 5}

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _fake_detect)
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        process_one_image(
            img_name,
            str(tmp_path / "imgs"),
            str(tmp_path / "lbls"),
            str(out),
            {},
            threading.Lock(),
            object(),
            "m",
            2,
            False,
            False,
            64,
            64,
            RunStats(),
            False,
            extra_body={"k": "v"},
        )
        assert seen.get("extra_body") == {"k": "v"}
        assert (out / "img0.txt").exists()
