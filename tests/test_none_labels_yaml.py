"""none_labels accepts a YAML list or a comma-separated string."""

import yaml


class TestNoneLabelsCoercion:
    def test_list_joins_to_string(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(
            task="auto_label",
            train_image="imgs/",
            none_labels=["none", "background", "unknown"],
        )
        assert cfg.none_labels == "none,background,unknown"

    def test_list_strips_blanks(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(
            task="auto_label", train_image="imgs/", none_labels=[" a ", "", "b"]
        )
        assert cfg.none_labels == "a,b"

    def test_string_passthrough(self):
        from schemes import PipelineConfig

        cfg = PipelineConfig(
            task="auto_label", train_image="imgs/", none_labels="none,background"
        )
        assert cfg.none_labels == "none,background"

    def test_config_yaml_list_end_to_end(self, tmp_path):
        from main import parse_args

        yp = tmp_path / "cfg.yaml"
        yp.write_text(
            yaml.safe_dump(
                {
                    "task": "auto_label",
                    "train_image": "imgs/",
                    "none_labels": ["none", "no_detection", "background"],
                }
            )
        )
        cfg = parse_args(["--config", str(yp)])
        assert cfg.none_labels == "none,no_detection,background"

    def test_example_yaml_loads(self):
        # The shipped example uses list form -- it must validate.
        from pathlib import Path

        from main import _load_yaml_config
        from schemes import PipelineConfig

        ex = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "auto_label_vllm.example.yaml"
        )
        data = _load_yaml_config(str(ex))
        assert isinstance(data.get("none_labels"), list)
        cfg = PipelineConfig(**{**data, "train_image": "imgs/"})
        assert "background" in cfg.none_labels.split(",")


class TestNoneLabelsRuntime:
    def test_prompt_renderer_accepts_list(self):
        from free_detection.agent.prompts import render_auto_label_prompt

        out = render_auto_label_prompt(
            ["spot"], none_labels=["none", "background"], drop_none=True
        )
        assert "'none'" in out and "'background'" in out

    def test_prompt_renderer_accepts_string(self):
        from free_detection.agent.prompts import render_auto_label_prompt

        out = render_auto_label_prompt(
            ["spot"], none_labels="none,background", drop_none=True
        )
        assert "'none'" in out and "'background'" in out

    def test_parse_none_labels_list(self):
        from auto_annotation.single_image import _parse_none_labels

        assert _parse_none_labels(["None", "background"]) == {"none", "background"}
        assert _parse_none_labels("none,background") == {"none", "background"}
