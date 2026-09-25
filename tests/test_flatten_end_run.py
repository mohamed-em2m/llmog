"""End-of-run flatten: batch_XXXX labels -> flat top-level labels/."""

import os
import time

from auto_annotation.reverse_batches import flatten_batches_to_labels


def _make_output(tmp_path):
    labels = tmp_path / "labels"
    b0 = labels / "batch_0000"
    b1 = labels / "batch_0001"
    b0.mkdir(parents=True)
    b1.mkdir(parents=True)
    return labels, b0, b1


class TestFlattenBatchesToLabels:
    def test_best_copy_wins(self, tmp_path):
        _, b0, b1 = _make_output(tmp_path)
        (b0 / "img1.txt").write_text("")  # empty copy loses
        (b1 / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (b0 / "img2.txt").write_text("1 0.1 0.1 0.3 0.3\n")

        out = flatten_batches_to_labels(tmp_path)
        assert (out / "img1.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
        assert (out / "img2.txt").read_text() == "1 0.1 0.1 0.3 0.3\n"
        # batch dirs untouched (resume still works)
        assert (b0 / "img1.txt").exists()
        assert (b1 / "img1.txt").exists()

    def test_newest_wins_on_tie(self, tmp_path):
        _, b0, b1 = _make_output(tmp_path)
        (b0 / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        old = time.time() - 100
        os.utime(b0 / "a.txt", (old, old))
        (b1 / "a.txt").write_text("1 0.2 0.2 0.2 0.2\n")

        flatten_batches_to_labels(tmp_path)
        assert (tmp_path / "labels" / "a.txt").read_text() == "1 0.2 0.2 0.2 0.2\n"

    def test_idempotent_rerun(self, tmp_path):
        _, b0, _ = _make_output(tmp_path)
        (b0 / "x.txt").write_text("2 0.5 0.5 0.5 0.5\n")

        flatten_batches_to_labels(tmp_path)
        first = (tmp_path / "labels" / "x.txt").read_text()
        flatten_batches_to_labels(tmp_path)  # second run: skips same-file copy
        assert (tmp_path / "labels" / "x.txt").read_text() == first

    def test_low_conf_jsons_copied_with_prefix(self, tmp_path):
        _, b0, _ = _make_output(tmp_path)
        (b0 / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (b0 / "x_low_confidence.json").write_text("[]")

        flatten_batches_to_labels(tmp_path)
        assert (tmp_path / "labels" / "batch_0000_x_low_confidence.json").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        _, b0, _ = _make_output(tmp_path)
        (b0 / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        flatten_batches_to_labels(tmp_path, dry_run=True)
        assert not (tmp_path / "labels" / "x.txt").exists()

    def test_cli_flags_default_on(self):
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
            ]
        )
        assert args.flatten is True
        args2 = parse_args(
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
                "--no_flatten",
            ]
        )
        assert args2.flatten is False
