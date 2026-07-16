"""CLI argument parsing for the auto-annotation pipeline."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relabel binary defect/no-defect YOLO annotations into multi-class defect labels using a VLM."
    )
    # Logging Configuration
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to a file where logs should also be written.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Where to save the relabeled output.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use for classification."
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="API key for the external/hosted model."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base URL for the external/hosted model.",
    )
    parser.add_argument(
        "--server_type",
        type=str,
        default="llama_cpp",
        choices=["llama_cpp", "vllm", "external"],
        help="Use a local llama.cpp server instead of an external API.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable the model's thinking/reasoning mode on the local llama.cpp server "
        "(--use_local_model only). Off by default: faster and usually unnecessary for a "
        "single classify-this-crop call.",
    )
    parser.add_argument(
        "--use_mtp",
        action="store_true",
        default=True,
        help="Enable draft-MTP speculative decoding on the local llama.cpp server "
        "(--use_local_model only). On by default for speed; pass --no_mtp to disable it "
        "if you hit compatibility issues with a given model/build.",
    )
    parser.add_argument(
        "--no_mtp",
        action="store_false",
        dest="use_mtp",
        help="Disable draft-MTP speculative decoding (--use_local_model only).",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=20000,
        help="Context size for the local llama.cpp server (--use_local_model only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the local llama.cpp server (--use_local_model only).",
    )
    parser.add_argument(
        "--parallel_slots",
        type=int,
        default=1,
        help="Number of parallel inference slots on the local llama.cpp server "
        "(--use_local_model only). If you raise this, --max_workers can be raised to match "
        "so multiple images are in flight at once.",
    )
    parser.add_argument(
        "--train_image",
        type=str,
        required=True,
        help="Path to the folder of training images.",
    )
    parser.add_argument(
        "--train_label",
        type=str,
        required=True,
        help="Path to the folder of YOLO training labels.",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the dataset yaml file (data.yaml).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Confidence (1-5) at/below which a box is ALSO logged to a *_low_confidence.json for manual review.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process this many images (quick sanity-check run instead of the full dataset). "
        "Applied after --start_index/--end_index.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order (seeded by --seed) before applying --start_index/--end_index/"
        "--num_samples, so a sample isn't just the first N images alphabetically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set, for reproducible samples.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index (0-based, inclusive) of the image range to process, applied after "
        "--shuffle and before --num_samples. Useful for splitting a large dataset across "
        "multiple runs/machines, e.g. --start_index 0 --end_index 1000 on one machine and "
        "--start_index 1000 --end_index 2000 on another.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index (0-based, exclusive) of the image range to process. See --start_index.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't call the model and don't write any files; just print what would happen.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Legacy resume check: skip images whose output label file already exists on disk. "
        "Has no effect when --inplace_saving is set (there is no separate output file to check "
        "in that mode) -- use --auto_resume instead, which is on by default. Auto-resume is the "
        "more robust option in general since it only counts an image as done once it's confirmed "
        "in the checkpoint; you can leave --resume off in normal usage.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="Automatically resume from '<output-folder>/.checkpoint.json' if one exists "
        "(on by default). This is what lets an interrupted/crashed run be continued by simply "
        "re-running the exact same command -- already-finished images and batches are skipped "
        "and the accumulated class_map is restored.",
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Disable auto-resume and ignore/clear any existing checkpoint, starting completely fresh.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of images per batch (default: 50). When not using --inplace_saving, each "
        "batch's relabeled annotations are written to their own 'batch_XXXX' subfolder under "
        "--output-folder, and a checkpoint marks each batch done as soon as it finishes, so a "
        "resumed run can skip whole finished batches quickly. Pass 0 to disable batching "
        "(single flat output folder, same as before).",
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of image file extensions to process.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of images to process concurrently (thread pool). Keep at 1 for a local "
        "llama.cpp server with parallel_slots=1; raise it for a remote API that supports "
        "concurrent requests.",
    )
    # vLLM Configuration Options
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=20000,
        help="vLLM maximum model length.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="vLLM pipeline parallel size.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM data type.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="vLLM quantization method.",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        default="auto",
        help="vLLM KV cache data type.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="vLLM maximum number of sequences.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Enforce eager execution in vLLM.",
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        default=True,
        help="Enable chunked prefill in vLLM.",
    )
    parser.add_argument(
        "--no_chunked_prefill",
        action="store_false",
        dest="enable_chunked_prefill",
        help="Disable chunked prefill in vLLM.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=True,
        help="Enable prefix caching in vLLM.",
    )
    parser.add_argument(
        "--no_prefix_caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="Disable prefix caching in vLLM.",
    )
    parser.add_argument(
        "--speculative_model",
        type=str,
        default=None,
        help="vLLM speculative model name.",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=None,
        help="Number of speculative tokens.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code in vLLM.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not trust remote code in vLLM.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="vLLM model download directory.",
    )
    parser.add_argument(
        "--limit_mm_per_prompt",
        type=str,
        default=None,
        help="vLLM limit multimodal items per prompt.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="vLLM chat template.",
    )
    parser.add_argument(
        "--extra_args",
        action="append",
        default=None,
        help="Extra arguments to pass to vLLM server.",
    )
    parser.add_argument(
        "--image_min_tokens",
        type=int,
        default=1024,
        help="Minimum number of tokens to use for image encoding.",
    )
    parser.add_argument(
        "--image_max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to use for image encoding.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the input image for the model.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the input image for the model.",
    )
    parser.add_argument(
        "--init_class_map",
        action="store_true",
        help="Initialize the class map from the YAML file.",
    )
    parser.add_argument(
        "--inplace_saving",
        action="store_true",
        help="save inplace the relabeled annotations in the original label folder instead of a separate output folder.",
    )
    args = parser.parse_args()

    if args.start_index is not None and args.start_index < 0:
        parser.error("--start_index must be >= 0")
    if (
        args.end_index is not None
        and args.start_index is not None
        and args.end_index <= args.start_index
    ):
        parser.error("--end_index must be greater than --start_index")

    return args
