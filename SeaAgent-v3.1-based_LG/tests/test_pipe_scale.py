from pipeline.cli import _merge_args_to_config, build_parser


def test_pipe_scale_point_zero_five_is_applied():
    args = build_parser().parse_args([
        "demo.mp4",
        "--output-size",
        "1080x1920",
        "--pipe-scale",
        "0.05",
    ])

    config = _merge_args_to_config(args, {"pipeline": {}})

    assert config["pipeline"]["pipe_output_size"] == [54, 96]
