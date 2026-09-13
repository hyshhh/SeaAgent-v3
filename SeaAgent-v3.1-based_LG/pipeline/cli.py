"""SeaAgent 视频监控流水线命令行入口。"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
console = Console()
_MIN_PIPE_SCALE = 0.05

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seaagent-pipeline", description="船舶检测、跟踪与轨迹记忆构建流水线")
    parser.add_argument("source", nargs="?", help="视频文件、相机编号、网络视频地址或帧目录")
    parser.add_argument("--playlist-json", default=None, help="按顺序处理的视频源 JSON 数组")
    parser.add_argument("--segment-gap-seconds", type=float, default=0.0, help="相邻视频片段之间的模拟时间间隔")
    parser.add_argument("--playlist-failure-policy", choices=("skip", "stop"), default="skip", help="片段失败后跳过或终止整个序列")
    parser.add_argument("--output", "-o", help="输出视频路径")
    parser.add_argument("--demo", action="store_true", default=None, help="绘制检测框和轨迹记忆状态")
    parser.add_argument("--display", action="store_true", help="显示实时窗口")
    parser.add_argument("--max-frames", type=int, default=0, help="最大处理帧数，零表示不限制")
    parser.add_argument("--yolo-model", default=None, help="检测模型路径")
    parser.add_argument("--device", default=None, help="推理设备")
    parser.add_argument("--conf", type=float, default=None, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="交并比阈值")
    parser.add_argument("--detect-every", type=int, default=None, help="每多少帧执行一次检测")
    parser.add_argument("--track-high-thresh", type=float, default=None, help="跟踪高分阈值")
    parser.add_argument("--track-low-thresh", type=float, default=None, help="跟踪低分阈值")
    parser.add_argument("--new-track-thresh", type=float, default=None, help="新建轨迹阈值")
    parser.add_argument("--match-thresh", type=float, default=None, help="跟踪关联阈值")
    parser.add_argument("--track-buffer", type=int, default=None, help="跟踪缓冲帧数")
    parser.add_argument("--max-stale-frames", type=int, default=None, help="轨迹最大保持帧数")
    parser.add_argument("--target-fps", type=float, default=None, help="目标处理帧率，零表示不限制")
    parser.add_argument("--monitor-start-time", type=float, default=None, help="连续监控序列中的模拟起始时间戳")
    parser.add_argument("--camera", action="store_true", help="相机输入标记")
    parser.add_argument("--frames-dir", default=None, help="浏览器摄像头共享帧目录")
    parser.add_argument("--virtual-fps", type=float, default=15.0, help="共享帧目录的虚拟帧率")
    parser.add_argument("--stream-dir", default=None, help="逐帧写入预览图的目录")
    parser.add_argument("--no-output", action="store_true", help="不保存结果视频")
    parser.add_argument("--save-output-video", action="store_true", default=None, help="保存结果视频")
    parser.add_argument("--no-save-output-video", action="store_true", default=None, help="不保存结果视频")
    parser.add_argument("--raw-stdout", action="store_true", help="将原始帧写入标准输出")
    parser.add_argument("--output-size", default=None, help="输出尺寸，格式为宽x高")
    parser.add_argument("--pipe-scale", type=float, default=None, help="输出缩放比例")
    parser.add_argument("--stop-file", default=None, help="停止信号文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细日志")
    return parser

def _merge_args_to_config(args, config: dict) -> dict:
    pipeline = config.setdefault("pipeline", {})
    if args.demo is not None:
        pipeline["demo"] = args.demo
    if args.yolo_model is not None:
        pipeline["yolo_model"] = args.yolo_model
    if args.device is not None:
        pipeline["device"] = args.device
    if args.conf is not None:
        pipeline["conf_threshold"] = args.conf
    if args.iou is not None:
        pipeline["iou_threshold"] = args.iou
    if args.detect_every is not None:
        pipeline["detect_every_n_frames"] = max(1, args.detect_every)
    tracker_params = pipeline.setdefault("tracker_params", {})
    if args.track_high_thresh is not None:
        tracker_params["track_high_thresh"] = args.track_high_thresh
    if args.track_low_thresh is not None:
        tracker_params["track_low_thresh"] = args.track_low_thresh
    if args.new_track_thresh is not None:
        tracker_params["new_track_thresh"] = args.new_track_thresh
    if args.match_thresh is not None:
        tracker_params["match_thresh"] = args.match_thresh
    if args.track_buffer is not None:
        tracker_params["track_buffer"] = max(1, args.track_buffer)
    if args.max_stale_frames is not None:
        pipeline["max_stale_frames"] = max(1, args.max_stale_frames)
    if args.target_fps is not None:
        pipeline["target_fps"] = max(0.0, args.target_fps)
    if args.monitor_start_time is not None:
        pipeline["monitor_start_time"] = max(0.0, args.monitor_start_time)
    if args.no_output:
        pipeline["no_output"] = True
    if args.save_output_video is not None:
        pipeline["save_output_video"] = args.save_output_video
    elif args.no_save_output_video is not None:
        pipeline["save_output_video"] = not args.no_save_output_video
    if args.raw_stdout:
        pipeline["raw_stdout"] = True
    if args.output_size:
        try:
            width, height = args.output_size.lower().split("x")
            pipeline["output_size"] = [int(width), int(height)]
        except ValueError:
            console.print("[red]--output-size 应为宽x高，例如 640x480[/red]")
            sys.exit(1)
    if args.pipe_scale is not None and _MIN_PIPE_SCALE <= args.pipe_scale < 1.0 and pipeline.get("output_size"):
        width, height = pipeline["output_size"]
        pipeline["pipe_output_size"] = [max(16, int(width * args.pipe_scale)) // 2 * 2, max(16, int(height * args.pipe_scale)) // 2 * 2]
    if args.stop_file:
        pipeline["stop_file"] = args.stop_file
    return config

def _print_config(args, config: dict) -> None:
    pipeline = config.get("pipeline", {})
    source = f"帧目录：{args.frames_dir}" if args.frames_dir else (f"播放列表：{args.playlist_json}" if args.playlist_json else args.source)
    lines = [
        "┌─ SeaAgent 流水线配置 ──────────────────┐",
        f"│ 输入源：{source}",
        f"│ 检测间隔：每 {pipeline.get('detect_every_n_frames', 1)} 帧",
        f"│ 候选间隔：每 {pipeline.get('candidate_every_n_frames', 10)} 帧",
        f"│ 正式池容量：{pipeline.get('keyframe_pool_size', 6)} 帧",
        f"│ 检测模型：{pipeline.get('yolo_model', 'yolov8n.pt')}",
        "└─────────────────────────────────────────┘",
    ]
    for line in lines:
        console.print(line)

def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.raw_stdout:
        console.file = sys.stderr
    from config import load_config
    config = _merge_args_to_config(args, load_config())
    _print_config(args, config)
    try:
        from pipeline.pipeline import ShipPipeline
        if args.frames_dir:
            from pipeline.virtual_camera import VirtualCamera
            frames_path = Path(args.frames_dir)
            if not frames_path.exists():
                console.print(f"[red]帧目录不存在：{frames_path}[/red]")
                sys.exit(1)
            source = VirtualCamera(frames_path, fps=args.virtual_fps)
        else:
            source = args.source
        pipeline = ShipPipeline(config=config)
        if args.playlist_json:
            try:
                playlist = json.loads(args.playlist_json)
            except json.JSONDecodeError as error:
                raise ValueError(f"播放列表 JSON 格式错误：{error}") from error
            if not isinstance(playlist, list) or not playlist or not all(isinstance(item, str) and item.strip() for item in playlist):
                raise ValueError("播放列表必须是非空的视频源字符串数组")

            def emit_segment(event: dict) -> None:
                target = sys.stderr if args.raw_stdout else sys.stdout
                print(f"__PIPELINE_SEGMENT__:{json.dumps(event, ensure_ascii=False)}", file=target, flush=True)

            stats = pipeline.process_playlist(
                sources=playlist,
                output_path=args.output,
                display=args.display and not args.stream_dir,
                max_frames=args.max_frames,
                stream_dir=args.stream_dir,
                segment_gap_seconds=args.segment_gap_seconds,
                failure_policy=args.playlist_failure_policy,
                segment_callback=emit_segment,
            )
        else:
            if source is None:
                raise ValueError("请提供输入源或 --playlist-json")
            stats = pipeline.process(
                source=source,
                output_path=args.output,
                display=args.display and not args.stream_dir,
                max_frames=args.max_frames,
                stream_dir=args.stream_dir,
            )
        table = Table(title="处理统计")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="white")
        for key, value in stats.items():
            table.add_row(key, str(value))
        console.print(table)
        target = sys.stderr if args.raw_stdout else sys.stdout
        print(f"\n__PIPELINE_SUMMARY__:{json.dumps(stats, ensure_ascii=False)}", file=target, flush=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
    except Exception as error:
        console.print(f"\n[red]错误：{error}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
