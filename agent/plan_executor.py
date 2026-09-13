"""按 PlanAgent 产出的 calls 确定性执行工具（对齐 old Observer）。

完整结果写入 working_scope，供后续 $ref 与最终合成；
回传给模型 / Reflect 的仅是压缩摘要，不把关键帧大 JSON 塞进对话。

定位：observe_node 的执行内核。模型只负责「提议调用哪些工具」，本文件负责
「这些调用是否合法、依赖是否就绪、参数是否齐全，然后真正执行」。所有失败都被
降级为 skip 记录，单个 call 出错不影响同批其他 call。

文件分区（以 P 编号为锚点检索）：
    P1  工具契约白名单      必填/允许参数，校验的唯一事实源
    P2  执行主流程          execute：逐 call 六步判断
    P3  依赖与契约校验      $ref 依赖、三层参数校验、skip 记录
    P4  参数富化与归一化    从 scope 补参；dedupTracks 入参规整
    P5  scope 收集器        汇总历史结果，供 P4 补参
    P6  事件与 $ref 解析    事件派发 + 引用求值（$ref/$slice/$default/$map）
    P7  摘要压缩            完整结果 → 回传模型的摘要
    P8  调用去重与语义签名  semantic_signature / sanitize_calls
"""
from __future__ import annotations

import json
from typing import Any, Callable


class PlanExecutor:
    """确定性执行器：解析 $ref、校验依赖、调用 ToolService。"""

    # ------------------------------------------------------------------------
    # P1 工具契约白名单
    # _REQUIRED_ARGUMENTS 声明每个工具必须提供的参数；_ALLOWED_ARGUMENTS 声明
    # 允许出现的参数。二者是参数校验的唯一事实源 —— 工具新增参数必须同步到这里，
    # 否则会被判为契约不符而 skip（lc_tools 的工具定义需与之保持一致）。
    # ------------------------------------------------------------------------
    _REQUIRED_ARGUMENTS = {
        "getFrames": ("trackIds",),
        "getClip": ("trackId",),
        "getRegistry": ("hullNumber",),
        "matchHull": ("hullNumberArray",),
        "matchText": ("description", "galleryImages"),
        "matchImage": ("queryImages", "galleryImages"),
        "dedupTracks": ("tracks", "keyframesByTrack"),
    }
    _ALLOWED_ARGUMENTS = {
        "getTrack": frozenset({"timeRange", "hullNumber", "finalMatchType", "offset", "limit"}),
        "getFrames": frozenset({"trackIds"}),
        "getClip": frozenset({"trackId", "timeRange", "scale"}),
        "getRegistry": frozenset({"hullNumber"}),
        "listRegistry": frozenset(),
        "matchHull": frozenset({"hullNumberArray"}),
        "matchText": frozenset({"description", "galleryImages", "topK"}),
        "matchImage": frozenset({"queryImages", "galleryImages", "topK", "registryItems"}),
        "verifyTarget": frozenset({"description", "registryReferenceIds", "keyframeIds", "shipSegmentIds"}),
        "showEvidence": frozenset({"keyframeIds", "shipSegmentIds", "registryReferenceIds"}),
        "dedupTracks": frozenset({"tracks", "keyframesByTrack"}),
    }

    def __init__(self, tools: Any):
        self.tools = tools

    # ------------------------------------------------------------------------
    # P2 执行主流程：execute —— 逐 call 确定性执行
    # 对每个 call 依次走六步判断（下文标注 ①–⑥），任一环不通过都只 skip 该
    # call、不中断后续。全部执行完后汇总 observations / scope / summary /
    # tool_records，其中 scope 就是写回 working_scope 的完整结果。
    # ------------------------------------------------------------------------
    def execute(
        self,
        calls: list[dict[str, Any]],
        scope: dict[str, Any] | None = None,
        on_tool_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        working = dict(scope or {})
        observations: list[dict[str, Any]] = []
        tool_chain: list[str] = []
        tool_records: list[dict[str, Any]] = []

        for index, call in enumerate(calls or []):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or f"step-{index + 1}")
            tool = str(call.get("tool") or "").strip()
            if not tool:
                continue

            # ① 条件判断：condition 不满足直接 skip，不进依赖与契约校验
            if not self._condition(call.get("condition"), working):
                result = {"ok": False, "error": "condition_not_met", "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "arguments": {},
                    "skipped": True,
                    "skipReason": "condition_not_met",
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                tool_records.append(self._skipped_record(observation))
                self._emit(on_tool_event, "skipped", observation)
                continue

            # ② matchImage 特例分支：本分支跳过 ③ 依赖预检，因为配图可能来自
            #    $ref、本轮 getFrames 或本轮 listRegistry 三处，$ref 解析为空不
            #    等于无图可用。故先 resolve + 从 scope 补图；参数仍不可用时记
            #    软失败（ok=True + visualAttempted），使 reflect 能区分
            #    「没试过」与「试了但没法试」，而不是直接 skip。
            if tool == "matchImage":
                arguments = self._resolve(call.get("arguments", {}), working)
                arguments = self._enrich_match_image_args(arguments, working)
                argument_issue = self._argument_contract_issue(tool, arguments) or self._required_argument_issue(tool, arguments)
                if argument_issue:
                    result = {
                        "ok": True,
                        "matchMode": "image_to_image",
                        "matches": [],
                        "error": argument_issue,
                        "hint": "无可搜库参考图或关键帧，视觉匹配未执行",
                        "visualAttempted": True,
                    }
                    observation = {
                        "id": call_id,
                        "tool": tool,
                        "arguments": self._compact_args(arguments),
                        "result": result,
                        "skipped": False,
                    }
                    working[call_id] = result
                    observations.append(observation)
                    summary = self.summarize_observation(observation)
                    tool_records.append({
                        "id": call_id,
                        "tool": tool,
                        "arguments": observation.get("arguments") or {},
                        "result": result,
                        "summary": summary,
                        "ok": True,
                        "skipped": False,
                        "phase": "completed",
                        "error": argument_issue,
                        **summary,
                    })
                    tool_chain.append(tool)
                    self._emit(on_tool_event, "completed", observation)
                    continue
                self._emit(
                    on_tool_event,
                    "running",
                    {"id": call_id, "tool": tool, "arguments": self._compact_args(arguments)},
                )
                try:
                    result = self.tools.execute(tool, arguments)
                    if not isinstance(result, dict):
                        result = {"ok": False, "error": "tool_result_invalid", "tool": tool}
                except Exception as error:
                    result = {"ok": False, "error": f"tool_execution_failed:{error}", "tool": tool}
                if isinstance(result, dict):
                    result.setdefault("visualAttempted", True)
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "arguments": self._compact_args(arguments),
                    "result": result,
                    "skipped": False,
                }
                working[call_id] = result
                observations.append(observation)
                tool_chain.append(tool)
                summary = self.summarize_observation(observation)
                tool_records.append({
                    "id": call_id,
                    "tool": tool,
                    "arguments": observation.get("arguments") or {},
                    "result": result,
                    "summary": summary,
                    "ok": result.get("ok") is not False,
                    "skipped": False,
                    "phase": "completed" if result.get("ok") is not False else "failed",
                    "error": result.get("error"),
                    **summary,
                })
                self._emit(
                    on_tool_event,
                    "completed" if result.get("ok") is not False else "failed",
                    observation,
                )
                continue

            # ③ 依赖预检：$ref 指向的根步骤若失败/为空，本 call 直接 skip
            dependency_issue = self._dependency_issue(call.get("arguments", {}), working)
            if dependency_issue:
                result = {"ok": False, "error": dependency_issue, "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "arguments": {},
                    "skipped": True,
                    "skipReason": dependency_issue,
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                tool_records.append(self._skipped_record(observation))
                self._emit(on_tool_event, "skipped", observation)
                continue

            # ④ 引用解析 + 参数富化：$ref 求值，dedupTracks 另从 scope 补 tracks
            arguments = self._resolve(call.get("arguments", {}), working)
            if tool == "dedupTracks":
                arguments = self._enrich_dedup_tracks_args(arguments, working)
            # ⑤ 契约校验：白名单 / 必填 / 解析后取值 三层，任一不过即 skip
            argument_issue = (
                self._argument_contract_issue(tool, arguments)
                or self._required_argument_issue(tool, arguments)
                or self._resolved_argument_issue(tool, arguments)
            )
            if argument_issue:
                result = {"ok": False, "error": argument_issue, "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "arguments": self._compact_args(arguments) if isinstance(arguments, dict) else {},
                    "skipped": True,
                    "skipReason": argument_issue,
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                tool_records.append(self._skipped_record(observation))
                self._emit(on_tool_event, "skipped", observation)
                continue

            # ⑥ 执行与记录：调 ToolService.execute，结果写 working_scope（供后续
            #    $ref），同时产出 observation / tool_record / 前端事件。工具异常
            #    在此被吞成 ok=False，不向外抛。
            self._emit(on_tool_event, "running", {"id": call_id, "tool": tool, "arguments": self._compact_args(arguments)})
            try:
                result = self.tools.execute(tool, arguments)
                if not isinstance(result, dict):
                    result = {"ok": False, "error": "tool_result_invalid", "tool": tool}
            except Exception as error:
                result = {"ok": False, "error": f"tool_execution_failed:{error}", "tool": tool}

            observation = {
                "id": call_id,
                "tool": tool,
                "arguments": self._compact_args(arguments),
                "result": result,
                "skipped": False,
            }
            working[call_id] = result
            observations.append(observation)
            tool_chain.append(tool)
            summary = self.summarize_observation(observation)
            tool_records.append({
                "id": call_id,
                "tool": tool,
                "arguments": observation.get("arguments") or {},
                "result": result,
                "summary": summary,
                "ok": result.get("ok") is not False,
                "skipped": False,
                "phase": "completed" if result.get("ok") is not False else "failed",
                "error": result.get("error"),
                **summary,
            })
            self._emit(
                on_tool_event,
                "completed" if result.get("ok") is not False else "failed",
                observation,
            )

        # ⑦ 汇总：executedCount 不含 skip；依赖未满足的 skip 不计入 failedCount
        summary = {
            "calls": [self.summarize_observation(item) for item in observations],
            "executedCount": sum(1 for item in observations if not item.get("skipped")),
            # 依赖未满足而 skip 不算失败
            "failedCount": sum(
                1
                for item in observations
                if not item.get("skipped") and (item.get("result") or {}).get("ok") is False
            ),
            "skippedCount": sum(1 for item in observations if item.get("skipped")),
        }
        return {
            "observations": observations,
            "scope": working,
            "summary": summary,
            "tool_chain": tool_chain,
            "tool_records": tool_records,
        }

    # ------------------------------------------------------------------------
    # P3 依赖与契约校验
    # _dependency_issue            检查 $ref 指向的根步骤是否可用（ok=False 视为不可用）
    # _empty_dependency            判定依赖值是否为空
    # _argument_contract_issue     参数超出白名单？
    # _required_argument_issue     必填参数缺失？
    # _resolved_argument_issue     解析后取值是否有效？
    # call_contract_issues         计划级批量校验，供 graph 判定是否触发契约守卫
    # _skipped_record              统一构造 skip 记录
    # ------------------------------------------------------------------------
    @classmethod
    def _dependency_issue(cls, value: Any, scope: dict[str, Any]) -> str | None:
        if isinstance(value, dict):
            if "$ref" in value:
                reference = str(value.get("$ref") or "").strip()
                if not reference:
                    return "dependency_reference_empty"
                root = reference.split(".", 1)[0]
                if root not in scope:
                    return f"dependency_missing:{reference}"
                source = scope.get(root)
                if isinstance(source, dict) and source.get("ok") is False:
                    return f"dependency_failed:{root}"
                present, resolved = cls._read_with_presence(scope, reference)
                if not present:
                    if "$default" in value:
                        # $default 可以是字面量或嵌套 $ref
                        default_value = value["$default"]
                        if isinstance(default_value, dict) and "$ref" in default_value:
                            issue = cls._dependency_issue(default_value, scope)
                            if issue:
                                return issue
                            resolved = cls._read(scope, str(default_value.get("$ref") or ""))
                        else:
                            resolved = default_value
                    else:
                        return f"dependency_field_missing:{reference}"
                if cls._empty_dependency(resolved):
                    if "$default" in value:
                        default_value = value["$default"]
                        if isinstance(default_value, dict) and "$ref" in default_value:
                            issue = cls._dependency_issue(default_value, scope)
                            if issue:
                                return issue
                        # 空主引用时允许回退 default，不在此判 empty
                    else:
                        return f"dependency_empty:{reference}"
            for item in value.values():
                if isinstance(value, dict) and "$ref" in value and item is value.get("$default"):
                    continue
                issue = cls._dependency_issue(item, scope)
                if issue:
                    return issue
        elif isinstance(value, list):
            for item in value:
                issue = cls._dependency_issue(item, scope)
                if issue:
                    return issue
        return None

    @staticmethod
    def _empty_dependency(value: Any) -> bool:
        return value is None or value == "" or (isinstance(value, (list, tuple, set, dict)) and not value)

    @classmethod
    def _argument_contract_issue(cls, tool: str, arguments: dict[str, Any]) -> str | None:
        allowed = cls._ALLOWED_ARGUMENTS.get(tool)
        if allowed is None:
            return f"tool_not_allowed:{tool}"
        extra = sorted(str(key) for key in (arguments or {}) if key not in allowed)
        if extra:
            return f"argument_not_allowed:{tool}:{','.join(extra)}"
        return None

    @classmethod
    def call_contract_issues(cls, calls: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "").strip()
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            issue = cls._argument_contract_issue(tool, arguments)
            if issue:
                issues.append(issue)
        return issues

    @classmethod
    def _required_argument_issue(cls, tool: str, arguments: dict[str, Any]) -> str | None:
        for field in cls._REQUIRED_ARGUMENTS.get(tool, ()):
            if cls._empty_dependency(arguments.get(field)):
                return f"argument_missing:{field}"
        if tool == "showEvidence" and not any(
            arguments.get(field) for field in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")
        ):
            return "argument_missing:evidence"
        return None

    @classmethod
    def _resolved_argument_issue(cls, tool: str, arguments: dict[str, Any]) -> str | None:
        """校验已经解析完 $ref 的运行时参数，避免类型错误进入业务工具。"""
        if tool != "dedupTracks":
            return None
        tracks = arguments.get("tracks")
        if not isinstance(tracks, list) or any(
            not isinstance(item, dict) or item.get("trackId") is None
            for item in tracks
        ):
            return "argument_invalid:dedupTracks:tracks_requires_track_records"
        grouped = arguments.get("keyframesByTrack")
        if not isinstance(grouped, dict):
            return "argument_invalid:dedupTracks:keyframesByTrack_requires_object"
        return None

    @staticmethod
    def _skipped_record(observation: dict[str, Any]) -> dict[str, Any]:
        """跳过步骤也写入 tool_records，供 Reflect 识别「已尝试 matchImage」。"""
        summary = PlanExecutor.summarize_observation(observation)
        return {
            "id": observation.get("id"),
            "tool": observation.get("tool"),
            "arguments": observation.get("arguments") or {},
            "result": observation.get("result") or {},
            "summary": summary,
            "ok": False,
            "skipped": True,
            "phase": "skipped",
            "skipReason": observation.get("skipReason"),
            "error": observation.get("skipReason") or (observation.get("result") or {}).get("error"),
            **summary,
        }

    # ------------------------------------------------------------------------
    # P4 参数富化与归一化
    # _enrich_match_image_args / _enrich_dedup_tracks_args
    #     从 scope 补参数（$ref 之外的另一条补参路径）
    # _normalize_dedup_tracks 及 _keyframes_by_track_from / _group_keyframes /
    # _normalize_keyframe_groups / _expand_to_registry_images
    #     把 dedupTracks 的松散入参（外部 id、分组形态、关键帧结构）统一成工具
    #     要求的规范形态
    # ------------------------------------------------------------------------
    @classmethod
    def _enrich_match_image_args(cls, arguments: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        """保证 query=库参考图、gallery=关键帧；空列表/库项外壳时从 scope 展开补齐。"""
        args = dict(arguments or {})
        # 1) 先把 registryItems 外壳 / 嵌套 references 展成参考图列表
        query = cls._expand_to_registry_images(args.get("queryImages"))
        if not query:
            query = cls._collect_registry_images(scope)
        if query:
            args["queryImages"] = query
        if cls._empty_dependency(args.get("registryItems")):
            registry_items = cls._collect_registry_items(scope)
            if registry_items:
                args["registryItems"] = registry_items
        gallery = args.get("galleryImages")
        if cls._empty_dependency(gallery) or (
            isinstance(gallery, list) and gallery and not any(
                isinstance(x, dict) and x.get("keyframeId") is not None for x in gallery
            )
        ):
            recovered = cls._collect_keyframes(scope)
            if recovered:
                args["galleryImages"] = recovered
        return args

    @classmethod
    def _enrich_dedup_tracks_args(cls, arguments: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        """把模型可能传入的 frames / getFrames 结果兜底整理为 dedupTracks 需要的 keyframesByTrack。"""
        args = dict(arguments or {})
        tracks = cls._normalize_dedup_tracks(args.get("tracks"), scope)
        if tracks:
            args["tracks"] = tracks
        grouped = cls._keyframes_by_track_from(args.get("keyframesByTrack"))
        if not grouped:
            grouped = cls._keyframes_by_track_from(args.get("frames"))
        if not grouped:
            grouped = cls._collect_keyframes_by_track(scope)
        if grouped:
            args["keyframesByTrack"] = grouped
        # dedupTracks 的函数签名不接受 frames，整理后移除别名参数，避免额外关键字导致执行失败。
        args.pop("frames", None)
        return args

    @classmethod
    def _normalize_dedup_tracks(cls, value: Any, scope: dict[str, Any]) -> list[dict[str, Any]]:
        """把 trackIds、getTrack 外壳或轨迹记录统一恢复为完整轨迹记录。"""
        available = cls._collect_tracks(scope)
        available_by_id = {
            str(item["trackId"]): item
            for item in available
            if isinstance(item, dict) and item.get("trackId") is not None
        }
        raw = value.get("tracks") if isinstance(value, dict) and isinstance(value.get("tracks"), list) else value
        if not isinstance(raw, (list, tuple)):
            return available

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                track_id = item.get("trackId")
            else:
                track_id = item
            if track_id is None:
                continue
            key = str(track_id)
            if key in seen:
                continue
            record = available_by_id.get(key)
            if record is None and isinstance(item, dict):
                record = item
            if not isinstance(record, dict) or record.get("trackId") is None:
                continue
            seen.add(key)
            normalized.append(record)
        return normalized or available

    @classmethod
    def _keyframes_by_track_from(cls, value: Any) -> dict[str, dict[str, Any]]:
        if cls._empty_dependency(value):
            return {}
        if isinstance(value, dict):
            nested = value.get("keyframesByTrack")
            if isinstance(nested, dict) and nested:
                return cls._normalize_keyframe_groups(nested)
            keyframes = value.get("keyframes")
            if isinstance(keyframes, list) and keyframes:
                return cls._group_keyframes(keyframes)
            # 已经是按轨迹分组的对象。
            if any(
                isinstance(v, (dict, list)) and (
                    isinstance(v, list) or isinstance(v.get("keyframes") if isinstance(v, dict) else None, list)
                )
                for v in value.values()
            ):
                return cls._normalize_keyframe_groups(value)
            return {}
        if isinstance(value, list):
            return cls._group_keyframes(value)
        return {}

    @staticmethod
    def _group_keyframes(frames: list[Any]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for frame in frames:
            if not isinstance(frame, dict) or frame.get("trackId") is None:
                continue
            track_id = str(frame["trackId"])
            bucket = grouped.setdefault(track_id, {"keyframes": [], "keyframeIds": []})
            bucket["keyframes"].append(frame)
            if frame.get("keyframeId") is not None:
                bucket["keyframeIds"].append(frame["keyframeId"])
        return grouped

    @classmethod
    def _normalize_keyframe_groups(cls, groups: dict[Any, Any]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for track_id, group in (groups or {}).items():
            key = str(track_id)
            if isinstance(group, dict):
                frames = group.get("keyframes") if isinstance(group.get("keyframes"), list) else []
                ids = group.get("keyframeIds") if isinstance(group.get("keyframeIds"), list) else []
            elif isinstance(group, list):
                frames = [item for item in group if isinstance(item, dict)]
                ids = [item.get("keyframeId") for item in frames if item.get("keyframeId") is not None]
            else:
                continue
            if not frames and not ids:
                continue
            normalized[key] = {"keyframes": frames, "keyframeIds": ids}
        return normalized

    @classmethod
    def _expand_to_registry_images(cls, value: Any) -> list[dict[str, Any]]:
        """把 registryReferences / registryItems / 嵌套 references 统一展成参考图记录。"""
        images: list[dict[str, Any]] = []

        def visit(node: Any) -> None:
            if isinstance(node, (list, tuple)):
                for child in node:
                    visit(child)
                return
            if not isinstance(node, dict):
                return
            # 已是参考图
            if node.get("referenceId") is not None or node.get("registryVectorId") is not None:
                if node.get("registryId") is not None or node.get("referenceId") is not None:
                    images.append(node)
                    return
            # 工具结果外壳
            if isinstance(node.get("registryReferences"), list):
                visit(node["registryReferences"])
            if isinstance(node.get("references"), list):
                visit(node["references"])
            if isinstance(node.get("registryItems"), list):
                visit(node["registryItems"])
            # 单条库项
            if node.get("registryId") is not None and (
                isinstance(node.get("references"), list) or isinstance(node.get("registryReferences"), list)
            ):
                visit(node.get("references") or node.get("registryReferences") or [])

        visit(value)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            if img.get("registryVectorId") is None and not img.get("imagePath") and not img.get("referenceId"):
                continue
            key = str(img.get("referenceId") or img.get("registryVectorId") or id(img))
            if key in seen:
                continue
            seen.add(key)
            unique.append(img)
        return unique

    # ------------------------------------------------------------------------
    # P5 scope 收集器
    # 从 working_scope 的历次工具结果中汇总出 images / registry / keyframes /
    # tracks，供 P4 的参数富化使用。全部为纯读取，不改 scope。
    # ------------------------------------------------------------------------
    @classmethod
    def _collect_registry_images(cls, scope: dict[str, Any]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for value in (scope or {}).values():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            # 优先可搜参考图
            refs = value.get("registryReferences")
            if isinstance(refs, list) and refs:
                images.extend(cls._expand_to_registry_images(refs))
            # 再展开库项嵌套 references（即使 searchable 列表为空）
            items = value.get("registryItems")
            if isinstance(items, list) and items:
                images.extend(cls._expand_to_registry_images(items))
            one = value.get("registryItem")
            if isinstance(one, dict):
                images.extend(cls._expand_to_registry_images(one))
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for img in images:
            key = str(img.get("referenceId") or img.get("registryVectorId") or id(img))
            if key in seen:
                continue
            seen.add(key)
            unique.append(img)
        return unique

    @staticmethod
    def _collect_registry_items(scope: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in (scope or {}).values():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            candidates = value.get("registryItems")
            if not isinstance(candidates, list):
                continue
            for item in candidates:
                if not isinstance(item, dict) or item.get("registryId") is None:
                    continue
                key = str(item["registryId"])
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
        return items

    @classmethod
    def _collect_keyframes_by_track(cls, scope: dict[str, Any]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for value in (scope or {}).values():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            grouped = cls._keyframes_by_track_from(value)
            for track_id, group in grouped.items():
                bucket = merged.setdefault(track_id, {"keyframes": [], "keyframeIds": []})
                bucket["keyframes"].extend(group.get("keyframes") or [])
                bucket["keyframeIds"].extend(group.get("keyframeIds") or [])
        for bucket in merged.values():
            seen_ids: set[str] = set()
            unique_frames: list[dict[str, Any]] = []
            for frame in bucket.get("keyframes") or []:
                if not isinstance(frame, dict):
                    continue
                key = str(frame.get("keyframeId") or id(frame))
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                unique_frames.append(frame)
            bucket["keyframes"] = unique_frames
            bucket["keyframeIds"] = list(dict.fromkeys(x for x in bucket.get("keyframeIds") or [] if x is not None))
        return merged

    @classmethod
    def _collect_tracks(cls, scope: dict[str, Any]) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in (scope or {}).values():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            for track in value.get("tracks") or []:
                if not isinstance(track, dict) or track.get("trackId") is None:
                    continue
                key = str(track["trackId"])
                if key in seen:
                    continue
                seen.add(key)
                tracks.append(track)
        return tracks

    @classmethod
    def _collect_keyframes(cls, scope: dict[str, Any]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for value in (scope or {}).values():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            kfs = value.get("keyframes")
            if isinstance(kfs, list) and kfs:
                frames.extend([f for f in kfs if isinstance(f, dict)])
        return frames

    # ------------------------------------------------------------------------
    # P6 事件与 $ref 解析
    # _emit                 向 observe_node 回调派发工具事件（异常不外抛）
    # resolve_references    $ref / $slice / $default / $map 的求值入口
    # _resolve / _condition / _read / _read_with_presence  具体求值实现
    # 注意：_dependency_issue 要求根步骤非 ok=False，而 _read 直接取值不做失败
    #       判定 —— 两者对「依赖失败」的处理并不对称。
    # ------------------------------------------------------------------------
    @staticmethod
    def _emit(callback: Callable[[dict[str, Any]], None] | None, phase: str, observation: dict[str, Any]) -> None:
        if not callback:
            return
        try:
            payload = {
                "phase": phase,
                "id": observation.get("id"),
                "tool": observation.get("tool"),
            }
            if isinstance(observation.get("arguments"), dict):
                payload["arguments"] = observation["arguments"]
            if observation.get("skipped"):
                payload["skipped"] = True
                payload["error"] = observation.get("skipReason")
            elif phase in {"completed", "failed"}:
                payload["summary"] = PlanExecutor.summarize_observation(observation)
                result = observation.get("result") or {}
                payload["ok"] = result.get("ok") is not False
                if result.get("error"):
                    payload["error"] = result["error"]
            callback(payload)
        except Exception:
            pass

    @staticmethod
    def _read_with_presence(value: Any, path: str) -> tuple[bool, Any]:
        current = value
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return False, None
        return True, current

    @classmethod
    def resolve_references(cls, value: Any, scope: dict[str, Any]) -> Any:
        """按工作域解析结构化引用，供执行与跨轮次语义去重共用。"""
        if isinstance(value, list):
            return [cls.resolve_references(item, scope) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" not in value:
            return {key: cls.resolve_references(item, scope) for key, item in value.items()}
        resolved = cls._read(scope, value["$ref"])
        if cls._empty_dependency(resolved) and "$default" in value:
            default_value = value["$default"]
            if isinstance(default_value, dict) and "$ref" in default_value:
                resolved = cls.resolve_references(default_value, scope)
            else:
                resolved = default_value
        if value.get("$map") and isinstance(resolved, list):
            resolved = [cls._read(item, value["$map"]) for item in resolved]
        if value.get("$slice") and isinstance(resolved, list):
            # focused 证据模式：限制下游（如 getFrames）消费的候选数量
            try:
                resolved = resolved[: max(0, int(value["$slice"]))]
            except (TypeError, ValueError):
                pass
        if value.get("$compact") and isinstance(resolved, list):
            resolved = [item for item in resolved if item not in (None, "", [])]
        if value.get("$list"):
            resolved = [] if resolved is None else resolved if isinstance(resolved, list) else [resolved]
        return value.get("$default") if resolved is None and "$default" in value else resolved

    def _resolve(self, value: Any, scope: dict[str, Any]) -> Any:
        return self.resolve_references(value, scope)

    def _condition(self, condition: dict[str, Any] | None, scope: dict[str, Any]) -> bool:
        if not condition:
            return True
        if not isinstance(condition, dict) or "ref" not in condition:
            return True
        current = self._read(scope, condition["ref"])
        if "equals" in condition:
            return current == condition["equals"]
        if "in" in condition:
            return current in condition["in"]
        return bool(current)

    @staticmethod
    def _read(value: Any, path: str) -> Any:
        present, current = PlanExecutor._read_with_presence(value, path)
        return current if present else None

    # ------------------------------------------------------------------------
    # P7 摘要压缩
    # _compact_args / summarize_observation 把完整工具结果压成回传模型的摘要，
    # 避免关键帧大 JSON 进入对话上下文（完整结果仍在 working_scope 里）。
    # ------------------------------------------------------------------------
    @staticmethod
    def _compact_args(arguments: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for key, value in (arguments or {}).items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                labels = []
                for item in value:
                    if not isinstance(item, dict):
                        labels.append(str(item))
                        continue
                    label = (
                        item.get("keyframeId")
                        or item.get("referenceId")
                        or item.get("registryId")
                        or item.get("trackId")
                        or item.get("hullNumber")
                    )
                    labels.append(label if label is not None else f"<dict:{len(item)}>")
                compact[key] = labels
            elif isinstance(value, dict) and len(str(value)) > 500:
                compact[key] = f"<object:{len(value)} keys>"
            else:
                compact[key] = value
        return compact

    @classmethod
    def summarize_observation(cls, observation: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "id": observation.get("id"),
            "tool": observation.get("tool"),
            "skipped": bool(observation.get("skipped")),
        }
        if summary["skipped"]:
            summary["skipReason"] = observation.get("skipReason")
            result = observation.get("result") or {}
            if result.get("error"):
                summary["error"] = result["error"]
            return summary
        result = observation.get("result") or {}
        summary["ok"] = result.get("ok") is not False
        if result.get("error"):
            summary["error"] = result["error"]
        tracks = result.get("tracks")
        if isinstance(tracks, list):
            summary["trackCount"] = len(tracks)
            summary["trackIds"] = [
                item.get("trackId") for item in tracks[:20] if isinstance(item, dict) and item.get("trackId") is not None
            ]
        keyframes = result.get("keyframes")
        if isinstance(keyframes, list):
            summary["keyframeCount"] = len(keyframes)
            summary["keyframeIds"] = [
                item.get("keyframeId")
                for item in keyframes[:20]
                if isinstance(item, dict) and item.get("keyframeId") is not None
            ]
        matches = result.get("matches")
        if isinstance(matches, list):
            summary["matchCount"] = len(matches)
        # 库项数与参考图数分开统计，避免 searchable 参考图为 0 时把库项数盖成 0
        if isinstance(result.get("registryItems"), list):
            summary["registryCount"] = len(result["registryItems"])
            summary["registryItemCount"] = len(result["registryItems"])
        if isinstance(result.get("registryReferences"), list):
            summary["registryReferenceCount"] = len(result["registryReferences"])
            if summary.get("registryCount") is None:
                summary["registryCount"] = len(result["registryReferences"])
        if isinstance(result.get("exactMatches"), dict):
            summary["exactMatchHullCount"] = len(result["exactMatches"])
        for key in (
            "found", "decision", "matchMode", "totalTrackCount", "returnedTrackCount",
            "highThresholdShipCount", "lowThresholdShipCount", "hasMore", "offset", "limit",
            "matchedHullNumbers", "shipSegmentId",
        ):
            if result.get(key) not in (None, "", []):
                summary[key] = result.get(key)
        return summary

    # ------------------------------------------------------------------------
    # P8 调用去重与语义签名
    # semantic_signature  归一化参数后生成签名：getTrack 补默认 offset、
    #                     matchImage 忽略 topK、集合类参数排序消除顺序差异
    # sanitize_calls      计划内去重：签名相同则丢弃后者并把 $ref 改写到保留项，
    #                     重名 id 加 -2/-3 后缀，最后统一重写前向引用
    # 注意：去重只影响「是否执行」，不改变实际执行参数。
    # ------------------------------------------------------------------------
    @staticmethod
    def _rewrite_call_refs(value: Any, aliases: dict[str, str]) -> Any:
        """把被去重步骤的引用改写到实际保留的步骤，避免后续依赖失效。"""
        if isinstance(value, list):
            return [PlanExecutor._rewrite_call_refs(item, aliases) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"$ref", "ref"} and isinstance(item, str):
                head, separator, tail = item.partition(".")
                target = aliases.get(head, head)
                rewritten[key] = f"{target}{separator}{tail}" if separator else target
            else:
                rewritten[key] = PlanExecutor._rewrite_call_refs(item, aliases)
        return rewritten

    @staticmethod
    def semantic_signature(tool: str, arguments: dict[str, Any]) -> str:
        """生成工具调用的语义签名；只影响去重，不改实际执行参数。"""
        normalized = dict(arguments)
        if tool == "getTrack":
            # 后端缺省 offset 即 0，显式与隐式写法应视为同一次查询。
            normalized.setdefault("offset", 0)
        elif tool == "getRegistry":
            normalized["hullNumber"] = str(normalized.get("hullNumber") or "").strip()
        elif tool == "matchImage":
            # 同一批图像不能仅因 topK 不同就重复完成昂贵的向量匹配。
            normalized.pop("topK", None)

        unordered_keys = {
            "trackIds", "queryImages", "galleryImages", "registryItems", "hullNumberArray",
        }

        def canonical(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {
                    str(child_key): canonical(child_value, str(child_key))
                    for child_key, child_value in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            if isinstance(value, list):
                items = [canonical(item) for item in value]
                if key in unordered_keys:
                    return sorted(
                        items,
                        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
                    )
                return items
            return value

        payload = canonical(normalized)
        return f"{tool}:{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"

    @staticmethod
    def sanitize_calls(calls: Any, *, max_calls: int = 8) -> list[dict[str, Any]]:
        if not isinstance(calls, list):
            return []
        cleaned: list[dict[str, Any]] = []
        used: set[str] = set()
        aliases: dict[str, str] = {}
        signatures: dict[str, int] = {}
        for index, item in enumerate(calls):
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if not tool:
                continue
            requested_id = str(item.get("id") or f"step-{index + 1}").strip() or f"step-{index + 1}"
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            arguments = PlanExecutor._rewrite_call_refs(arguments, aliases)
            condition = item.get("condition") if isinstance(item.get("condition"), dict) else None
            condition = PlanExecutor._rewrite_call_refs(condition, aliases) if condition else None
            signature = PlanExecutor.semantic_signature(tool, arguments)
            if signature in signatures:
                kept_index = signatures[signature]
                aliases[requested_id] = str(cleaned[kept_index]["id"])
                # 后出现的重复调用若带有依赖条件，则补到保留项上，避免空依赖仍被执行。
                if condition and "condition" not in cleaned[kept_index]:
                    cleaned[kept_index]["condition"] = condition
                continue
            if len(cleaned) >= max_calls:
                break

            call_id = requested_id
            base = call_id
            n = 1
            while call_id in used:
                n += 1
                call_id = f"{base}-{n}"
            used.add(call_id)
            aliases[requested_id] = call_id
            entry: dict[str, Any] = {"id": call_id, "tool": tool, "arguments": arguments}
            if condition:
                entry["condition"] = condition
            signatures[signature] = len(cleaned)
            cleaned.append(entry)

        # 再处理一次前向引用或后续才发现的别名。
        for entry in cleaned:
            entry["arguments"] = PlanExecutor._rewrite_call_refs(entry.get("arguments") or {}, aliases)
            if isinstance(entry.get("condition"), dict):
                entry["condition"] = PlanExecutor._rewrite_call_refs(entry["condition"], aliases)
        return cleaned
