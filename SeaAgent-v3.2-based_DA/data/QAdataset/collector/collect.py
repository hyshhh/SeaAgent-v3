"""Generate a deterministic SeaQA subset from server-side memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from tools.service import ToolService

TZ = timezone(timedelta(hours=8))
EMPTY_HULLS = {"", "NULL", "NONE", "UNKNOWN"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, default=str) for row in rows
    )
    path.write_text(text + "\n", encoding="utf-8")


def normalize_hull(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in EMPTY_HULLS:
        return ""
    return "".join(character for character in text if character.isalnum() or "\u4e00" <= character <= "\u9fff")


def format_time(timestamp: float, *, with_date: bool = True) -> str:
    value = datetime.fromtimestamp(float(timestamp), TZ)
    return value.isoformat(timespec="seconds") if with_date else value.strftime("%H:%M:%S")


def short_description(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    first = next((part.strip() for part in text.split("；") if part.strip()), text)
    if len(first) > limit:
        first = first[: limit - 1].rstrip(" ,.;:") + "…"
    return first


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Collector:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        project_root = (self.root / self.config["paths"]["project_root"]).resolve()
        self.project_root = project_root
        self.memory_dir = project_root / self.config["paths"]["memory_dir"]
        self.registry_dir = project_root / self.config["paths"]["registry_dir"]
        self.output_dir = (self.root / self.config["paths"]["output_dir"]).resolve()
        if self.root not in self.output_dir.parents:
            raise ValueError("output_dir must stay inside data/QAdataset")
        self.features_dir = self.output_dir / "features"
        self.images_dir = self.output_dir / "images"
        self.rng = random.Random(int(self.config["seed"]))
        self.service = ToolService(load_config())

        self.tracks = read_csv(self.memory_dir / "tracks.csv")
        self.keyframes = read_csv(self.memory_dir / "track_keyframes.csv")
        self.registry = read_csv(self.registry_dir / "registry.csv")
        self.references = read_csv(self.registry_dir / "registry_reference_images.csv")
        self.track_by_id = {str(item["track_id"]): item for item in self.tracks}
        self.keyframes_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
        for frame in self.keyframes:
            self.keyframes_by_track[str(frame["track_id"])].append(frame)
        for frames in self.keyframes_by_track.values():
            frames.sort(key=lambda item: float(item.get("timestamp") or 0))
        self.registry_by_hull = {
            normalize_hull(item.get("hull_number")): item for item in self.registry
        }
        self.registry_ref_counts = Counter(
            str(item["registry_id"]) for item in self.references
        )
        self.questions: list[dict[str, Any]] = []
        self.entity_by_track: dict[str, str] = {}
        self.entity_groups: dict[str, list[str]] = {}
        self.fuzzy_groups: list[list[str]] = []
        self.registry_cache: dict[str, dict[str, Any]] = {}
        self.dedup_cache: dict[str, dict[str, Any]] = {}

    def run(self) -> None:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.load_registry_verification_cache()
        self.build_entity_groups()
        self.add_hull_questions()
        self.add_description_questions()
        self.add_image_questions()
        self.add_total_count_questions()
        self.add_registered_count_questions()
        self.add_unregistered_count_questions()
        self.validate_questions()
        self.write_outputs()

    def load_registry_verification_cache(self) -> None:
        path = self.features_dir / "registry_verifications.json"
        if path.is_file():
            for row in json.loads(path.read_text(encoding="utf-8")):
                key = f"{row.get('hullNumber')}|{row.get('trackId')}|{row.get('matchedRegistryReferenceId')}"
                self.registry_cache[key] = row

    def verify_registry_candidates(self) -> None:
        if not self.config["verification"].get("registry_use_vlm", True):
            return
        for item in self.registry:
            registry_id = str(item["registry_id"])
            if not self.registry_ref_counts[registry_id]:
                continue
            hull = str(item["hull_number"])
            matches = self.service.match_registry_tracks(
                hull, top_k=int(self.config["collection"]["registry_top_k"])
            ).get("matches", [])
            confirmed = 0
            for match in matches:
                reference_id = str(match.get("matchedRegistryReferenceId") or "")
                keyframe_id = str(match.get("matchedKeyframeId") or "")
                if not reference_id or not keyframe_id:
                    continue
                key = f"{hull}|{match.get('trackId')}|{reference_id}"
                if key not in self.registry_cache:
                    verification = self.service.verify_vessel_images(
                        reference_registry_reference_id=reference_id,
                        candidate_keyframe_id=keyframe_id,
                    )
                    self.registry_cache[key] = {
                        "hullNumber": hull,
                        "registryId": registry_id,
                        "trackId": str(match.get("trackId")),
                        "embeddingScore": match.get("score"),
                        "matchedRegistryReferenceId": reference_id,
                        "matchedKeyframeId": keyframe_id,
                        "startTimeText": match.get("startTimeText"),
                        "endTimeText": match.get("endTimeText"),
                        "ok": verification.get("ok"),
                        "decision": verification.get("decision"),
                        "facts": verification.get("facts", []),
                    }
                if self.registry_cache[key].get("decision") == "match":
                    confirmed += 1
                if confirmed >= int(self.config["collection"]["registry_matches_per_hull"]):
                    break
        write_json(
            self.features_dir / "registry_verifications.json",
            list(self.registry_cache.values()),
        )

    def build_entity_groups(self) -> None:
        self.verify_registry_candidates()
        tracks, _ = self.service._scan_tracks(None)
        frames, _, _ = self.service._load_search_frames(tracks)
        selected: dict[str, list[dict[str, Any]]] = {}
        for track_id, items in frames.items():
            selected[str(track_id)] = sorted(
                items,
                key=lambda item: float(item.get("retentionScore") or 0),
                reverse=True,
            )[:3]
        all_frames = [frame for values in selected.values() for frame in values]
        vectors = self.service._load_indexed_vectors(
            all_frames,
            vector_key="keyframeVectorId",
            index=self.service.vectors.keyframes,
        )
        track_map = {str(item["trackId"]): item for item in tracks}
        track_ids = list(track_map)
        pair_scores: dict[tuple[str, str], float] = {}
        pair_details: dict[tuple[str, str], dict[str, Any]] = {}
        for index, left_id in enumerate(track_ids):
            for right_id in track_ids[index + 1 :]:
                if (
                    self.service._overlap(track_map[left_id], track_map[right_id])
                    or not selected[left_id]
                    or not selected[right_id]
                ):
                    continue
                scored = []
                for left in selected[left_id]:
                    for right in selected[right_id]:
                        left_vector = vectors.get(int(left["keyframeVectorId"]))
                        right_vector = vectors.get(int(right["keyframeVectorId"]))
                        if left_vector is not None and right_vector is not None:
                            scored.append(
                                (float(left_vector @ right_vector), left, right)
                            )
                if not scored:
                    continue
                key = tuple(sorted((left_id, right_id)))
                top_scores = sorted(
                    (score for score, _, _ in scored), reverse=True
                )[:3]
                pair_scores[key] = sum(top_scores) / len(top_scores)
                best_score, left_frame, right_frame = max(scored, key=lambda item: item[0])
                pair_details[key] = {
                    "pairId": f"{key[0]}::{key[1]}",
                    "referenceTrackId": left_id,
                    "candidateTrackId": right_id,
                    "score": round(pair_scores[key], 6),
                    "bestFrameScore": round(best_score, 6),
                    "referenceKeyframeId": str(left_frame.get("keyframeId") or ""),
                    "candidateKeyframeId": str(right_frame.get("keyframeId") or ""),
                }
        high = float(self.service.settings["dedup_high"])
        review = float(self.service.settings["dedup_review"])
        low = float(self.service.settings["dedup_low"])
        review_pairs = [
            detail
            for key, detail in pair_details.items()
            if review <= pair_scores[key] < high
        ]
        if self.config["verification"].get("dedup_use_vlm", True):
            for detail in review_pairs:
                pair_id = detail["pairId"]
                if pair_id not in self.dedup_cache:
                    result = self.service.verify_vessel_images(
                        reference_keyframe_id=detail["referenceKeyframeId"],
                        candidate_keyframe_id=detail["candidateKeyframeId"],
                        pair_id=pair_id,
                    )
                    self.dedup_cache[pair_id] = {
                        **detail,
                        "ok": result.get("ok"),
                        "decision": result.get("decision"),
                        "facts": result.get("facts", []),
                    }
        confirmed = {
            pair_id for pair_id, result in self.dedup_cache.items()
            if result.get("decision") == "match"
        }
        exact_scores = {
            key: score for key, score in pair_scores.items() if score >= high
        }
        exact_scores.update({
            key: score
            for key, score in pair_scores.items()
            if pair_details[key]["pairId"] in confirmed
        })
        exact_groups = self.service._complete_link(track_ids, track_map, exact_scores, review)
        fuzzy_groups = self.service._complete_link(track_ids, track_map, pair_scores, low)
        self.fuzzy_groups = fuzzy_groups
        for group in exact_groups:
            entity_id = f"entity-{min(int(value) if value.isdigit() else 10**9 for value in group):06d}"
            self.entity_groups[entity_id] = sorted(group, key=lambda value: int(value) if value.isdigit() else value)
            for track_id in group:
                self.entity_by_track[track_id] = entity_id
        write_json(
            self.features_dir / "dedup_verifications.json",
            list(self.dedup_cache.values()),
        )
        write_json(
            self.features_dir / "entity_groups.json",
            {
                "seed": self.config["seed"],
                "high": high,
                "review": review,
                "low": low,
                "exact_entities": self.entity_groups,
                "fuzzy_groups": fuzzy_groups,
            },
        )

    def registry_matches(self) -> list[dict[str, Any]]:
        return [
            row for row in self.registry_cache.values()
            if row.get("decision") == "match" and row.get("trackId") in self.track_by_id
        ]

    def entity_for_track(self, track_id: str) -> dict[str, Any]:
        entity_id = self.entity_by_track[track_id]
        group = self.entity_groups[entity_id]
        source = self.track_by_id[track_id]
        evidence = [
            frame["keyframe_id"]
            for group_track in group
            for frame in self.keyframes_by_track.get(group_track, [])[:1]
        ][: int(self.config["collection"]["max_evidence_tracks"])]
        return {
            "entity_id": entity_id,
            "track_ids": group,
            "evidence_keyframe_ids": evidence,
            "hull_numbers": sorted({
                normalize_hull(self.track_by_id[value].get("final_hull_number"))
                for value in group
                if normalize_hull(self.track_by_id[value].get("final_hull_number"))
            }),
            "source_track_id": track_id,
            "source_keyframe_id": evidence[0] if evidence else None,
            "start_time": float(source["start_time"]),
            "end_time": float(source["end_time"]),
        }

    def add_question(
        self,
        category: str,
        question: str,
        inputs: dict[str, Any],
        ground_truth: dict[str, Any],
        provenance: dict[str, Any],
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        self.questions.append({
            "id": f"SeaQA-{len(self.questions) + 1:04d}",
            "category": category,
            "question": question,
            "time_range": (
                {"start": format_time(start_time), "end": format_time(end_time)}
                if start_time is not None and end_time is not None
                else None
            ),
            "inputs": inputs,
            "ground_truth": ground_truth,
            "provenance": provenance,
        })

    def track_window(self, track_id: str) -> tuple[float, float]:
        track = self.track_by_id[track_id]
        padding = float(self.config["collection"]["time_padding_seconds"])
        return float(track["start_time"]) - padding, float(track["end_time"]) + padding

    def add_hull_questions(self) -> None:
        matches = sorted(
            self.registry_matches(),
            key=lambda item: (str(item.get("hullNumber")), str(item.get("startTimeText"))),
        )
        selected = matches[:12]
        for index, row in enumerate(selected, 1):
            track_id = str(row["trackId"])
            start, end = self.track_window(track_id)
            entity = self.entity_for_track(track_id)
            hull = str(row["hullNumber"])
            self.add_question(
                "hull_number_lookup",
                f"在 {format_time(start)} 至 {format_time(end)} 内，舷号 {hull} 的船是否出现过？",
                {"hull_number": hull},
                {
                    "answerable": True,
                    "answer": "appears",
                    "entities": [{**entity, "registry_id": row.get("registryId"), "hull_number": hull}],
                    "unresolved_entities": [],
                    "method": "registry_embedding_plus_vlm",
                },
                {
                    "source": "server trajectory and registry memory",
                    "source_track_id": track_id,
                    "source_keyframe_id": row.get("matchedKeyframeId"),
                    "registry_reference_id": row.get("matchedRegistryReferenceId"),
                    "generated_question": index,
                },
                start,
                end,
            )
        unresolved_hulls = ["0014", "0123", "0789", "A01", "0857", "320", "无极"] * 2
        anchors = self.tracks[10::18]
        for hull, track in zip(unresolved_hulls[:11], anchors[:11]):
            start, end = self.track_window(str(track["track_id"]))
            registry = self.registry_by_hull.get(normalize_hull(hull), {})
            reason = (
                "registry_entry_has_no_reference_image"
                if not self.registry_ref_counts.get(str(registry.get("registry_id")), 0)
                else "no_exhaustive_vlm_confirmation_for_the_requested_window"
            )
            self.add_question(
                "hull_number_lookup",
                f"在 {format_time(start)} 至 {format_time(end)} 内，舷号 {hull} 的船是否出现过？",
                {"hull_number": hull},
                {
                    "answerable": False,
                    "answer": "unresolved",
                    "entities": [],
                    "unresolved_entities": [{
                        "entity_id": f"registry:{registry.get('registry_id')}",
                        "registry_id": registry.get("registry_id"),
                        "hull_number": hull,
                        "reason": reason,
                    }],
                    "method": "registry_evidence_insufficient",
                },
                {
                    "source": "server registry and trajectory memory",
                    "source_track_id": str(track["track_id"]),
                    "registry_id": registry.get("registry_id"),
                },
                start,
                end,
            )

    def select_spread_tracks(self, count: int, *, require_hull: bool = False) -> list[dict[str, str]]:
        candidates = [
            track for track in self.tracks
            if track.get("final_description")
            and (not require_hull or normalize_hull(track.get("final_hull_number")))
            and self.keyframes_by_track.get(str(track["track_id"]))
        ]
        candidates.sort(key=lambda item: float(item["start_time"]))
        if len(candidates) <= count:
            return candidates
        step = len(candidates) / count
        return [candidates[min(len(candidates) - 1, int(index * step))] for index in range(count)]

    def add_description_questions(self) -> None:
        tracks = self.select_spread_tracks(int(self.config["distribution"]["description_lookup"]), require_hull=True)
        for track in tracks:
            track_id = str(track["track_id"])
            start, end = self.track_window(track_id)
            description = short_description(
                track.get("final_description"),
                int(self.config["collection"]["max_description_chars"]),
            )
            entity = self.entity_for_track(track_id)
            self.add_question(
                "description_lookup",
                f"在 {format_time(start)} 至 {format_time(end)} 内，是否出现过以下外观的船：{description}？",
                {"description": description},
                {
                    "answerable": True,
                    "answer": "appears",
                    "entities": [entity],
                    "unresolved_entities": [],
                    "method": "annotated_track_description",
                },
                {
                    "source": "server track description annotation",
                    "source_track_id": track_id,
                    "source_keyframe_id": entity.get("source_keyframe_id"),
                },
                start,
                end,
            )

    def add_image_questions(self) -> None:
        tracks = self.select_spread_tracks(int(self.config["distribution"]["image_lookup"]))
        for index, track in enumerate(tracks, 1):
            track_id = str(track["track_id"])
            frames = self.keyframes_by_track[track_id]
            frame = frames[len(frames) // 2]
            source = Path(frame["keyframe_path"])
            if not source.is_file():
                continue
            suffix = source.suffix.lower() or ".jpg"
            image_name = f"query-{index:03d}{suffix}"
            destination = self.images_dir / image_name
            shutil.copy2(source, destination)
            start, end = self.track_window(track_id)
            entity = self.entity_for_track(track_id)
            self.add_question(
                "image_lookup",
                f"在 {format_time(start)} 至 {format_time(end)} 内，是否出现过与上传图片相同的船？",
                {"image_path": f"images/{image_name}"},
                {
                    "answerable": True,
                    "answer": "appears",
                    "entities": [entity],
                    "unresolved_entities": [],
                    "method": "source_track_keyframe",
                },
                {
                    "source": "server trajectory keyframe",
                    "source_track_id": track_id,
                    "source_keyframe_id": frame["keyframe_id"],
                    "source_original_path": str(source),
                    "image_sha256": file_sha256(destination),
                },
                start,
                end,
            )

    def groups_in_window(self, start: float, end: float) -> list[str]:
        return sorted({
            self.entity_by_track[str(track["track_id"])]
            for track in self.tracks
            if float(track["end_time"]) >= start and float(track["start_time"]) <= end
        })

    def add_total_count_questions(self) -> None:
        count = int(self.config["distribution"]["total_count"])
        window = float(self.config["collection"]["count_window_seconds"])
        first = min(float(item["start_time"]) for item in self.tracks)
        last = max(float(item["end_time"]) for item in self.tracks)
        for index in range(count):
            center = first + (last - first) * (index + 0.5) / count
            start, end = center - window / 2, center + window / 2
            entities = [self.entity_for_track(group[0]) for group in (
                self.entity_groups[entity_id] for entity_id in self.groups_in_window(start, end)
            )]
            self.add_question(
                "total_count",
                f"在 {format_time(start)} 至 {format_time(end)} 内，一共出现了多少艘不同船？",
                {},
                {
                    "answerable": True,
                    "answer": len(entities),
                    "count": len(entities),
                    "entities": entities,
                    "unresolved_entities": [],
                    "method": "exact_entity_groups",
                },
                {
                    "source": "server trajectory memory",
                    "entity_group_count": len(entities),
                    "seed": self.config["seed"],
                },
                start,
                end,
            )

    def add_registered_count_questions(self) -> None:
        matches = sorted(
            self.registry_matches(),
            key=lambda item: float(self.track_by_id[str(item["trackId"])]["start_time"]),
        )
        count = int(self.config["distribution"]["registered_count"])
        for index in range(count):
            row = matches[index % len(matches)]
            track_id = str(row["trackId"])
            start, end = self.track_window(track_id)
            entity = self.entity_for_track(track_id)
            entity.update({"registry_id": row.get("registryId"), "hull_number": row.get("hullNumber")})
            self.add_question(
                "registered_count",
                f"在 {format_time(start)} 至 {format_time(end)} 内，已确认在库的船实体有多少艘？",
                {},
                {
                    "answerable": True,
                    "answer": 1,
                    "count": 1,
                    "entities": [entity],
                    "unresolved_entities": [],
                    "method": "registry_embedding_plus_vlm",
                },
                {
                    "source": "server registry and trajectory memory",
                    "source_track_id": track_id,
                    "registry_reference_id": row.get("matchedRegistryReferenceId"),
                },
                start,
                end,
            )

    def add_unregistered_count_questions(self) -> None:
        tracks = self.select_spread_tracks(
            int(self.config["distribution"]["unregistered_count"]),
            require_hull=True,
        )
        registry_hulls = set(self.registry_by_hull)
        selected = [
            track for track in tracks
            if normalize_hull(track.get("final_hull_number")) not in registry_hulls
        ]
        for track in selected:
            track_id = str(track["track_id"])
            start, end = self.track_window(track_id)
            entity = self.entity_for_track(track_id)
            hull = normalize_hull(track.get("final_hull_number"))
            self.add_question(
                "unregistered_count",
                f"在 {format_time(start)} 至 {format_time(end)} 内，已确认不在先验库的船实体有多少艘？",
                {},
                {
                    "answerable": True,
                    "answer": 1,
                    "count": 1,
                    "entities": [{**entity, "hull_number": hull, "registry_id": None}],
                    "unresolved_entities": [],
                    "method": "readable_hull_not_in_registry",
                },
                {
                    "source": "server track hull annotation and registry",
                    "source_track_id": track_id,
                    "hull_number": hull,
                },
                start,
                end,
            )

    def validate_questions(self) -> None:
        expected = sum(int(value) for value in self.config["distribution"].values())
        if len(self.questions) != expected:
            raise RuntimeError(f"generated {len(self.questions)} questions, expected {expected}")
        distribution = Counter(item["category"] for item in self.questions)
        if distribution != Counter({key: int(value) for key, value in self.config["distribution"].items()}):
            raise RuntimeError(f"category distribution mismatch: {distribution}")
        ids = [item["id"] for item in self.questions]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate question IDs")
        for item in self.questions:
            if not item["question"].strip():
                raise RuntimeError(f"empty question: {item['id']}")
            if not item["ground_truth"]:
                raise RuntimeError(f"missing truth: {item['id']}")

    def write_outputs(self) -> None:
        questions_path = self.output_dir / "questions.jsonl"
        write_jsonl(questions_path, self.questions)
        category_counts = Counter(item["category"] for item in self.questions)
        source_hashes = {
            path.name: file_sha256(path)
            for path in (
                self.memory_dir / "tracks.csv",
                self.memory_dir / "track_keyframes.csv",
                self.registry_dir / "registry.csv",
                self.registry_dir / "registry_reference_images.csv",
            )
            if path.is_file()
        }
        manifest = {
            "dataset": "SeaQA-100",
            "version": int(self.config["version"]),
            "seed": int(self.config["seed"]),
            "question_count": len(self.questions),
            "category_counts": dict(category_counts),
            "source_counts": {
                "tracks": len(self.tracks),
                "keyframes": len(self.keyframes),
                "registry_entries": len(self.registry),
                "registry_references": len(self.references),
                "exact_entities": len(self.entity_groups),
                "fuzzy_entities": len(self.fuzzy_groups),
            },
            "source_hashes": source_hashes,
            "question_file_sha256": file_sha256(questions_path),
            "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        }
        write_json(self.output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a SeaQA subset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    args = parser.parse_args()
    Collector(args.config).run()


if __name__ == "__main__":
    main()
