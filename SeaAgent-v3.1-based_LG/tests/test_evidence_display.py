import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from agent.controller import AgentController
from tools.service import ToolService


class _RegistryRepository:
    def references_by_ids(self, reference_ids):
        records = {
            "ref-003-a": {"referenceId": "ref-003-a", "registryId": "registry-003-a"},
            "ref-003-b": {"referenceId": "ref-003-b", "registryId": "registry-003-b"},
            "ref-004": {"referenceId": "ref-004", "registryId": "registry-004"},
        }
        return [records[value] for value in reference_ids if value in records]

    def list_registry(self):
        return [
            {"registryId": "registry-003-a", "hullNumber": "003"},
            {"registryId": "registry-003-b", "hullNumber": "003"},
            {"registryId": "registry-004", "hullNumber": "004"},
        ]


class _DisplayTools:
    def getFrames(self, track_ids):
        return {
            "keyframesByTrack": {
                str(track_id): {
                    "keyframes": [{"keyframeId": f"frame-{track_id}", "retentionScore": 1.0}]
                }
                for track_id in track_ids
            }
        }

    def _representative_registry_reference_ids(self, reference_ids):
        return list(dict.fromkeys(reference_ids))


class EvidenceDisplayTest(unittest.TestCase):
    def test_crop_is_scaled_to_fixed_canvas(self):
        crop = np.full((40, 80, 3), 255, dtype=np.uint8)
        canvas = ToolService._fit_crop_to_canvas(crop, (640, 360))
        self.assertEqual(canvas.shape, (360, 640, 3))
        self.assertGreater(np.count_nonzero(canvas == 255), crop.size)

    def test_image_input_flattens_registry_and_keyframe_groups(self):
        records = ToolService._flatten_image_records([
            {"references": [{"referenceId": "ref-1"}]},
            {"keyframesByTrack": {"1": {"keyframes": [{"keyframeId": "frame-1"}]}}},
        ])

        self.assertEqual(records, [{"referenceId": "ref-1"}, {"keyframeId": "frame-1"}])

    def test_registry_evidence_keeps_one_image_per_hull(self):
        service = ToolService.__new__(ToolService)
        service.repository = _RegistryRepository()
        selected = service._representative_registry_reference_ids(["ref-003-a", "ref-003-b", "ref-004"])
        self.assertEqual(selected, ["ref-003-a", "ref-004"])

    def test_all_tracks_receive_lazy_evidence_groups(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []
        tracks = [{"trackId": str(index)} for index in range(1, 99)]
        controller._display_tracks(tracks, include_clips=True, include_registry=False)
        self.assertEqual(len(controller.display_groups), len(tracks))
        self.assertEqual([item["clipTrackId"] for item in controller.display_groups], [str(index) for index in range(1, 99)])
        self.assertTrue(all(len(item["keyframeIds"]) == 1 for item in controller.display_groups))

    def test_evidence_tracks_are_sorted_by_similarity(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []
        tracks = [
            {"trackId": "1", "embeddingScore": 0.42},
            {"trackId": "2", "embeddingScore": 0.91},
            {"trackId": "3", "embeddingScore": 0.73},
        ]

        controller._display_tracks(tracks, include_clips=False, include_registry=False)

        self.assertEqual([item["trackId"] for item in controller.display_groups], ["2", "3", "1"])
        self.assertEqual([item["embeddingScore"] for item in controller.display_groups], [0.91, 0.73, 0.42])

    def test_registry_out_state_is_forwarded_to_evidence_groups(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []

        controller._display_tracks([
            {"trackId": "17", "embeddingScore": 0.41, "scoreBand": "mismatch", "registryOutState": "confirmed_out"},
            {"trackId": "19", "embeddingScore": 0.63, "scoreBand": "uncertain", "registryOutState": "gray_zone"},
        ], include_clips=True, include_registry=False)

        by_track = {item["trackId"]: item for item in controller.display_groups}
        self.assertEqual(by_track["17"]["registryOutState"], "confirmed_out")
        self.assertEqual(by_track["19"]["registryOutState"], "gray_zone")
        self.assertEqual(by_track["17"]["scoreBand"], "mismatch")
        self.assertEqual(by_track["19"]["scoreBand"], "uncertain")

    def test_track_clip_is_used_even_when_a_cached_segment_exists(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []

        controller._display_tracks([{"trackId": "1", "shipSegmentIds": ["segment-1"]}], include_clips=True, include_registry=False)

        self.assertEqual(controller.display_groups[0]["clipTrackId"], "1")

    def test_image_preview_uses_requested_compression_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            preview_dir = root / "clips"
            self.assertTrue(cv2.imwrite(str(source), np.full((100, 200, 3), 255, dtype=np.uint8)))
            service = ToolService.__new__(ToolService)
            service.config = {"paths": {"clip_dir": str(preview_dir)}, "pipeline": {"evidence": {}}}

            preview = service.getImagePreview(source, 0.5)
            image = cv2.imread(str(preview), cv2.IMREAD_COLOR)

            self.assertEqual(image.shape[:2], (50, 100))
            self.assertEqual(service.getImagePreview(source, 1.0), source)

    def test_each_track_keeps_its_own_registry_reference(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []
        tracks = [
            {"trackId": "1", "registryReferenceIds": ["ref-003-a"]},
            {"trackId": "2", "registryReferenceIds": ["ref-003-a"]},
        ]

        controller._display_tracks(tracks, include_clips=False, include_registry=True)

        self.assertEqual(
            [item["registryReferenceIds"] for item in controller.display_groups],
            [["ref-003-a"], ["ref-003-a"]],
        )

    def test_count_synthesis_uses_working_scope(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {"questionType": "count"}
        controller.session_id = "session-test"
        controller.question = "一共出现多少艘船？"
        controller.rounds = []
        controller.tool_chain = []
        controller.tool_records = []
        controller.display_record = None
        controller.display_groups = []
        controller.display_limit = 3
        controller.event_handler = None
        controller.working_scope = {
            "tracksSnapshot": {
                "tracks": [
                    {"trackId": "1", "embeddingScore": 0.4},
                    {"trackId": "2", "embeddingScore": 0.9},
                    {"trackId": "3", "embeddingScore": 0.8},
                ]
            },
            "deduplicatedCount": {
                "highThresholdShipCount": 2,
                "highGroups": [["1", "2"], ["3"]],
            },
        }

        result = controller._synthesize("sufficient", "去重完成")

        self.assertEqual(result["count"], 2)
        self.assertIn("统计结果：2 艘船", result["conclusion"])
        self.assertEqual(result["minimumCount"], 2)
        self.assertEqual(result["confirmedCount"], 2)
        self.assertEqual(result["planMode"], "langgraph")

    def test_sensitive_count_uses_minimum_and_builds_merge_evidence_groups(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {"questionType": "count", "operation": "count"}
        controller.session_id = "session-sensitive-count"
        controller.question = "视频中一共出现多少艘船？"
        controller.rounds = []
        controller.tool_chain = []
        controller.tool_records = []
        controller.display_record = None
        controller.display_groups = []
        controller.display_limit = 3
        controller.event_handler = None
        controller.working_scope = {
            "tracks": {
                "tracks": [
                    {"trackId": "1", "startTime": 1, "endTime": 2},
                    {"trackId": "10", "startTime": 3, "endTime": 4},
                    {"trackId": "11", "startTime": 5, "endTime": 6},
                    {"trackId": "20", "startTime": 7, "endTime": 8},
                ]
            },
            "frames": {
                "keyframesByTrack": {
                    track_id: {"keyframes": [{"keyframeId": f"frame-{track_id}", "retentionScore": 0.9}]}
                    for track_id in ("1", "10", "11", "20")
                }
            },
            "dedup": {
                "trackCount": 30,
                "minimumShipCount": 15,
                "confirmedShipCount": 21,
                "highThresholdShipCount": 21,
                "lowThresholdShipCount": 15,
                "countStability": "sensitive",
                "confirmedMergeGroups": [
                    {"groupId": "confirmed-1", "trackIds": ["1", "10"], "minimumScore": 0.94}
                ],
                "pendingMergeGroups": [
                    {
                        "groupId": "pending-1",
                        "trackIds": ["1", "10", "11"],
                        "currentGroups": [["1", "10"], ["11"]],
                        "minimumScore": 0.82,
                        "possibleReduction": 1,
                    },
                    {
                        "groupId": "pending-2",
                        "trackIds": ["20", "21"],
                        "currentGroups": [["20"], ["21"]],
                        "minimumScore": 0.79,
                        "possibleReduction": 1,
                    },
                ],
            },
        }

        result = controller._synthesize("sufficient", "去重完成")

        self.assertEqual(result["count"], 15)
        self.assertEqual(result["countRange"], {"minimum": 15, "confirmed": 21})
        self.assertEqual(result["conclusion"], "统计结果：至少 15 艘船")
        self.assertIn("按高阈值确认合并后为 21 艘", result["answerText"])
        self.assertIn("最少为 15 艘", result["answerText"])
        self.assertEqual([group["groupType"] for group in result["displayGroups"]], ["confirmed", "pending", "pending"])
        self.assertEqual(result["displayGroups"][0]["mergedTrackIds"], ["1", "10"])
        self.assertEqual(result["displayGroups"][0]["keyframeIds"], ["frame-1", "frame-10"])
        self.assertNotIn("clipTrackId", result["displayGroups"][0])


    def test_count_evidence_lists_every_vessel_unit_with_tracks_times_and_merge_state(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {"questionType": "count", "operation": "count"}
        controller.session_id = "session-count-ledger"
        controller.question = "这个时间段一共出现多少艘船？"
        controller.rounds = []
        controller.tool_chain = []
        controller.tool_records = []
        controller.display_record = None
        controller.display_groups = []
        controller.display_limit = 2
        controller.event_handler = None
        track_ids = ("17", "19", "21", "22", "23", "24")
        controller.working_scope = {
            "tracks": {
                "tracks": [
                    {"trackId": track_id, "startTime": index * 10 + 1, "endTime": index * 10 + 8}
                    for index, track_id in enumerate(track_ids)
                ]
            },
            "frames": {
                "keyframesByTrack": {
                    track_id: {"keyframes": [{"keyframeId": f"frame-{track_id}", "retentionScore": 0.9}]}
                    for track_id in track_ids
                }
            },
            "dedup": {
                "trackCount": 6,
                "minimumShipCount": 4,
                "confirmedShipCount": 5,
                "highGroups": [["21", "22"], ["17"], ["19"], ["23"], ["24"]],
                "lowGroups": [["21", "22"], ["17", "19"], ["23"], ["24"]],
                "confirmedMergeGroups": [
                    {"groupId": "confirmed-1", "trackIds": ["21", "22"], "minimumScore": 0.907}
                ],
                "pendingMergeGroups": [
                    {
                        "groupId": "pending-1",
                        "trackIds": ["17", "19"],
                        "currentGroups": [["17"], ["19"]],
                        "minimumScore": 0.810,
                    }
                ],
            },
        }

        result = controller._synthesize("sufficient", "去重完成")

        self.assertEqual(len(result["tracks"]), 6)
        ledger = result["countEvidence"]
        self.assertEqual(ledger["minimumShipCount"], 4)
        self.assertEqual(ledger["evidenceUnitCount"], 4)
        self.assertTrue(ledger["coverageComplete"])
        self.assertEqual([unit["trackIds"] for unit in ledger["vesselUnits"]], [
            ["21", "22"], ["17", "19"], ["23"], ["24"]
        ])
        self.assertEqual([unit["mergeState"] for unit in ledger["vesselUnits"]], [
            "confirmed", "pending", "independent", "independent"
        ])
        self.assertEqual(ledger["vesselUnits"][0]["tracks"][0]["startTime"], 21)
        self.assertEqual(ledger["vesselUnits"][0]["tracks"][0]["keyframeId"], "frame-21")
        self.assertEqual(ledger["vesselUnits"][1]["tracks"][1]["endTime"], 18)




if __name__ == "__main__":
    unittest.main()
