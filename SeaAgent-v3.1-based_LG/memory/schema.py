"""三层记忆的 CSV 字段定义。"""
TRACK_FIELDS = (
    "track_id", "start_time", "end_time", "video_start_time", "video_end_time", "final_hull_number",
    "final_description", "final_match_type", "trajectory_path",
)
KEYFRAME_FIELDS = (
    "keyframe_id", "track_id", "timestamp", "keyframe_path", "bbox",
    "quality_score", "retention_score", "has_readable_hull_number",
    "vlm_hull_number", "readability_confidence", "description",
    "keyframe_vector_id", "is_embedded",
)
REGISTRY_FIELDS = ("registry_id", "hull_number", "aliases", "description", "structured_attributes")
REGISTRY_IMAGE_FIELDS = ("reference_id", "registry_id", "image_path", "registry_vector_id", "is_embedded")
QA_SESSION_FIELDS = ("session_id", "query_info", "final_result")
QA_ROUND_FIELDS = ("round_id", "session_id", "plan", "reflection")
QA_EVIDENCE_FIELDS = ("evidence_id", "round_id", "tool_result", "evidence_source")
