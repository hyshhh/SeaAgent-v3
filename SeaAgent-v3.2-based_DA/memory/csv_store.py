"""线程安全、原子写入的轻量 CSV 表。"""

from __future__ import annotations

import csv
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)


# 问答审计字段可能包含较多轨迹和关键帧摘要；默认 128KB 上限过小。
csv.field_size_limit(16 * 1024 * 1024)


class CsvTable:
    def __init__(self, path: str | Path, fields: Iterable[str]):
        self.path = Path(path)
        self.fields = tuple(fields)
        self._lock = threading.RLock()
        self._ensure()

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_unlocked([])

    def header(self) -> tuple[str, ...]:
        """文件当前的表头（可能落后于 ``self.fields``）；文件不存在或为空时返回空元组。"""
        with self._lock:
            if not self.path.exists():
                return ()
            with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
                return tuple(next(csv.reader(stream), []))

    def rows(self) -> list[dict[str, str]]:
        with self._lock:
            with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]

    def find(self, predicate: Callable[[dict[str, str]], bool]) -> list[dict[str, str]]:
        return [row for row in self.rows() if predicate(row)]

    def replace_all(self, rows: Iterable[Mapping[str, object]], *, expect_stale_columns: bool = False) -> None:
        """整表替换；``expect_stale_columns`` 供迁移调用，表示旧列是有意被替代的，无需告警。"""
        with self._lock:
            self._write_unlocked(rows, warn_stale=not expect_stale_columns)

    def upsert(self, row: Mapping[str, object], key: str) -> None:
        key_value = str(row.get(key, ""))
        if not key_value:
            raise ValueError(f"缺少主键字段：{key}")
        with self._lock:
            rows = self.rows()
            normalized = self._normalize(row)
            for index, current in enumerate(rows):
                if current.get(key) == key_value:
                    rows[index] = normalized
                    break
            else:
                rows.append(normalized)
            self._write_unlocked(rows)

    def delete(self, predicate: Callable[[dict[str, str]], bool]) -> list[dict[str, str]]:
        with self._lock:
            rows = self.rows()
            removed = [row for row in rows if predicate(row)]
            if removed:
                self._write_unlocked(row for row in rows if not predicate(row))
            return removed

    def _normalize(self, row: Mapping[str, object]) -> dict[str, str]:
        result: dict[str, str] = {}
        for field in self.fields:
            value = row.get(field, "")
            if isinstance(value, bool):
                result[field] = "true" if value else "false"
            else:
                result[field] = "" if value is None else str(value)
        return result

    def _write_unlocked(self, rows: Iterable[Mapping[str, object]], *, warn_stale: bool = True) -> None:
        # 整表按当前字段重写：文件里多出来的旧列会被 dict 写入器直接丢弃。
        # 这里出声提醒，避免字段改名后历史数据在某次写入时无声蒸发（先迁移、再写入）。
        stale = [name for name in self.header() if name and name not in self.fields]
        if stale and warn_stale:
            logger.warning("表 %s 按当前字段重写，将丢弃旧列：%s（请先迁移数据）", self.path.name, "、".join(stale))
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(self._normalize(row))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
