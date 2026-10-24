"""Small SQLite-backed registry for student records."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ValidationError(ValueError):
    """Raised when a record field fails validation."""


@dataclass
class StudentRecord:
    record_id: str
    name: str
    age: int
    track: str
    email: str
    courses: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        courses = ", ".join(self.courses) or "-"
        return (f"{self.record_id}  {self.name:<18} age {self.age:<3} "
                f"{self.track:<14} {self.email:<26} courses: {courses}")


def clean_name(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValidationError("name must not be empty")
    return value


def clean_age(value) -> int:
    try:
        age = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"age must be a whole number, got {value!r}")
    if not 1 <= age <= 120:
        raise ValidationError("age must be between 1 and 120")
    return age


def clean_email(value: str) -> str:
    value = (value or "").strip()
    if not EMAIL_RE.match(value):
        raise ValidationError(f"invalid email address: {value!r}")
    return value


def split_courses(text: str | None) -> list[str]:
    seen: list[str] = []
    for part in (text or "").split(","):
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return seen


class Registry:
    """Persistent store; identifiers keep increasing across restarts."""

    def __init__(self, path: str | Path = "students.db"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                seq     INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT NOT NULL,
                age     INTEGER NOT NULL,
                track   TEXT NOT NULL DEFAULT '',
                email   TEXT NOT NULL UNIQUE,
                courses TEXT NOT NULL DEFAULT ''
            )""")
        self.conn.commit()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _to_record(row: sqlite3.Row) -> StudentRecord:
        return StudentRecord(f"S{row['seq']:04d}", row["name"], row["age"],
                             row["track"], row["email"],
                             split_courses(row["courses"]))

    @staticmethod
    def _seq(record_id: str) -> int:
        m = re.fullmatch(r"[Ss](\d+)", record_id.strip())
        if not m:
            raise ValidationError(f"malformed id: {record_id!r}")
        return int(m.group(1))

    # -- operations ------------------------------------------------------
    def add(self, name, age, track, email, courses=None) -> StudentRecord:
        try:
            cur = self.conn.execute(
                "INSERT INTO students (name, age, track, email, courses) "
                "VALUES (?, ?, ?, ?, ?)",
                (clean_name(name), clean_age(age), (track or "").strip(),
                 clean_email(email), ", ".join(courses or [])))
        except sqlite3.IntegrityError:
            raise ValidationError("a student with this email already exists")
        self.conn.commit()
        return self.get(f"S{cur.lastrowid:04d}")

    def get(self, record_id: str) -> StudentRecord | None:
        row = self.conn.execute("SELECT * FROM students WHERE seq = ?",
                                (self._seq(record_id),)).fetchone()
        return self._to_record(row) if row else None

    def all(self) -> list[StudentRecord]:
        rows = self.conn.execute("SELECT * FROM students ORDER BY seq")
        return [self._to_record(r) for r in rows]

    def search(self, text: str) -> list[StudentRecord]:
        like = f"%{text.strip()}%"
        rows = self.conn.execute(
            "SELECT * FROM students WHERE name LIKE ? OR track LIKE ? "
            "OR courses LIKE ? ORDER BY seq", (like, like, like))
        return [self._to_record(r) for r in rows]

    def update(self, record_id: str, **changes) -> StudentRecord | None:
        rec = self.get(record_id)
        if rec is None:
            return None
        if "name" in changes:
            rec.name = clean_name(changes["name"])
        if "age" in changes:
            rec.age = clean_age(changes["age"])
        if "track" in changes:
            rec.track = changes["track"].strip()
        if "email" in changes:
            rec.email = clean_email(changes["email"])
        self.conn.execute(
            "UPDATE students SET name=?, age=?, track=?, email=? WHERE seq=?",
            (rec.name, rec.age, rec.track, rec.email, self._seq(record_id)))
        self.conn.commit()
        return rec

    def set_courses(self, record_id: str, add=(), remove=()) -> StudentRecord | None:
        rec = self.get(record_id)
        if rec is None:
            return None
        for c in add:
            if c.strip() and c.strip() not in rec.courses:
                rec.courses.append(c.strip())
        rec.courses = [c for c in rec.courses if c not in set(remove)]
        self.conn.execute("UPDATE students SET courses=? WHERE seq=?",
                          (", ".join(rec.courses), self._seq(record_id)))
        self.conn.commit()
        return rec

    def delete(self, record_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM students WHERE seq = ?",
                                (self._seq(record_id),))
        self.conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self.conn.close()
