"""Catalog persistence (SQLAlchemy 2.0). SQLite by default; any SQLAlchemy URL works
(e.g. ``postgresql+psycopg://...``) via ``MELODYAI_DATABASE_URL``."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(300))
    artist: Mapped[str] = mapped_column(String(300), default="Unknown")
    album: Mapped[str | None] = mapped_column(String(300), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_type: Mapped[str] = mapped_column(String(32), default="midi")
    reference_filename: Mapped[str | None] = mapped_column(String(300), nullable=True)
    composition_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String(500), nullable=True)
    license: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    index_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    n_notes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_windows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    listen_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extra_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    lyrics: Mapped[Lyrics | None] = relationship(back_populates="song", cascade="all, delete-orphan",
                                                  uselist=False)

    @property
    def extra(self) -> dict:
        return json.loads(self.extra_json or "{}")


class Lyrics(Base):
    __tablename__ = "lyrics"

    song_id: Mapped[str] = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"), primary_key=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(500), nullable=True)
    license: Mapped[str | None] = mapped_column(String(200), nullable=True)

    song: Mapped[Song] = relationship(back_populates="lyrics")


class Recognition(Base):
    __tablename__ = "recognitions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    mode: Mapped[str] = mapped_column(String(16))
    resolved_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    top_song_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    result_json: Mapped[str] = mapped_column(Text)
    audio_retained: Mapped[bool] = mapped_column(Boolean, default=False)


class Job(Base):
    """Asynchronous job (evaluation, training, reindexing)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def make_session_factory(url: str):
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk(dbapi_conn, _):  # enforce ON DELETE CASCADE in SQLite
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
