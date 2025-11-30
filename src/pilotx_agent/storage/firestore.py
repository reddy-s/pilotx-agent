from __future__ import annotations

import logging
import os
import pickle

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.cloud.firestore_v1 import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter
from typing_extensions import override

from ..config import ServiceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_model_dump(obj) -> Dict[str, Any] | None:
    try:
        return obj.model_dump(exclude_none=True, mode="json")  # type: ignore[attr-defined]
    except Exception:
        return None


class FirestoreSessionService(BaseSessionService):

    def __init__(self):
        config = ServiceConfig.get_or_create_instance()
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.info("GOOGLE_APPLICATION_CREDENTIALS not set; relying on ADC.")
        self.client: AsyncClient = AsyncClient(database=config.firebase["database"])
        self.col_sessions = self.client.collection(config.firebase["collection"])
        logger.info(
            "FirestoreSessionService initialised (project=%s)", self.client.project
        )

    def __repr__(self):
        return f"<FirestoreSessionService project={self.client.project}>"

    @staticmethod
    def _generate_id() -> str:
        return uuid4().hex

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        now = _now_utc()
        # Use provided session_id or generate one
        sid = session_id or self._generate_id()
        doc_ref = self.col_sessions.document(f"{app_name}:{user_id}:{sid}")
        if state:
            filtered_state = {
                k: v for k, v in state.items() if not k.startswith("temp:")
            }
        else:
            filtered_state = {}

        await doc_ref.set(
            {
                "app_name": app_name,
                "user_id": user_id,
                "id": sid,
                "state": filtered_state,
                "create_time": now,
                "update_time": now,
                "ttl": now + timedelta(days=180),
            }
        )

        return Session(
            app_name=str(app_name),
            user_id=str(user_id),
            id=str(sid),
            state=state or {},
            last_update_time=now.timestamp(),
        )

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        doc_ref = self.col_sessions.document(f"{app_name}:{user_id}:{session_id}")
        snap = await doc_ref.get()
        if not snap.exists:
            return None

        data = snap.to_dict() or {}
        update_ts = data.get("update_time", _now_utc()).timestamp()
        session = Session(
            app_name=str(app_name),
            user_id=str(user_id),
            id=str(session_id),
            state=data.get("state", {}),
            last_update_time=update_ts,
        )

        # Load events from subcollection
        events_col = doc_ref.collection("events")
        query = events_col.order_by("timestamp")
        if config and config.after_timestamp:
            query = query.where(
                filter=FieldFilter(
                    "timestamp",
                    ">=",
                    datetime.fromtimestamp(config.after_timestamp, tz=timezone.utc),
                )
            )

        events_docs = [d async for d in query.stream()]
        events: list[Event] = [
            self._doc_to_event(d.to_dict() or {}) for d in events_docs
        ]

        # Keep only events with ts <= update_time (Vertex parity)
        events = [e for e in events if e.timestamp <= update_ts]

        # Post-filtering like Vertex implementation
        if config:
            if config.num_recent_events:
                events = events[-config.num_recent_events :]
            elif config.after_timestamp:
                i = len(events) - 1
                while i >= 0 and events[i].timestamp >= config.after_timestamp:
                    i -= 1
                if i >= 0:
                    events = events[i:]

        session.events = events
        return session

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        if not user_id:
            raise ValueError("user_id must be provided to list sessions.")
        q = self.col_sessions.where(
            filter=FieldFilter("app_name", "==", app_name)
        ).where(filter=FieldFilter("user_id", "==", user_id))
        snaps = [s async for s in q.stream()]

        sessions: list[Session] = []
        for s in snaps:
            d = s.to_dict() or {}
            sessions.append(
                Session(
                    app_name=app_name,
                    user_id=user_id,
                    id=d.get("id"),
                    state=d.get("state", {}),
                    last_update_time=(d.get("update_time") or _now_utc()).timestamp(),
                )
            )
        return ListSessionsResponse(sessions=sessions)

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        doc_ref = self.col_sessions.document(f"{app_name}:{user_id}:{session_id}")
        # Delete events subcollection in batches
        events_col = doc_ref.collection("events")
        batch = self.client.batch()
        # Stream then batch-delete
        async for ev in events_col.stream():
            batch.delete(ev.reference)
        await batch.commit()
        # Delete the session document
        await doc_ref.delete()

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        # Update in-memory first (BaseSessionService mutates session.state)
        await super().append_event(session=session, event=event)

        doc_ref = self.col_sessions.document(
            f"{session.app_name}:{session.user_id}:{session.id}"
        )
        events_col = doc_ref.collection("events")

        # Persist event document
        event_doc = self._event_to_doc(session, event)
        await events_col.add(event_doc)

        # Touch session's update_time and persist merged state
        now = _now_utc()

        filtered_state = {
            k: v for k, v in session.state.items() if not k.startswith("temp:")
        }
        await doc_ref.update(
            {
                "state": filtered_state,
                "update_time": now,
            }
        )

        return event

    def _event_to_doc(self, session: Session, event: Event) -> Dict[str, Any]:
        content_json = _safe_model_dump(event.content)
        grounding_json = _safe_model_dump(event.grounding_metadata)

        actions_bytes: bytes | None = None
        if event.actions is not None:
            actions_bytes = pickle.dumps(event.actions)

        return {
            "id": event.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "session_id": session.id,
            "invocation_id": event.invocation_id,
            "author": event.author,
            "branch": event.branch,
            "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc),
            "content": content_json,
            "actions": actions_bytes,  # stored as bytes in Firestore
            "long_running_tool_ids": (
                list(event.long_running_tool_ids)
                if event.long_running_tool_ids
                else None
            ),
            "grounding_metadata": grounding_json,
            "partial": event.partial,
            "turn_complete": event.turn_complete,
            "error_code": event.error_code,
            "error_message": event.error_message,
            "interrupted": event.interrupted,
        }

    def _doc_to_event(self, d: Dict[str, Any]) -> Event:
        actions_obj: EventActions | None = None
        raw_actions = d.get("actions")
        if isinstance(raw_actions, (bytes, bytearray)):
            try:
                actions_obj = pickle.loads(bytes(raw_actions))
            except Exception:
                actions_obj = None

        return Event(
            id=d.get("id", ""),
            invocation_id=d.get("invocation_id", ""),
            author=d.get("author", ""),
            branch=d.get("branch"),
            actions=actions_obj or EventActions(),
            timestamp=(d.get("timestamp") or _now_utc()).timestamp(),
            content=d.get("content"),
            long_running_tool_ids=set(d.get("long_running_tool_ids") or []),
            partial=d.get("partial"),
            turn_complete=d.get("turn_complete"),
            error_code=d.get("error_code"),
            error_message=d.get("error_message"),
            interrupted=d.get("interrupted"),
            grounding_metadata=d.get("grounding_metadata"),
        )
