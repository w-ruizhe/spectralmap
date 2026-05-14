#!/usr/bin/env python3
"""Write Codex-generated topic lists directly into the local Papers database."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HOME = Path.home()
PAPERS_DIR = HOME / "Library/Application Support/Papers"
DB_PATH = PAPERS_DIR / "d3931a53-32c4-43aa-999d-1d5c35304e2c.db"
TOPICS_JSON = Path("papers_topic_organization/papers_by_topic.json")
BACKUP_DIR = Path("papers_topic_organization/backups")
PARENT_LIST_NAME = "Codex Topic Organization"
CREATED_BY = "codex direct-db topic organizer"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_topic_assignments() -> tuple[list[str], dict[str, list[str]]]:
    rows = json.loads(TOPICS_JSON.read_text(encoding="utf-8"))
    all_item_ids: set[str] = set()
    topic_items: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        item_id = row["id"]
        for topic in row.get("topics") or []:
            if not topic:
                continue
            all_item_ids.add(item_id)
            topic_items[topic].add(item_id)
    return sorted(all_item_ids), {topic: sorted(ids) for topic, ids in sorted(topic_items.items())}


def load_user_and_collection(conn: sqlite3.Connection) -> tuple[str, str, dict, dict]:
    collection_id, collection_raw = conn.execute("select id, json from collections limit 1").fetchone()
    user_raw = conn.execute("select json from user_details where id = ?", (collection_id,)).fetchone()
    collection = json.loads(collection_raw)
    user = json.loads(user_raw[0]) if user_raw else {}
    return collection_id, collection_raw, collection, user


def make_list_json(
    *,
    list_id: str,
    collection_id: str,
    name: str,
    item_ids: list[str],
    seq: int,
    user: dict,
    parent_id: str | None = None,
    created: str | None = None,
) -> dict:
    now = iso_now()
    data = {
        "seq": seq,
        "name": name,
        "created": created or now,
        "deleted": False,
        "user_id": collection_id,
        "modified": now,
        "createdby": CREATED_BY,
        "modifiedby": CREATED_BY,
        "import_data": {},
        "collection_id": collection_id,
        "id": list_id,
        "type": "list",
        "item_ids": item_ids,
        "user_name": user.get("name"),
        "user_email": user.get("email"),
    }
    if parent_id:
        data["parent_id"] = parent_id
    return data


def backup_db() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"{DB_PATH.name}.{stamp}.bak"
    shutil.copy2(DB_PATH, backup)
    return backup


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    if not TOPICS_JSON.exists():
        raise FileNotFoundError(TOPICS_JSON)

    all_item_ids, topic_items = load_topic_assignments()
    backup = backup_db()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("pragma foreign_keys = on")
        conn.execute("begin immediate")
        collection_id, collection_raw, collection, user = load_user_and_collection(conn)
        existing_rows = conn.execute("select id, json from lists").fetchall()
        existing_by_name = {}
        existing_children = {}
        max_seq = 0
        for list_id, raw in existing_rows:
            data = json.loads(raw)
            max_seq = max(max_seq, int(data.get("seq") or 0))
            existing_by_name[data.get("name")] = (list_id, data)

        parent_id, parent_data = existing_by_name.get(PARENT_LIST_NAME, (str(uuid.uuid4()), None))
        if parent_data:
            parent_created = parent_data.get("created")
        else:
            parent_created = None

        next_seq = max(max_seq, int(collection.get("tip") or 0))
        next_seq += 1
        parent_json = make_list_json(
            list_id=parent_id,
            collection_id=collection_id,
            name=PARENT_LIST_NAME,
            item_ids=all_item_ids,
            seq=next_seq,
            user=user,
            created=parent_created,
        )
        conn.execute(
            "insert into lists (id, collection_id, json) values (?, ?, ?) "
            "on conflict(id) do update set collection_id = excluded.collection_id, json = excluded.json",
            (parent_id, collection_id, json.dumps(parent_json, separators=(",", ":"))),
        )

        for list_id, raw in existing_rows:
            data = json.loads(raw)
            if data.get("parent_id") == parent_id:
                existing_children[data.get("name")] = (list_id, data)

        written = [(PARENT_LIST_NAME, len(all_item_ids))]
        for topic, item_ids in topic_items.items():
            child_id, child_data = existing_children.get(topic, (str(uuid.uuid4()), None))
            next_seq += 1
            child_json = make_list_json(
                list_id=child_id,
                collection_id=collection_id,
                name=topic,
                item_ids=item_ids,
                seq=next_seq,
                user=user,
                parent_id=parent_id,
                created=child_data.get("created") if child_data else None,
            )
            conn.execute(
                "insert into lists (id, collection_id, json) values (?, ?, ?) "
                "on conflict(id) do update set collection_id = excluded.collection_id, json = excluded.json",
                (child_id, collection_id, json.dumps(child_json, separators=(",", ":"))),
            )
            written.append((topic, len(item_ids)))

        for stale_topic, (stale_id, stale_data) in existing_children.items():
            if stale_topic in topic_items:
                continue
            next_seq += 1
            stale_data["deleted"] = True
            stale_data["item_ids"] = []
            stale_data["modified"] = iso_now()
            stale_data["modifiedby"] = CREATED_BY
            stale_data["seq"] = next_seq
            conn.execute(
                "insert into lists (id, collection_id, json) values (?, ?, ?) "
                "on conflict(id) do update set collection_id = excluded.collection_id, json = excluded.json",
                (stale_id, collection_id, json.dumps(stale_data, separators=(",", ":"))),
            )

        collection["tip"] = max(int(collection.get("tip") or 0), next_seq)
        collection["updated_at"] = int(time.time())
        conn.execute("update collections set json = ? where id = ?", (json.dumps(collection, separators=(",", ":")), collection_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Backup: {backup}")
    print(f"Updated database: {DB_PATH}")
    print(f"Parent list: {PARENT_LIST_NAME} ({len(all_item_ids)} items)")
    for topic, count in written[1:]:
        print(f"- {topic}: {count}")


if __name__ == "__main__":
    main()
