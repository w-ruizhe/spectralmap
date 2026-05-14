#!/usr/bin/env python3
"""Build a browser eval script that syncs topic lists into Papers/ReadCube."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "papers_topic_organization" / "papers_by_topic.json"
OUTPUT = ROOT / "papers_topic_organization" / "sync_topic_lists_eval.js"

COLLECTION_ID = "d3931a53-32c4-43aa-999d-1d5c35304e2c"
PARENT_NAME = "Codex Topic Organization"
CLIENT_QUERY = (
    "client=desktop_wrapper&"
    "client_id=4e7fce6a-efa4-4297-8ec7-412f721517b4&"
    "client_version=5.0.28%20/%20webapp%205.4.15"
)
CATEGORY_NAMES = [
    "Brown dwarf",
    "Hot Jupiter",
    "White Dwarf Pollution",
    "Numerical",
    "Machine Learning",
    "Dynamics",
    "Formation",
    "Retrieval",
    "Review",
    "High-Res",
    "Secondary Eclipse",
    "Rotational Light Curve",
    "Transit",
]


def main() -> None:
    papers = json.loads(INPUT.read_text())
    topic_items: dict[str, list[str]] = {}

    for paper in papers:
        paper_id = paper["id"]
        for topic in paper.get("topics", []):
            topic_items.setdefault(topic, []).append(paper_id)

    topic_items = {
        topic: sorted(set(topic_items.get(topic, [])))
        for topic in CATEGORY_NAMES
    }

    payload = json.dumps(topic_items, separators=(",", ":"), ensure_ascii=False)
    script = f"""(async () => {{
  const COLLECTION_ID = {json.dumps(COLLECTION_ID)};
  const PARENT_NAME = {json.dumps(PARENT_NAME)};
  const CLIENT_QUERY = {json.dumps(CLIENT_QUERY)};
  const TOPIC_ITEMS = {payload};
  const BASE = `https://sync.readcube.com/collections/${{COLLECTION_ID}}`;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function api(path, options = {{}}) {{
    const separator = path.includes("?") ? "&" : "?";
    const url = `${{BASE}}${{path}}${{separator}}${{CLIENT_QUERY}}`;
    const headers = {{
      "Content-Type": "application/json",
      ...(options.headers || {{}})
    }};
    const response = await fetch(url, {{
      credentials: "include",
      ...options,
      headers
    }});
    const text = await response.text();
    let data;
    try {{
      data = text ? JSON.parse(text) : {{}};
    }} catch (error) {{
      throw new Error(`${{response.status}} ${{response.statusText}}: ${{text.slice(0, 500)}}`);
    }}
    if (!response.ok || data.status === "error") {{
      throw new Error(`${{response.status}} ${{response.statusText}}: ${{JSON.stringify(data).slice(0, 1000)}}`);
    }}
    return data;
  }}

  function chunk(values, size) {{
    const chunks = [];
    for (let index = 0; index < values.length; index += size) {{
      chunks.push(values.slice(index, index + size));
    }}
    return chunks;
  }}

  let lists = (await api("/lists")).lists || [];
  let parent = lists.find((list) =>
    !list.deleted && !list.parent_id && list.name === PARENT_NAME
  );

  if (!parent) {{
    const created = await api("/lists", {{
      method: "POST",
      body: JSON.stringify({{
        list: {{
          collection_id: COLLECTION_ID,
          name: PARENT_NAME
        }}
      }})
    }});
    parent = created.list;
    lists = [...lists, parent];
  }}

  const allItemIds = [...new Set(Object.values(TOPIC_ITEMS).flat())].sort();
  const summary = {{
    parent: {{ id: parent.id, name: parent.name, item_count: allItemIds.length }},
    topics: []
  }};

  for (const batch of chunk(allItemIds, 100)) {{
    await api(`/lists/${{parent.id}}/remove_items`, {{
      method: "POST",
      body: JSON.stringify({{ item_ids: batch }})
    }});
    await sleep(75);
    await api(`/lists/${{parent.id}}/add_items`, {{
      method: "POST",
      body: JSON.stringify({{ item_ids: batch }})
    }});
    await sleep(75);
  }}

  for (const [topic, ids] of Object.entries(TOPIC_ITEMS)) {{
    let child = lists.find((list) =>
      !list.deleted && list.parent_id === parent.id && list.name === topic
    );

    if (!child) {{
      const created = await api("/lists", {{
        method: "POST",
        body: JSON.stringify({{
          list: {{
            collection_id: COLLECTION_ID,
            parent_id: parent.id,
            name: topic
          }}
        }})
      }});
      child = created.list;
      lists = [...lists, child];
    }}

    for (const batch of chunk(allItemIds, 100)) {{
      await api(`/lists/${{child.id}}/remove_items`, {{
        method: "POST",
        body: JSON.stringify({{ item_ids: batch }})
      }});
      await sleep(75);
    }}

    for (const batch of chunk(ids, 100)) {{
      await api(`/lists/${{child.id}}/add_items`, {{
        method: "POST",
        body: JSON.stringify({{ item_ids: batch }})
      }});
      await sleep(75);
    }}

    summary.topics.push({{ id: child.id, name: topic, item_count: ids.length }});
  }}

  const desiredNames = new Set(Object.keys(TOPIC_ITEMS));
  for (const list of lists) {{
    if (!list.deleted && list.parent_id === parent.id && !desiredNames.has(list.name)) {{
      await api(`/lists/${{list.id}}`, {{ method: "DELETE" }});
      await sleep(75);
      summary.topics.push({{ id: list.id, name: list.name, deleted: true }});
    }}
  }}

  window.dispatchEvent(new Event("focus"));
  return JSON.stringify(summary, null, 2);
}})()"""

    OUTPUT.write_text(script)
    print(f"Wrote {OUTPUT}")
    print(f"Topics: {len(topic_items)}")
    print(f"Membership assignments: {sum(len(v) for v in topic_items.values())}")


if __name__ == "__main__":
    main()
