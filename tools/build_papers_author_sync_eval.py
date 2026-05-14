#!/usr/bin/env python3
"""Build first-author Papers organization artifacts and browser sync script."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "papers_topic_organization" / "papers_by_topic.json"
OUTPUT_DIR = ROOT / "papers_author_organization"
AUTHORS_JSON = OUTPUT_DIR / "first_author_lists.json"
README = OUTPUT_DIR / "README.md"
SYNC_JS = OUTPUT_DIR / "sync_first_author_lists_eval.js"

COLLECTION_ID = "d3931a53-32c4-43aa-999d-1d5c35304e2c"
PARENT_NAME = "Codex First Author Organization"
MIN_FIRST_AUTHOR_PAPERS = 3
CLIENT_QUERY = (
    "client=desktop_wrapper&"
    "client_id=4e7fce6a-efa4-4297-8ec7-412f721517b4&"
    "client_version=5.0.28%20/%20webapp%205.4.15"
)


def compact(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def first_author_name(authors: list[object]) -> str:
    if not authors:
        return ""
    first = compact(authors[0])
    if not first:
        return ""
    if "," in first:
        return first.split(",", 1)[0].strip()
    return first.split()[-1]


def write_readme(author_rows: dict[str, list[dict[str, object]]]) -> None:
    total = sum(len(rows) for rows in author_rows.values())
    with README.open("w", encoding="utf-8") as fh:
        fh.write("# Papers First Author Organization\n\n")
        fh.write(
            "This organization creates one Papers sublist for every first author "
            f"with at least {MIN_FIRST_AUTHOR_PAPERS} first-authored papers.\n\n"
        )
        fh.write(f"- Parent list: `{PARENT_NAME}`\n")
        fh.write(f"- First-author lists: {len(author_rows)}\n")
        fh.write(f"- Papers in first-author lists: {total}\n")
        fh.write(f"- Machine-readable table: `first_author_lists.json`\n\n")
        fh.write("## First Author Counts\n\n")
        fh.write("| First author | Papers |\n")
        fh.write("|---|---:|\n")
        for author, rows in sorted(author_rows.items(), key=lambda item: (-len(item[1]), item[0].lower())):
            fh.write(f"| {author} | {len(rows)} |\n")
        fh.write("\n## Papers By First Author\n\n")
        for author, rows in sorted(author_rows.items(), key=lambda item: (-len(item[1]), item[0].lower())):
            fh.write(f"### {author} ({len(rows)})\n\n")
            for row in sorted(rows, key=lambda item: (str(item.get("year") or ""), str(item.get("title") or "").lower())):
                year = row.get("year") or "n.d."
                title = row.get("title") or row["id"]
                fh.write(f"- {year} - {title}\n")
            fh.write("\n")


def build_sync_script(author_items: dict[str, list[str]], all_library_item_ids: list[str]) -> str:
    payload = json.dumps(author_items, separators=(",", ":"), ensure_ascii=False)
    all_items_payload = json.dumps(all_library_item_ids, separators=(",", ":"))
    return f"""(async () => {{
  const COLLECTION_ID = {json.dumps(COLLECTION_ID)};
  const PARENT_NAME = {json.dumps(PARENT_NAME)};
  const CLIENT_QUERY = {json.dumps(CLIENT_QUERY)};
  const AUTHOR_ITEMS = {payload};
  const ALL_LIBRARY_ITEM_IDS = {all_items_payload};
  const BASE = `https://sync.readcube.com/collections/${{COLLECTION_ID}}`;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function api(path, options = {{}}) {{
    const separator = path.includes("?") ? "&" : "?";
    const url = `${{BASE}}${{path}}${{separator}}${{CLIENT_QUERY}}`;
    const response = await fetch(url, {{
      credentials: "include",
      ...options,
      headers: {{
        "Content-Type": "application/json",
        ...(options.headers || {{}})
      }}
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

  const groupedItemIds = [...new Set(Object.values(AUTHOR_ITEMS).flat())].sort();
  const summary = {{
    parent: {{ id: parent.id, name: parent.name, item_count: groupedItemIds.length }},
    authors: []
  }};

  for (const batch of chunk(ALL_LIBRARY_ITEM_IDS, 100)) {{
    await api(`/lists/${{parent.id}}/remove_items`, {{
      method: "POST",
      body: JSON.stringify({{ item_ids: batch }})
    }});
    await sleep(50);
  }}
  for (const batch of chunk(groupedItemIds, 100)) {{
    await api(`/lists/${{parent.id}}/add_items`, {{
      method: "POST",
      body: JSON.stringify({{ item_ids: batch }})
    }});
    await sleep(50);
  }}

  for (const [author, ids] of Object.entries(AUTHOR_ITEMS)) {{
    const matchingChildren = lists.filter((list) =>
      !list.deleted && list.parent_id === parent.id && list.name === author
    );
    let child = matchingChildren[0];
    if (!child) {{
      const created = await api("/lists", {{
        method: "POST",
        body: JSON.stringify({{
          list: {{
            collection_id: COLLECTION_ID,
            parent_id: parent.id,
            name: author
          }}
        }})
      }});
      child = created.list;
      lists = [...lists, child];
    }}

    for (const duplicate of matchingChildren.slice(1)) {{
      await api(`/lists/${{duplicate.id}}`, {{ method: "DELETE" }});
      await sleep(50);
      summary.authors.push({{ id: duplicate.id, name: duplicate.name, deleted_duplicate: true }});
    }}

    for (const batch of chunk(ids, 100)) {{
      await api(`/lists/${{child.id}}/add_items`, {{
        method: "POST",
        body: JSON.stringify({{ item_ids: batch }})
      }});
      await sleep(50);
    }}
    summary.authors.push({{ id: child.id, name: author, item_count: ids.length }});
  }}

  const desiredNames = new Set(Object.keys(AUTHOR_ITEMS));
  for (const list of lists) {{
    if (!list.deleted && list.parent_id === parent.id && !desiredNames.has(list.name)) {{
      await api(`/lists/${{list.id}}`, {{ method: "DELETE" }});
      await sleep(50);
      summary.authors.push({{ id: list.id, name: list.name, deleted: true }});
    }}
  }}

  return JSON.stringify(summary, null, 2);
}})()"""


def main() -> None:
    papers = json.loads(SOURCE.read_text(encoding="utf-8"))
    counts = Counter(first_author_name(row.get("authors") or []) for row in papers)
    selected_authors = {
        author
        for author, count in counts.items()
        if author and count >= MIN_FIRST_AUTHOR_PAPERS
    }

    author_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in papers:
        author = first_author_name(row.get("authors") or [])
        if author in selected_authors:
            author_rows[author].append(row)

    author_rows = {
        author: rows
        for author, rows in sorted(author_rows.items(), key=lambda item: (-len(item[1]), item[0].lower()))
    }
    author_items = {
        author: sorted(str(row["id"]) for row in rows)
        for author, rows in author_rows.items()
    }
    all_library_item_ids = sorted(str(row["id"]) for row in papers)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUTHORS_JSON.write_text(json.dumps(author_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(author_rows)
    SYNC_JS.write_text(build_sync_script(author_items, all_library_item_ids), encoding="utf-8")

    print(f"Wrote {README}")
    print(f"Wrote {AUTHORS_JSON}")
    print(f"Wrote {SYNC_JS}")
    print(f"First-author lists: {len(author_rows)}")
    print(f"Membership assignments: {sum(len(rows) for rows in author_rows.values())}")


if __name__ == "__main__":
    main()
