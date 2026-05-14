#!/usr/bin/env python3
"""Create a non-destructive topic organization for a local Papers library."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


HOME = Path.home()
PAPERS_DIR = HOME / "Library/Application Support/Papers"
SHARED_DB = HOME / "Library/Application Support/ReadCube Shared/shared.db"
OUTPUT_DIR = Path("papers_topic_organization")


GENERATED_PARENT_NAME = "Codex Topic Organization"


TOPICS = {
    "Brown dwarf": {
        "lists": {"Brown Dwarf", "Lhuman 16"},
        "terms": [
            "brown dwarf", "brown dwarfs", "directly imaged", "direct imaging",
            "planetary-mass companion", "substellar", "luhman 16", "luhman",
            "t dwarf", "l dwarf", "y dwarf", "ultracool", "vhs 1256",
            "simp", "2mass", "wise", "hr 8799",
        ],
    },
    "Hot Jupiter": {
        "lists": {"WASP-121b"},
        "terms": [
            "hot jupiter", "hot-jupiter", "ultrahot jupiter", "ultra-hot jupiter",
            "highly irradiated", "irradiated giant", "gas giant", "wasp-43",
            "wasp-121", "wasp-18", "wasp-33", "hd 189733", "hd 209458",
            "kelt-9", "hats", "hat-p", "toi-", "dayside", "nightside",
        ],
    },
    "White Dwarf Pollution": {
        "lists": {"White Dwarf"},
        "terms": [
            "white dwarf", "white dwarfs", "polluted", "pollution",
            "metal-polluted", "metal polluted", "daz", "dz", "wd 1145",
            "wd1145", "degenerate", "koester", "farihi", "zuckerman",
            "debris disk", "debris disc", "circumstellar", "photospheric",
            "diffusion", "accretion onto white dwarfs", "planetary debris",
        ],
    },
    "Numerical": {
        "lists": {"Numerical", "SPH"},
        "terms": [
            "numerical", "simulation", "simulations", "simulate", "simulated",
            "model", "models", "modeling", "modelling", "hydrodynamic",
            "hydrodynamics", "magnetohydrodynamic", "mhd", "gcm",
            "general circulation model", "sph", "n-body", "n body",
            "monte carlo", "finite difference", "computational",
        ],
    },
    "Machine Learning": {
        "lists": {"Machine Learning"},
        "terms": [
            "machine learning", "deep learning", "neural network",
            "neural networks", "random forest", "support vector",
            "gaussian process", "classifier", "classification",
            "regression", "emulator", "autoencoder", "cnn", "pca",
        ],
    },
    "Dynamics": {
        "lists": {"Dynamics", "Dynamical Tides", "Disintegrating Planets"},
        "terms": [
            "dynamics", "dynamical", "orbit", "orbits", "orbital",
            "eccentricity", "eccentric", "resonance", "migration",
            "scattering", "kozai", "lidov", "tides", "tidal",
            "tidal disruption", "roche", "disintegrating", "instability",
            "angular momentum", "spin-orbit", "precession",
        ],
    },
    "Formation": {
        "lists": {"Formation"},
        "terms": [
            "formation", "planet formation", "core accretion",
            "gravitational instability", "protoplanetary", "planetesimal",
            "planetesimals", "embryo", "embryos", "collision", "collisions",
            "impact", "impacts", "collisional cascade", "accretion disk",
            "accretion disc", "disk", "disc", "debris disk", "debris disc",
        ],
    },
    "Retrieval": {
        "lists": {"Retrieval"},
        "terms": [
            "retrieval", "retrievals", "atmospheric retrieval",
            "inverse problem", "bayesian retrieval", "abundance retrieval",
            "nested sampling", "mcmc", "posterior", "priors",
            "petitradtrans", "picaso", "chimera", "taurex", "nemesis",
        ],
    },
    "Review": {
        "lists": {"Review", "Reviews"},
        "terms": [
            "review", "reviews", "overview", "perspective", "introduction",
            "current challenges", "where do we stand", "white paper",
            "roadmap", "tutorial", "lecture", "book", "textbook",
        ],
    },
    "High-Res": {
        "lists": {"HRS"},
        "terms": [
            "high-resolution", "high resolution", "high-dispersion",
            "high dispersion", "hrs", "cross-correlation", "cross correlation",
            "doppler", "doppler spectroscopy", "radial velocity",
            "spectrograph", "igrins", "crires", "espresso",
        ],
    },
    "Secondary Eclipse": {
        "lists": {"Mapping", "WASP-121b"},
        "terms": [
            "secondary eclipse", "occultation", "eclipse", "eclipse mapping",
            "dayside emission", "dayside spectrum", "emission spectrum",
            "thermal emission", "brightness map", "map", "mapping",
        ],
    },
    "Rotational Light Curve": {
        "lists": {"Brown Dwarf"},
        "terms": [
            "rotational light curve", "rotational lightcurve", "light curve",
            "lightcurve", "variability", "variable", "rotation", "rotating",
            "spot", "spots", "cloud patch", "weather", "periodic",
        ],
    },
    "Transit": {
        "lists": {"Detectaility"},
        "terms": [
            "transit", "transits", "transiting", "transmission spectrum",
            "transmission spectroscopy", "primary transit", "transit depth",
            "limb darkening", "ingress", "egress", "radius ratio",
            "light-curve fitting", "light curve fitting",
        ],
    },
}

UNCATEGORIZED = None


def sqlite_uri(path: Path) -> str:
    return f"file:{path}?mode=ro&immutable=1"


def find_papers_db() -> Path:
    candidates = sorted(PAPERS_DIR.glob("*.db"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Papers database found in {PAPERS_DIR}")
    return candidates[0]


def load_path_map() -> dict[str, str]:
    paths: dict[str, str] = {}
    if not SHARED_DB.exists():
        return paths
    conn = sqlite3.connect(sqlite_uri(SHARED_DB), uri=True)
    try:
        for sha, path in conn.execute("select hash, path from paths"):
            paths[sha] = path
    finally:
        conn.close()
    return paths


def load_lists(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, set[str]]]:
    raw_lists = []
    generated_ids: set[str] = set()
    list_names: dict[str, str] = {}
    item_lists: dict[str, set[str]] = defaultdict(set)
    for list_id, raw in conn.execute("select id, json from lists"):
        data = json.loads(raw)
        if data.get("deleted"):
            continue
        raw_lists.append((list_id, data))
        if data.get("name") == GENERATED_PARENT_NAME:
            generated_ids.add(list_id)

    generated_ids.update(
        list_id
        for list_id, data in raw_lists
        if data.get("parent_id") in generated_ids
    )

    for list_id, data in raw_lists:
        if list_id in generated_ids:
            continue
        name = data.get("name") or list_id
        list_names[list_id] = name
        for item_id in data.get("item_ids") or []:
            item_lists[item_id].add(name)
    return list_names, item_lists


def compact(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "; ".join(str(x) for x in text)
    return re.sub(r"\s+", " ", str(text)).strip()


def score_topics(text: str, existing_lists: set[str]) -> list[tuple[str, int]]:
    haystack = f" {text.lower()} "
    scores: Counter[str] = Counter()
    for topic, config in TOPICS.items():
        for list_name in existing_lists:
            if list_name in config["lists"]:
                scores[topic] += 12
        for term in config["terms"]:
            if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", haystack):
                scores[topic] += 1 + min(3, len(term.split()) // 2)
    ranked = [(topic, score) for topic, score in scores.most_common() if score > 0]
    if not ranked:
        return []
    top_score = ranked[0][1]
    selected = [(topic, score) for topic, score in ranked if score >= 3 or score >= max(2, top_score * 0.35)]
    if not selected:
        selected = [ranked[0]]
    return selected[:6]


def clean_filename(value: str, limit: int = 140) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rstrip(" .") or "Untitled"


def topic_dir_name(topic: str) -> str:
    replacements = {
        "&": "and",
        "/": "-",
        ",": "",
    }
    name = topic
    for old, new in replacements.items():
        name = name.replace(old, new)
    return clean_filename(name, 80)


def first_author(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    first = str(authors[0]).strip()
    return first.split(",")[0].split()[-1] if first else "Unknown"


def collect_items(conn: sqlite3.Connection, item_lists: dict[str, set[str]], path_map: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item_id, raw in conn.execute("select id, json from items"):
        data = json.loads(raw)
        if data.get("deleted"):
            continue
        article = data.get("article") or {}
        ext_ids = data.get("ext_ids") or {}
        files = data.get("files") or []
        authors = article.get("authors") or []
        if not isinstance(authors, list):
            authors = [authors]
        file_paths = []
        for file_info in files:
            sha = file_info.get("sha256")
            path = path_map.get(sha)
            if path and Path(path).exists():
                file_paths.append(path)
        title = compact(article.get("title")) or compact(files[0].get("name") if files else "") or item_id
        abstract = compact(article.get("abstract"))
        journal = compact(article.get("journal"))
        year = compact(article.get("year"))
        text = " ".join([
            title,
            abstract,
            journal,
            compact(authors),
            compact(item_lists.get(item_id, set())),
            compact(ext_ids),
        ])
        scored = score_topics(text, item_lists.get(item_id, set()))
        topics = [topic for topic, _score in scored]
        rows.append({
            "id": item_id,
            "primary_topic": topics[0] if topics else "",
            "topics": topics,
            "topic_scores": dict(scored),
            "existing_lists": sorted(item_lists.get(item_id, set())),
            "title": title,
            "year": year,
            "authors": authors,
            "journal": journal,
            "doi": compact(ext_ids.get("doi")),
            "arxiv": compact(ext_ids.get("arxiv")),
            "file_paths": file_paths,
            "has_local_pdf": bool(file_paths),
        })
    return rows


def make_symlinks(rows: list[dict[str, object]]) -> None:
    topics_root = OUTPUT_DIR / "topics"
    if topics_root.exists():
        shutil.rmtree(topics_root)
    topics_root.mkdir(parents=True, exist_ok=True)
    used_names: Counter[str] = Counter()
    for row in rows:
        file_paths = row["file_paths"]
        if not file_paths:
            continue
        for topic in row["topics"]:
            if not topic:
                continue
            topic_dir = topics_root / topic_dir_name(str(topic))
            topic_dir.mkdir(parents=True, exist_ok=True)
            for idx, file_path in enumerate(file_paths, start=1):
                path = Path(str(file_path))
                year = str(row["year"] or "n.d.")
                author = first_author(row["authors"])  # type: ignore[arg-type]
                title = clean_filename(str(row["title"]), 90)
                suffix = f"_{idx}" if len(file_paths) > 1 else ""
                base = clean_filename(f"{year} - {author} - {title}{suffix}") + path.suffix.lower()
                key = str(topic_dir / base)
                used_names[key] += 1
                if used_names[key] > 1:
                    stem = Path(base).stem
                    base = f"{stem} ({used_names[key]}){path.suffix.lower()}"
                target = topic_dir / base
                if not target.exists():
                    os.symlink(path, target)


def write_outputs(rows: list[dict[str, object]], db_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_sorted = sorted(rows, key=lambda r: (str(r["primary_topic"]), str(r["year"]), str(r["title"]).lower()))
    csv_path = OUTPUT_DIR / "papers_by_topic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "primary_topic", "all_topics", "existing_lists", "year", "title",
                "authors", "journal", "doi", "arxiv", "has_local_pdf", "file_paths", "id",
            ],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({
                "primary_topic": row["primary_topic"],
                "all_topics": "; ".join(row["topics"]),
                "existing_lists": "; ".join(row["existing_lists"]),
                "year": row["year"],
                "title": row["title"],
                "authors": "; ".join(row["authors"]),
                "journal": row["journal"],
                "doi": row["doi"],
                "arxiv": row["arxiv"],
                "has_local_pdf": row["has_local_pdf"],
                "file_paths": "; ".join(row["file_paths"]),
                "id": row["id"],
            })

    with (OUTPUT_DIR / "papers_by_topic.json").open("w", encoding="utf-8") as fh:
        json.dump(rows_sorted, fh, indent=2, ensure_ascii=False)

    topic_counts = Counter(str(row["primary_topic"] or "Unassigned") for row in rows)
    multi_counts: Counter[str] = Counter()
    local_counts: Counter[str] = Counter()
    for row in rows:
        if not row["topics"] and row["has_local_pdf"]:
            local_counts["Unassigned"] += 1
        for topic in row["topics"]:
            multi_counts[str(topic)] += 1
            if topic and row["has_local_pdf"]:
                local_counts[str(topic)] += 1

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_sorted:
        grouped[str(row["primary_topic"] or "Unassigned")].append(row)

    with (OUTPUT_DIR / "README.md").open("w", encoding="utf-8") as fh:
        fh.write("# Papers Topic Organization\n\n")
        fh.write("This is a topic organization of the local Papers library using the user-requested category vocabulary. ")
        fh.write("Papers may appear in multiple categories.\n\n")
        fh.write(f"- Source database: `{db_path}`\n")
        fh.write(f"- Active library items analyzed: {len(rows)}\n")
        fh.write(f"- Items assigned to at least one category: {sum(1 for row in rows if row['topics'])}\n")
        fh.write(f"- Items without a category match: {sum(1 for row in rows if not row['topics'])}\n")
        fh.write(f"- Active items with local PDFs linked: {sum(1 for row in rows if row['has_local_pdf'])}\n")
        fh.write(f"- Topic folders with PDF symlinks: `topics/`\n")
        fh.write(f"- Machine-readable table: `papers_by_topic.csv`\n")
        fh.write(f"- Full metadata: `papers_by_topic.json`\n\n")
        fh.write("## Topic Counts\n\n")
        fh.write("| Primary topic | Items | Local PDFs | Multi-label mentions |\n")
        fh.write("|---|---:|---:|---:|\n")
        for topic, count in topic_counts.most_common():
            fh.write(f"| {topic} | {count} | {local_counts[topic]} | {multi_counts[topic]} |\n")
        fh.write("\n## Papers By Primary Topic\n\n")
        for topic, topic_rows in grouped.items():
            fh.write(f"### {topic} ({len(topic_rows)})\n\n")
            for row in topic_rows:
                year = row["year"] or "n.d."
                authors = row["authors"]
                author = first_author(authors) if isinstance(authors, list) else "Unknown"
                local = "PDF" if row["has_local_pdf"] else "metadata only"
                fh.write(f"- {year} - {author}: {row['title']} ({local})\n")
            fh.write("\n")

    make_symlinks(rows)


def main() -> None:
    db_path = find_papers_db()
    path_map = load_path_map()
    conn = sqlite3.connect(sqlite_uri(db_path), uri=True)
    try:
        _list_names, item_lists = load_lists(conn)
        rows = collect_items(conn, item_lists, path_map)
    finally:
        conn.close()
    write_outputs(rows, db_path)
    print(f"Wrote {OUTPUT_DIR}/README.md")
    print(f"Analyzed {len(rows)} active Papers items")
    print(f"Linked {sum(1 for row in rows if row['has_local_pdf'])} items with local PDFs")


if __name__ == "__main__":
    main()
