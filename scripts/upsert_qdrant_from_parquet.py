from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.postgres_client import PostgresClient
from db.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from db.sparse_encoder import encode_sparse, sparse_doc_len


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest precomputed real-estate embedding export into local Qdrant and PostgreSQL."
    )
    parser.add_argument(
        "--points-path",
        default="bds_embeddings/bds_qdrant_points.parquet",
        help="Path to bds_qdrant_points.parquet exported by the notebook.",
    )
    parser.add_argument(
        "--manifest-path",
        default="bds_embeddings/bds_embedding_manifest.json",
        help="Optional manifest path used to read embedding dimension.",
    )
    parser.add_argument(
        "--source-records-path",
        default="bds_embeddings/bds_source_records.jsonl",
        help="Path to bds_source_records.jsonl exported by the notebook.",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Do not upsert vectors into Qdrant.",
    )
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Do not upsert source records into PostgreSQL.",
    )
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "").strip())
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY", "").strip())
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost").strip())
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--pg-host", default=os.getenv("POSTGRES_HOST", "localhost").strip())
    parser.add_argument("--pg-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", "postgres").strip())
    parser.add_argument("--pg-password", default=os.getenv("POSTGRES_PASSWORD", "postgres"))
    parser.add_argument("--pg-db", default=os.getenv("POSTGRES_DB", "real_estate").strip())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--collection-prefix", default="")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate target collections before upload.",
    )
    parser.add_argument(
        "--reset-postgres",
        action="store_true",
        help="Truncate PostgreSQL source tables before importing source records.",
    )
    return parser.parse_args()


def collection_name(logical_name: str, prefix: str) -> str:
    return f"{prefix}{logical_name}" if prefix else logical_name


def read_vector_size(df: pd.DataFrame, manifest_path: Path) -> int:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        size = int(manifest.get("embedding_dim") or 0)
        if size:
            return size
    if len(df) == 0:
        raise ValueError("Cannot infer vector size from an empty parquet file.")
    return len(df.iloc[0].vector)


def iter_source_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            record = json.loads(item["record_json"])
            record["id"] = item["source_record_id"]
            yield line_no, item["collection"], record


def upsert_postgres_records(args: argparse.Namespace) -> dict[str, int]:
    records_path = Path(args.source_records_path)
    if not records_path.exists():
        raise FileNotFoundError(records_path)

    pg = PostgresClient(
        host=args.pg_host,
        port=args.pg_port,
        user=args.pg_user,
        password=args.pg_password,
        database=args.pg_db,
    )
    if args.reset_postgres:
        reset_postgres_source_tables(pg)

    counts: dict[str, int] = {}

    total = sum(1 for line in records_path.open("r", encoding="utf-8") if line.strip())
    for _, collection, record in tqdm(
        iter_source_records(records_path),
        total=total,
        desc="upsert postgres",
        unit="record",
        dynamic_ncols=True,
    ):
        remove_stale_unique_conflict(pg, collection, record)
        if collection == "listings":
            pg.upsert_listing(record)
        elif collection == "projects":
            pg.upsert_project(record)
        elif collection == "articles":
            pg.upsert_article(record)
        elif collection == "social_neighborhood":
            pg.upsert_social_neighborhood(record)
        else:
            raise ValueError(f"Unsupported source record collection: {collection}")
        counts[collection] = counts.get(collection, 0) + 1

    try:
        pg.refresh_map_pins()
    except Exception:
        pass

    return counts


def reset_postgres_source_tables(pg: PostgresClient) -> None:
    """Remove old source rows so PostgreSQL matches the current embedding export."""
    with pg.get_cursor() as cur:
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS map_pins;")
        cur.execute(
            """
            TRUNCATE TABLE
                social_neighborhood,
                articles,
                listings,
                projects
            RESTART IDENTITY;
            """
        )


def remove_stale_unique_conflict(pg: PostgresClient, collection: str, record: dict) -> None:
    """Keep PostgreSQL primary IDs aligned with Qdrant source_record_id.

    Earlier local ingests may have stored the same source URL under a different
    ID. The retrieval pipeline hydrates by source_record_id, so the current
    deterministic ID from the embedding export must win.
    """
    table_and_key = {
        "listings": ("listings", "url", record.get("url")),
        "projects": ("projects", "url", record.get("url")),
        "articles": ("articles", "url", record.get("url")),
        "social_neighborhood": (
            "social_neighborhood",
            "thread_url",
            record.get("thread_url") or record.get("url"),
        ),
    }.get(collection)
    if not table_and_key:
        return

    table, key_column, key_value = table_and_key
    if not key_value:
        return
    with pg.get_cursor() as cur:
        cur.execute(
            f"DELETE FROM {table} WHERE {key_column} = %s AND id <> %s",
            (key_value, record["id"]),
        )


def upsert_qdrant_vectors(args: argparse.Namespace) -> dict[str, int | None]:
    points_path = Path(args.points_path)
    manifest_path = Path(args.manifest_path)
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    df = pd.read_parquet(points_path)
    vector_size = read_vector_size(df, manifest_path)
    endpoint = args.url or f"http://{args.host}:{args.port}"
    api_key = args.api_key or None

    client = QdrantClient(
        url=endpoint,
        api_key=api_key,
        timeout=120,
        check_compatibility=False,
    )
    existing = {c.name for c in client.get_collections().collections}
    counts: dict[str, int | None] = {}

    print(f"Loaded {len(df):,} vectors from {points_path}")
    print(f"Qdrant endpoint: {endpoint}")
    print(f"Vector size: {vector_size}, distance: DOT")

    if args.recreate:
        for stale_collection in [
            collection_name(name, args.collection_prefix)
            for name in ("listings", "projects", "articles", "social_neighborhood")
        ]:
            if stale_collection in existing:
                print(f"Deleting existing collection {stale_collection}")
                client.delete_collection(collection_name=stale_collection)
                existing.remove(stale_collection)

    for logical_collection in sorted(df["collection"].unique()):
        qdrant_collection = collection_name(logical_collection, args.collection_prefix)
        part = df[df["collection"] == logical_collection].reset_index(drop=True)

        if qdrant_collection not in existing:
            print(f"Creating collection {qdrant_collection}")
            client.create_collection(
                collection_name=qdrant_collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.DOT),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            existing.add(qdrant_collection)

        doc_lengths = [sparse_doc_len(text) for text in part["text"]]
        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

        for start in tqdm(
            range(0, len(part), args.batch_size),
            desc=f"upsert {qdrant_collection}",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch = part.iloc[start : start + args.batch_size]
            points = []
            for row in batch.itertuples(index=False):
                payload = json.loads(row.metadata_json)
                payload["document_text"] = row.text
                payload["logical_collection"] = logical_collection
                payload["source_record_id"] = row.source_record_id
                points.append(
                    PointStruct(
                        id=row.point_id,
                        vector={
                            DENSE_VECTOR_NAME: list(row.vector),
                            SPARSE_VECTOR_NAME: encode_sparse(
                                row.text,
                                avg_doc_len=avg_doc_len,
                            ),
                        },
                        payload=payload,
                    )
                )
            client.upsert(collection_name=qdrant_collection, points=points, wait=True)

        counts[qdrant_collection] = client.get_collection(
            collection_name=qdrant_collection
        ).points_count

    return counts


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()

    results = {}

    if not args.skip_postgres:
        print("Starting PostgreSQL upsert")
        results["postgres"] = upsert_postgres_records(args)
        print("PostgreSQL upsert done")
        print(json.dumps(results["postgres"], ensure_ascii=False, indent=2))

    if not args.skip_qdrant:
        print("Starting Qdrant upsert")
        results["qdrant"] = upsert_qdrant_vectors(args)
        print("Qdrant upsert done")
        print(json.dumps(results["qdrant"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
