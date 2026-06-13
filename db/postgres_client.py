import logging
import json
import re
import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values

from db.normalizer import (
    extract_location_parts,
    normalize_label,
    parse_area,
    parse_float_field,
    parse_int_field,
    parse_price,
    parse_price_per_m2_vnd,
    parse_price_vnd,
)

log = logging.getLogger("bds_database.postgres")

class PostgresClient:
    """Manages transactional relational data tables inside PostgreSQL 16."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        database: str = "real_estate"
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": database
        }
        self.conn = None
        self.connect()
        self.init_schemas()

    def connect(self):
        """Establish connection with the PostgreSQL server."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.conn.autocommit = True
            log.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            log.error(f"Failed to connect to PostgreSQL: {e}")
            raise e

    def get_cursor(self):
        """Get an active database cursor, reconnecting if needed."""
        try:
            if self.conn is None or self.conn.closed:
                self.connect()
            return self.conn.cursor()
        except Exception:
            self.connect()
            return self.conn.cursor()

    def init_schemas(self):
        """Ensure all relational and document table schemas are created and indexed."""
        postgis_available = False
        with self.get_cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                postgis_available = True
            except Exception as e:
                log.warning(
                    "PostGIS extension is unavailable. Geo columns will be plain lat/lng only: %s",
                    e,
                )

        queries = [
            # 0. Canonical places, projects, and landmarks for geo resolution
            """
            CREATE TABLE IF NOT EXISTS locations (
                id VARCHAR(64) PRIMARY KEY,
                name TEXT NOT NULL,
                location_type VARCHAR(64) NOT NULL,
                province VARCHAR(128),
                district VARCHAR(128),
                ward VARCHAR(128),
                address TEXT,
                aliases TEXT[] DEFAULT '{}',
                google_place_id TEXT,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                source VARCHAR(64),
                confidence DOUBLE PRECISION,
                raw_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 1. Listings Table
            """
            CREATE TABLE IF NOT EXISTS listings (
                id VARCHAR(64) PRIMARY KEY,
                loai_hinh VARCHAR(32),
                loai_nha_dat VARCHAR(64),
                province VARCHAR(128),
                district VARCHAR(128),
                ward VARCHAR(128),
                khu_vuc TEXT,
                dia_chi TEXT,
                price_vnd BIGINT,
                gia_trieu DOUBLE PRECISION,
                price_per_m2_vnd BIGINT,
                gia_per_m2 VARCHAR(64),
                gia_raw VARCHAR(64),
                dien_tich_m2 DOUBLE PRECISION,
                so_phong_ngu INTEGER,
                so_phong_tam INTEGER,
                huong_nha VARCHAR(32),
                huong_ban_cong VARCHAR(32),
                phap_ly VARCHAR(128),
                noi_that VARCHAR(128),
                tieu_de TEXT,
                mo_ta TEXT,
                mo_ta_chi_tiet TEXT,
                du_an VARCHAR(256),
                so_tang INTEGER,
                mat_tien DOUBLE PRECISION,
                duong_vao DOUBLE PRECISION,
                chieu_dai DOUBLE PRECISION,
                chieu_rong DOUBLE PRECISION,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                thumbnail_url TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                expires_at DATE,
                price_change_pct DOUBLE PRECISION,
                posted_at DATE,
                project_id VARCHAR(64),
                url TEXT UNIQUE,
                hinh_anh TEXT[],
                ngay_dang VARCHAR(64),
                nguoi_dang VARCHAR(128),
                raw_json JSONB,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 2. Projects Table
            """
            CREATE TABLE IF NOT EXISTS projects (
                id VARCHAR(64) PRIMARY KEY,
                ten_du_an VARCHAR(256),
                loai_du_an VARCHAR(128),
                chu_dau_tu VARCHAR(256),
                province VARCHAR(128),
                district VARCHAR(128),
                ward VARCHAR(128),
                khu_vuc TEXT,
                dia_chi TEXT,
                quy_mo TEXT,
                dien_tich_m2 DOUBLE PRECISION,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                gia VARCHAR(128),
                gia_tu_vnd BIGINT,
                gia_den_vnd BIGINT,
                trang_thai VARCHAR(128),
                phap_ly VARCHAR(128),
                so_toa INTEGER,
                so_can_ho INTEGER,
                nam_ban_giao INTEGER,
                nam_khoi_cong INTEGER,
                mat_do_xay_dung VARCHAR(64),
                mo_ta_chi_tiet TEXT,
                tien_ich TEXT[],
                thumbnail_url TEXT,
                url TEXT UNIQUE,
                hinh_anh TEXT[],
                raw_json JSONB,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 3. POI cache for Google Maps / Places and manually curated landmarks
            """
            CREATE TABLE IF NOT EXISTS pois (
                id VARCHAR(64) PRIMARY KEY,
                place_id TEXT UNIQUE,
                name TEXT NOT NULL,
                category VARCHAR(64) NOT NULL,
                address TEXT,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                rating DOUBLE PRECISION,
                review_count INTEGER,
                source VARCHAR(64),
                raw_json JSONB,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 4. Cached nearby measurements for property/project ranking
            """
            CREATE TABLE IF NOT EXISTS entity_poi_distances (
                entity_type VARCHAR(32) NOT NULL,
                entity_id VARCHAR(64) NOT NULL,
                poi_id VARCHAR(64) NOT NULL,
                distance_m DOUBLE PRECISION,
                travel_time_s INTEGER,
                travel_mode VARCHAR(32) DEFAULT 'straight_line',
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_type, entity_id, poi_id, travel_mode)
            );
            """,
            # 3. Articles Table
            """
            CREATE TABLE IF NOT EXISTS articles (
                id VARCHAR(64) PRIMARY KEY,
                tieu_de TEXT,
                mo_ta TEXT,
                mo_ta_chi_tiet TEXT,
                url TEXT,
                source_type VARCHAR(32),
                danh_muc VARCHAR(128),
                ngay_dang VARCHAR(64),
                raw_json JSONB,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 4. Social Neighborhood Table
            """
            CREATE TABLE IF NOT EXISTS social_neighborhood (
                id VARCHAR(64) PRIMARY KEY,
                source_type VARCHAR(32),
                keyword VARCHAR(256),
                linked_location_id VARCHAR(64),
                linked_project_id VARCHAR(64),
                video_id VARCHAR(64),
                thread_url TEXT,
                stats_views BIGINT,
                stats_likes BIGINT,
                reactions INTEGER,
                relevance_score DOUBLE PRECISION,
                sentiment_score DOUBLE PRECISION,
                topic_tags TEXT[] DEFAULT '{}',
                published_at TIMESTAMP,
                title TEXT,
                text_content TEXT,
                comments_json JSONB,
                raw_json JSONB,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 6. Precomputed market-report facts by period/location/type.
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id VARCHAR(128) PRIMARY KEY,
                period DATE NOT NULL,
                province VARCHAR(128),
                district VARCHAR(128),
                ward VARCHAR(128),
                property_type VARCHAR(128),
                listing_type VARCHAR(32),
                listing_count INTEGER NOT NULL DEFAULT 0,
                median_price_vnd BIGINT,
                avg_price_vnd BIGINT,
                median_price_per_m2_vnd BIGINT,
                avg_area_m2 DOUBLE PRECISION,
                min_price_vnd BIGINT,
                max_price_vnd BIGINT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]
        
        with self.get_cursor() as cur:
            for q in queries:
                cur.execute(q)
            self._ensure_columns(cur)
            self._ensure_indexes(cur, postgis_available)
        log.info("PostgreSQL table schemas verified/initialized successfully.")

    def _ensure_columns(self, cur):
        """Add columns for users migrating from the earlier schema."""
        column_queries = [
            # listings — original
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS province VARCHAR(128);",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS district VARCHAR(128);",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS ward VARCHAR(128);",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS price_vnd BIGINT;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS price_per_m2_vnd BIGINT;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS posted_at DATE;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS project_id VARCHAR(64);",
            # listings — new
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS gia_raw VARCHAR(64);",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS expires_at DATE;",
            "ALTER TABLE listings ADD COLUMN IF NOT EXISTS price_change_pct DOUBLE PRECISION;",
            # projects — original
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS province VARCHAR(128);",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS district VARCHAR(128);",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS ward VARCHAR(128);",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;",
            # projects — new
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS gia_tu_vnd BIGINT;",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS gia_den_vnd BIGINT;",
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;",
            # social
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS linked_location_id VARCHAR(64);",
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS linked_project_id VARCHAR(64);",
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS relevance_score DOUBLE PRECISION;",
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE PRECISION;",
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS topic_tags TEXT[] DEFAULT '{}';",
            "ALTER TABLE social_neighborhood ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;",
            # articles — drop stale UNIQUE constraint on url (chunks share a url, only id is unique)
            "ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_url_key;",
        ]
        for q in column_queries:
            cur.execute(q)

    def _ensure_indexes(self, cur, postgis_available: bool):
        """Create indexes used by filters, market reports, and geo ranking."""
        if postgis_available:
            geo_queries = [
                "ALTER TABLE locations ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);",
                "ALTER TABLE listings ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);",
                "ALTER TABLE projects ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);",
                "ALTER TABLE pois ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);",
                """
                UPDATE locations
                SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                WHERE geom IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
                """,
                """
                UPDATE listings
                SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                WHERE geom IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
                """,
                """
                UPDATE projects
                SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                WHERE geom IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
                """,
                """
                UPDATE pois
                SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                WHERE geom IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
                """,
                "CREATE INDEX IF NOT EXISTS idx_locations_geom ON locations USING GIST (geom);",
                "CREATE INDEX IF NOT EXISTS idx_listings_geom ON listings USING GIST (geom);",
                "CREATE INDEX IF NOT EXISTS idx_projects_geom ON projects USING GIST (geom);",
                "CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST (geom);",
            ]
            for q in geo_queries:
                cur.execute(q)

        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_listings_filters ON listings (loai_hinh, loai_nha_dat, province, district);",
            "CREATE INDEX IF NOT EXISTS idx_listings_price ON listings (price_vnd);",
            "CREATE INDEX IF NOT EXISTS idx_listings_area ON listings (dien_tich_m2);",
            "CREATE INDEX IF NOT EXISTS idx_listings_price_m2 ON listings (price_per_m2_vnd);",
            "CREATE INDEX IF NOT EXISTS idx_listings_posted_at ON listings (posted_at);",
            "CREATE INDEX IF NOT EXISTS idx_projects_location ON projects (province, district, loai_du_an);",
            "CREATE INDEX IF NOT EXISTS idx_pois_category ON pois (category);",
            "CREATE INDEX IF NOT EXISTS idx_social_links ON social_neighborhood (linked_location_id, linked_project_id);",
            "CREATE INDEX IF NOT EXISTS idx_social_relevance ON social_neighborhood (relevance_score);",
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_dims ON market_snapshots (period, province, district, property_type, listing_type);",
        ]
        for q in index_queries:
            cur.execute(q)

    def reset_tables(self):
        """Drop and recreate all relational tables."""
        tables = [
            "entity_poi_distances",
            "market_snapshots",
            "social_neighborhood",
            "pois",
            "articles",
            "listings",
            "projects",
            "locations",
        ]
        with self.get_cursor() as cur:
            for t in tables:
                cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE;")
        log.info("All PostgreSQL tables dropped.")
        self.init_schemas()

    @staticmethod
    def _parse_listing_date(raw: Optional[str]) -> Optional[date]:
        """Best-effort parser for crawled relative date strings."""
        if not raw:
            return None
        text = str(raw).strip().lower()
        today = date.today()
        if "hôm nay" in text:
            return today
        if "hôm qua" in text:
            return today - timedelta(days=1)
        m = re.search(r"(\d+)\s+ngày", text)
        if m:
            return today - timedelta(days=int(m.group(1)))
        m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
        if m:
            day, month, year = map(int, m.groups())
            return date(year, month, day)
        return None

    @staticmethod
    def _as_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return parse_float_field(str(val))

    @staticmethod
    def _as_int(val: Any) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(val)
        except Exception:
            parsed = parse_int_field(str(val))
            return parsed

    @staticmethod
    def _text_array(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(v) for v in val if v is not None]
        if val:
            return [str(val)]
        return []

    def upsert_listing(self, item: Dict[str, Any]):
        """Upsert property listing record into PostgreSQL listings table."""
        q = """
            INSERT INTO listings (
                id, loai_hinh, loai_nha_dat, province, district, ward, khu_vuc, dia_chi,
                price_vnd, gia_trieu, price_per_m2_vnd, gia_per_m2, gia_raw, dien_tich_m2,
                so_phong_ngu, so_phong_tam, huong_nha, huong_ban_cong, phap_ly, noi_that, tieu_de,
                mo_ta, mo_ta_chi_tiet, du_an, so_tang, mat_tien, duong_vao, chieu_dai, chieu_rong,
                latitude, longitude, thumbnail_url, is_active, expires_at,
                posted_at, project_id, url, hinh_anh, ngay_dang, nguoi_dang, raw_json
            ) VALUES (
                %(id)s, %(loai_hinh)s, %(loai_nha_dat)s, %(province)s, %(district)s, %(ward)s, %(khu_vuc)s, %(dia_chi)s,
                %(price_vnd)s, %(gia_trieu)s, %(price_per_m2_vnd)s, %(gia_per_m2)s, %(gia_raw)s, %(dien_tich_m2)s,
                %(so_phong_ngu)s, %(so_phong_tam)s, %(huong_nha)s, %(huong_ban_cong)s, %(phap_ly)s, %(noi_that)s, %(tieu_de)s,
                %(mo_ta)s, %(mo_ta_chi_tiet)s, %(du_an)s, %(so_tang)s, %(mat_tien)s, %(duong_vao)s, %(chieu_dai)s, %(chieu_rong)s,
                %(latitude)s, %(longitude)s, %(thumbnail_url)s, %(is_active)s, %(expires_at)s,
                %(posted_at)s, %(project_id)s, %(url)s, %(hinh_anh)s, %(ngay_dang)s, %(nguoi_dang)s, %(raw_json)s
            ) ON CONFLICT (id) DO UPDATE SET
                loai_hinh = EXCLUDED.loai_hinh,
                loai_nha_dat = EXCLUDED.loai_nha_dat,
                province = EXCLUDED.province,
                district = EXCLUDED.district,
                ward = EXCLUDED.ward,
                khu_vuc = EXCLUDED.khu_vuc,
                dia_chi = EXCLUDED.dia_chi,
                price_vnd = EXCLUDED.price_vnd,
                gia_trieu = EXCLUDED.gia_trieu,
                price_per_m2_vnd = EXCLUDED.price_per_m2_vnd,
                gia_per_m2 = EXCLUDED.gia_per_m2,
                dien_tich_m2 = EXCLUDED.dien_tich_m2,
                so_phong_ngu = EXCLUDED.so_phong_ngu,
                so_phong_tam = EXCLUDED.so_phong_tam,
                huong_nha = EXCLUDED.huong_nha,
                huong_ban_cong = EXCLUDED.huong_ban_cong,
                phap_ly = EXCLUDED.phap_ly,
                noi_that = EXCLUDED.noi_that,
                tieu_de = EXCLUDED.tieu_de,
                mo_ta = EXCLUDED.mo_ta,
                mo_ta_chi_tiet = EXCLUDED.mo_ta_chi_tiet,
                du_an = EXCLUDED.du_an,
                so_tang = EXCLUDED.so_tang,
                mat_tien = EXCLUDED.mat_tien,
                duong_vao = EXCLUDED.duong_vao,
                chieu_dai = EXCLUDED.chieu_dai,
                chieu_rong = EXCLUDED.chieu_rong,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                thumbnail_url = EXCLUDED.thumbnail_url,
                is_active = EXCLUDED.is_active,
                expires_at = EXCLUDED.expires_at,
                posted_at = EXCLUDED.posted_at,
                project_id = EXCLUDED.project_id,
                url = EXCLUDED.url,
                hinh_anh = EXCLUDED.hinh_anh,
                ngay_dang = EXCLUDED.ngay_dang,
                nguoi_dang = EXCLUDED.nguoi_dang,
                raw_json = EXCLUDED.raw_json,
                crawled_at = CURRENT_TIMESTAMP;
        """
        loc = extract_location_parts(item.get("dia_chi") or item.get("khu_vuc"))
        price_vnd = item.get("price_vnd") or parse_price_vnd(item.get("gia"))
        price_million = item.get("gia_trieu")
        if price_million is None and price_vnd is not None:
            price_million = price_vnd / 1_000_000
        area_m2 = item.get("dien_tich_m2") or parse_area(item.get("dien_tich"))
        lat = self._as_float(item.get("latitude") or item.get("lat"))
        lng = self._as_float(item.get("longitude") or item.get("lng") or item.get("lon"))
        # thumbnail: first image URL
        images = self._text_array(item.get("hinh_anh"))
        thumbnail = images[0] if images else item.get("thumbnail_url")
        # expires_at: posted_at + 90 days
        posted = self._parse_listing_date(item.get("ngay_dang"))
        from datetime import timedelta
        expires = (posted + timedelta(days=90)) if posted else None
        params = {
            "id": item["id"],
            "loai_hinh": item.get("loai_hinh"),
            "loai_nha_dat": normalize_label(item.get("loai_nha_dat")),
            "province": loc["province"],
            "district": loc["district"],
            "ward": loc["ward"],
            "khu_vuc": item.get("khu_vuc"),
            "dia_chi": item.get("dia_chi"),
            "price_vnd": price_vnd,
            "gia_trieu": self._as_float(price_million),
            "price_per_m2_vnd": item.get("price_per_m2_vnd") or parse_price_per_m2_vnd(item.get("gia_per_m2")),
            "gia_per_m2": item.get("gia_per_m2"),
            "gia_raw": item.get("gia") or item.get("gia_raw"),
            "dien_tich_m2": self._as_float(area_m2),
            "so_phong_ngu": self._as_int(item.get("so_phong_ngu")),
            "so_phong_tam": self._as_int(item.get("so_phong_tam")),
            "huong_nha": normalize_label(item.get("huong_nha")),
            "huong_ban_cong": normalize_label(item.get("huong_ban_cong")),
            "phap_ly": normalize_label(item.get("phap_ly")),
            "noi_that": normalize_label(item.get("noi_that")),
            "tieu_de": item.get("tieu_de"),
            "mo_ta": item.get("mo_ta"),
            "mo_ta_chi_tiet": item.get("mo_ta_chi_tiet"),
            "du_an": item.get("du_an"),
            "so_tang": self._as_int(item.get("so_tang")),
            "mat_tien": self._as_float(item.get("mat_tien")),
            "duong_vao": self._as_float(item.get("duong_vao")),
            "chieu_dai": self._as_float(item.get("chieu_dai")),
            "chieu_rong": self._as_float(item.get("chieu_rong")),
            "latitude": lat,
            "longitude": lng,
            "thumbnail_url": thumbnail,
            "is_active": item.get("is_active", True),
            "expires_at": expires,
            "posted_at": posted,
            "project_id": item.get("project_id"),
            "url": item.get("url"),
            "hinh_anh": images,
            "ngay_dang": item.get("ngay_dang"),
            "nguoi_dang": item.get("nguoi_dang"),
            "raw_json": json.dumps(item)
        }
        
        with self.get_cursor() as cur:
            cur.execute(q, params)
            if lat is not None and lng is not None:
                try:
                    cur.execute(
                        """
                        UPDATE listings
                        SET geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                        WHERE id = %s;
                        """,
                        (lng, lat, item["id"]),
                    )
                except Exception:
                    pass

    def upsert_project(self, item: Dict[str, Any]):
        """Upsert project specifications into projects table."""
        q = """
            INSERT INTO projects (
                id, ten_du_an, loai_du_an, chu_dau_tu, province, district, ward, khu_vuc, dia_chi,
                quy_mo, dien_tich_m2, latitude, longitude,
                gia, gia_tu_vnd, gia_den_vnd,
                trang_thai, phap_ly, so_toa, so_can_ho, nam_ban_giao, nam_khoi_cong,
                mat_do_xay_dung, mo_ta_chi_tiet, tien_ich, thumbnail_url, url, hinh_anh, raw_json
            ) VALUES (
                %(id)s, %(ten_du_an)s, %(loai_du_an)s, %(chu_dau_tu)s, %(province)s, %(district)s, %(ward)s, %(khu_vuc)s, %(dia_chi)s,
                %(quy_mo)s, %(dien_tich_m2)s, %(latitude)s, %(longitude)s,
                %(gia)s, %(gia_tu_vnd)s, %(gia_den_vnd)s,
                %(trang_thai)s, %(phap_ly)s, %(so_toa)s, %(so_can_ho)s, %(nam_ban_giao)s, %(nam_khoi_cong)s,
                %(mat_do_xay_dung)s, %(mo_ta_chi_tiet)s, %(tien_ich)s, %(thumbnail_url)s, %(url)s, %(hinh_anh)s, %(raw_json)s
            ) ON CONFLICT (id) DO UPDATE SET
                ten_du_an = EXCLUDED.ten_du_an,
                loai_du_an = EXCLUDED.loai_du_an,
                chu_dau_tu = EXCLUDED.chu_dau_tu,
                province = EXCLUDED.province,
                district = EXCLUDED.district,
                ward = EXCLUDED.ward,
                khu_vuc = EXCLUDED.khu_vuc,
                dia_chi = EXCLUDED.dia_chi,
                quy_mo = EXCLUDED.quy_mo,
                dien_tich_m2 = EXCLUDED.dien_tich_m2,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                gia = EXCLUDED.gia,
                gia_tu_vnd = EXCLUDED.gia_tu_vnd,
                gia_den_vnd = EXCLUDED.gia_den_vnd,
                trang_thai = EXCLUDED.trang_thai,
                phap_ly = EXCLUDED.phap_ly,
                so_toa = EXCLUDED.so_toa,
                so_can_ho = EXCLUDED.so_can_ho,
                nam_ban_giao = EXCLUDED.nam_ban_giao,
                nam_khoi_cong = EXCLUDED.nam_khoi_cong,
                mat_do_xay_dung = EXCLUDED.mat_do_xay_dung,
                mo_ta_chi_tiet = EXCLUDED.mo_ta_chi_tiet,
                tien_ich = EXCLUDED.tien_ich,
                thumbnail_url = EXCLUDED.thumbnail_url,
                url = EXCLUDED.url,
                hinh_anh = EXCLUDED.hinh_anh,
                raw_json = EXCLUDED.raw_json,
                crawled_at = CURRENT_TIMESTAMP;
        """
        # Convert dien_tich to m2 if it represents hectares
        dien_tich = item.get("dien_tich")
        dien_tich_val = self._as_float(dien_tich)
        if dien_tich and isinstance(dien_tich, str) and "ha" in dien_tich.lower() and dien_tich_val:
            dien_tich_val = dien_tich_val * 10000.0 # 1 ha = 10000 m2
        loc = extract_location_parts(item.get("dia_chi") or item.get("khu_vuc"))
        lat = self._as_float(item.get("latitude") or item.get("lat"))
        lng = self._as_float(item.get("longitude") or item.get("lng") or item.get("lon"))

        images = self._text_array(item.get("hinh_anh"))
        thumbnail = images[0] if images else item.get("thumbnail_url")
        # Parse price range from raw string e.g. "từ 2 tỷ – 5 tỷ"
        gia_raw = item.get("gia", "")
        gia_tu_vnd = item.get("gia_tu_vnd") or parse_price_vnd(gia_raw)
        gia_den_vnd = item.get("gia_den_vnd")
        params = {
            "id": item["id"],
            "ten_du_an": item.get("ten_du_an"),
            "loai_du_an": normalize_label(item.get("loai_du_an")),
            "chu_dau_tu": normalize_label(item.get("chu_dau_tu")),
            "province": loc["province"],
            "district": loc["district"],
            "ward": loc["ward"],
            "khu_vuc": item.get("khu_vuc"),
            "dia_chi": item.get("dia_chi"),
            "quy_mo": item.get("quy_mo"),
            "dien_tich_m2": dien_tich_val,
            "latitude": lat,
            "longitude": lng,
            "gia": gia_raw,
            "gia_tu_vnd": gia_tu_vnd,
            "gia_den_vnd": gia_den_vnd,
            "trang_thai": normalize_label(item.get("trang_thai")),
            "phap_ly": normalize_label(item.get("phap_ly")),
            "so_toa": self._as_int(item.get("so_toa")),
            "so_can_ho": self._as_int(item.get("so_can_ho")),
            "nam_ban_giao": self._as_int(item.get("nam_ban_giao")),
            "nam_khoi_cong": self._as_int(item.get("nam_khoi_cong")),
            "mat_do_xay_dung": normalize_label(item.get("mat_do_xay_dung")),
            "mo_ta_chi_tiet": item.get("mo_ta_chi_tiet"),
            "tien_ich": self._text_array(item.get("tien_ich")),
            "thumbnail_url": thumbnail,
            "url": item.get("url"),
            "hinh_anh": images,
            "raw_json": json.dumps(item)
        }
        
        with self.get_cursor() as cur:
            cur.execute(q, params)
            if lat is not None and lng is not None:
                try:
                    cur.execute(
                        """
                        UPDATE projects
                        SET geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                        WHERE id = %s;
                        """,
                        (lng, lat, item["id"]),
                    )
                except Exception:
                    pass

    def refresh_map_pins(self):
        """
        Create (or refresh) the map_pins materialized view.

        Combines listings + projects + pois into a single geo-queryable table
        for the frontend map UI. Call this after bulk ingestion.

        Requires PostGIS. Falls back silently if not available.
        """
        ddl = """
            DROP MATERIALIZED VIEW IF EXISTS map_pins;
            CREATE MATERIALIZED VIEW map_pins AS
            SELECT
                id,
                'listing'   AS pin_type,
                tieu_de     AS label,
                gia_raw     AS price_label,
                loai_nha_dat AS category,
                loai_hinh   AS listing_type,
                latitude,
                longitude,
                thumbnail_url,
                url
            FROM listings
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND is_active = TRUE

            UNION ALL

            SELECT
                id,
                'project'   AS pin_type,
                ten_du_an   AS label,
                gia         AS price_label,
                loai_du_an  AS category,
                NULL        AS listing_type,
                latitude,
                longitude,
                thumbnail_url,
                url
            FROM projects
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL

            UNION ALL

            SELECT
                id,
                'poi'       AS pin_type,
                name        AS label,
                NULL        AS price_label,
                category    AS category,
                NULL        AS listing_type,
                latitude,
                longitude,
                NULL        AS thumbnail_url,
                NULL        AS url
            FROM pois
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

            CREATE INDEX IF NOT EXISTS map_pins_lat_lng ON map_pins (latitude, longitude);
        """
        try:
            with self.get_cursor() as cur:
                cur.execute(ddl)
            log.info("map_pins materialized view refreshed")
        except Exception as e:
            log.warning(f"refresh_map_pins failed (PostGIS may be unavailable): {e}")

    def fetch_map_pins(
        self,
        lat_min: float, lat_max: float,
        lng_min: float, lng_max: float,
        pin_types: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Fetch map pins within a bounding box for the frontend map view.

        Args:
            lat_min, lat_max, lng_min, lng_max: bounding box coordinates
            pin_types: ['listing', 'project', 'poi'] or None for all
            limit: max pins to return (default 500)

        Returns:
            List of dicts with: id, pin_type, label, price_label,
            category, listing_type, latitude, longitude, thumbnail_url, url
        """
        conditions = [
            "latitude BETWEEN %(lat_min)s AND %(lat_max)s",
            "longitude BETWEEN %(lng_min)s AND %(lng_max)s",
        ]
        params: Dict[str, Any] = {
            "lat_min": lat_min, "lat_max": lat_max,
            "lng_min": lng_min, "lng_max": lng_max,
            "limit": limit,
        }
        if pin_types:
            conditions.append("pin_type = ANY(%(pin_types)s)")
            params["pin_types"] = pin_types

        where = " AND ".join(conditions)
        q = f"""
            SELECT id, pin_type, label, price_label, category,
                   listing_type, latitude, longitude, thumbnail_url, url
            FROM map_pins
            WHERE {where}
            LIMIT %(limit)s
        """
        rows = []
        try:
            with self.get_cursor() as cur:
                cur.execute(q, params)
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(cols, row)))
        except Exception as e:
            log.warning(f"fetch_map_pins failed (map_pins view may not exist yet): {e}")
        return rows

    def upsert_article(self, item: Dict[str, Any]):
        """Upsert parsed news or wiki article."""
        q = """
            INSERT INTO articles (
                id, tieu_de, mo_ta, mo_ta_chi_tiet, url, source_type, danh_muc, ngay_dang, raw_json
            ) VALUES (
                %(id)s, %(tieu_de)s, %(mo_ta)s, %(mo_ta_chi_tiet)s, %(url)s, %(source_type)s, %(danh_muc)s, %(ngay_dang)s, %(raw_json)s
            ) ON CONFLICT (id) DO UPDATE SET
                tieu_de = EXCLUDED.tieu_de,
                mo_ta = EXCLUDED.mo_ta,
                mo_ta_chi_tiet = EXCLUDED.mo_ta_chi_tiet,
                url = EXCLUDED.url,
                source_type = EXCLUDED.source_type,
                danh_muc = EXCLUDED.danh_muc,
                ngay_dang = EXCLUDED.ngay_dang,
                raw_json = EXCLUDED.raw_json,
                crawled_at = CURRENT_TIMESTAMP;
        """
        params = {
            "id": item["id"],
            "tieu_de": item.get("tieu_de") or item.get("title"),
            "mo_ta": item.get("mo_ta") or item.get("description"),
            "mo_ta_chi_tiet": item.get("mo_ta_chi_tiet") or item.get("content"),
            "url": item.get("url"),
            "source_type": item.get("source_type"),
            "danh_muc": item.get("danh_muc") or item.get("category"),
            "ngay_dang": item.get("ngay_dang") or item.get("date"),
            "raw_json": json.dumps(item)
        }
        
        with self.get_cursor() as cur:
            cur.execute(q, params)

    def upsert_social_neighborhood(self, item: Dict[str, Any]):
        """Upsert qualitative social platform discussions."""
        q = """
            INSERT INTO social_neighborhood (
                id, source_type, keyword, linked_location_id, linked_project_id, video_id, thread_url,
                stats_views, stats_likes, reactions, relevance_score, sentiment_score, topic_tags,
                published_at, title, text_content, comments_json, raw_json
            ) VALUES (
                %(id)s, %(source_type)s, %(keyword)s, %(linked_location_id)s, %(linked_project_id)s, %(video_id)s, %(thread_url)s,
                %(stats_views)s, %(stats_likes)s, %(reactions)s, %(relevance_score)s, %(sentiment_score)s, %(topic_tags)s,
                %(published_at)s, %(title)s, %(text_content)s, %(comments_json)s, %(raw_json)s
            ) ON CONFLICT (id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                keyword = EXCLUDED.keyword,
                linked_location_id = EXCLUDED.linked_location_id,
                linked_project_id = EXCLUDED.linked_project_id,
                video_id = EXCLUDED.video_id,
                thread_url = EXCLUDED.thread_url,
                stats_views = EXCLUDED.stats_views,
                stats_likes = EXCLUDED.stats_likes,
                reactions = EXCLUDED.reactions,
                relevance_score = EXCLUDED.relevance_score,
                sentiment_score = EXCLUDED.sentiment_score,
                topic_tags = EXCLUDED.topic_tags,
                published_at = EXCLUDED.published_at,
                title = EXCLUDED.title,
                text_content = EXCLUDED.text_content,
                comments_json = EXCLUDED.comments_json,
                raw_json = EXCLUDED.raw_json,
                crawled_at = CURRENT_TIMESTAMP;
        """
        stats = item.get("stats", {})
        posts_or_comments = item.get("comments") or item.get("posts") or []
        reactions = item.get("reactions")
        if reactions is None and isinstance(posts_or_comments, list):
            reactions = sum(self._as_int(c.get("like_count") or c.get("reactions_count")) or 0 for c in posts_or_comments)
        text_content = item.get("text_content") or item.get("description") or item.get("transcript_text") or item.get("snippet")
        
        params = {
            "id": item["id"],
            "source_type": item.get("source_type"),
            "keyword": item.get("keyword"),
            "linked_location_id": item.get("linked_location_id"),
            "linked_project_id": item.get("linked_project_id"),
            "video_id": item.get("video_id"),
            "thread_url": item.get("thread_url") or item.get("url"),
            "stats_views": stats.get("views") or stats.get("view_count"),
            "stats_likes": stats.get("likes") or stats.get("like_count"),
            "reactions": reactions,
            "relevance_score": item.get("relevance_score"),
            "sentiment_score": item.get("sentiment_score"),
            "topic_tags": self._text_array(item.get("topic_tags")),
            "published_at": item.get("published_at"),
            "title": item.get("title") or item.get("thread_title"),
            "text_content": text_content,
            "comments_json": json.dumps(posts_or_comments),
            "raw_json": json.dumps(item)
        }
        
        with self.get_cursor() as cur:
            cur.execute(q, params)

    def upsert_poi(self, item: Dict[str, Any]):
        """Upsert OpenStreetMap or other structured POI records."""
        q = """
            INSERT INTO pois (
                id, place_id, name, category, address, latitude, longitude,
                rating, review_count, source, raw_json
            ) VALUES (
                %(id)s, %(place_id)s, %(name)s, %(category)s, %(address)s, %(latitude)s, %(longitude)s,
                %(rating)s, %(review_count)s, %(source)s, %(raw_json)s
            ) ON CONFLICT (id) DO UPDATE SET
                place_id = EXCLUDED.place_id,
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                address = EXCLUDED.address,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                rating = EXCLUDED.rating,
                review_count = EXCLUDED.review_count,
                source = EXCLUDED.source,
                raw_json = EXCLUDED.raw_json,
                fetched_at = CURRENT_TIMESTAMP;
        """
        place_id = item.get("place_id")
        if not place_id:
            osm_type = item.get("osm_type") or "unknown"
            osm_id = item.get("osm_id") or item.get("id") or item.get("name")
            place_id = f"{item.get('source') or 'poi'}:{osm_type}:{osm_id}"

        poi_id = item.get("id") or hashlib.md5(str(place_id).encode("utf-8")).hexdigest()
        lat = self._as_float(item.get("latitude") or item.get("lat"))
        lng = self._as_float(item.get("longitude") or item.get("lng") or item.get("lon"))
        params = {
            "id": poi_id,
            "place_id": place_id,
            "name": item.get("name") or "Unknown POI",
            "category": item.get("category") or "unknown",
            "address": item.get("address") or item.get("formatted_address"),
            "latitude": lat,
            "longitude": lng,
            "rating": self._as_float(item.get("rating")),
            "review_count": self._as_int(item.get("review_count") or item.get("user_ratings_total")),
            "source": item.get("source") or "unknown",
            "raw_json": json.dumps(item),
        }

        with self.get_cursor() as cur:
            cur.execute(q, params)
            if lat is not None and lng is not None:
                try:
                    cur.execute(
                        """
                        UPDATE pois
                        SET geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                        WHERE id = %s;
                        """,
                        (lng, lat, poi_id),
                    )
                except Exception:
                    pass

    def fetch_by_ids(self, table_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve original hydrated JSON payloads by IDs."""
        if not ids: return []
        
        # Guard SQL Injection
        valid_tables = ["listings", "projects", "articles", "social_neighborhood"]
        if table_name not in valid_tables:
            raise ValueError(f"Invalid table: {table_name}")
            
        q = f"SELECT id, raw_json FROM {table_name} WHERE id IN %s"
        results = []
        try:
            with self.get_cursor() as cur:
                cur.execute(q, (tuple(ids),))
                rows = cur.fetchall()
                for row in rows:
                    item_id, payload_str = row
                    if isinstance(payload_str, str):
                        payload = json.loads(payload_str)
                    else:
                        payload = payload_str  # parsed natively by psycopg2 jsonb adapter
                    results.append(payload)
        except Exception as e:
            log.warning(f"Error fetching from table {table_name}: {e}")
        return results

    def fetch_nearby_pois(
        self,
        entity_ids: List[str],
        entity_type: str = "listing",
        categories: Optional[List[str]] = None,
        top_n_per_category: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch pre-cached nearby POIs for a set of listing/project IDs.

        This is a pure Postgres read from entity_poi_distances JOIN pois —
        no external API calls are made at inference time.

        Args:
            entity_ids: List of listing/project IDs to enrich.
            entity_type: 'listing' or 'project'.
            categories: OSM categories to filter (e.g. ['school', 'transit_station']).
                        If None, all categories are returned.
            top_n_per_category: Max POIs per category per entity.

        Returns:
            Dict mapping entity_id -> list of POI dicts with keys:
            name, category, distance_m, address.
        """
        if not entity_ids:
            return {}

        cat_filter = ""
        params: Dict[str, Any] = {
            "entity_type": entity_type,
            "entity_ids": tuple(entity_ids),
        }
        if categories:
            cat_filter = "AND p.category = ANY(%(categories)s)"
            params["categories"] = categories

        q = f"""
            SELECT
                epd.entity_id,
                p.name,
                p.category,
                p.address,
                epd.distance_m,
                p.latitude,
                p.longitude
            FROM entity_poi_distances epd
            JOIN pois p ON p.id = epd.poi_id
            WHERE epd.entity_type = %(entity_type)s
              AND epd.entity_id IN %(entity_ids)s
              {cat_filter}
            ORDER BY epd.entity_id, p.category, epd.distance_m
        """

        result: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in entity_ids}
        # Track per-(entity_id, category) counts for top_n_per_category
        category_counts: Dict[tuple, int] = {}

        try:
            with self.get_cursor() as cur:
                cur.execute(q, params)
                rows = cur.fetchall()
                for entity_id, name, category, address, distance_m, lat, lng in rows:
                    key = (entity_id, category)
                    count = category_counts.get(key, 0)
                    if count >= top_n_per_category:
                        continue
                    category_counts[key] = count + 1
                    result[entity_id].append({
                        "name": name,
                        "category": category,
                        "address": address or "",
                        "distance_m": int(distance_m) if distance_m else None,
                        "lat": lat,
                        "lng": lng,
                    })
        except Exception as e:
            log.warning(f"fetch_nearby_pois failed: {e}")

        return result

    def fetch_market_stats(
        self,
        province: Optional[str] = None,
        district: Optional[str] = None,
        listing_type: Optional[str] = None,   # 'ban' | 'cho_thue' | None (both)
        property_type: Optional[str] = None,
        months: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Fetch pre-computed market statistics from market_snapshots.

        Aggregates median/avg price, price/m², listing count, and trend
        directly from the listings corpus — NOT from news articles.

        Covers:
          - Nhà bán    (loai_hinh = 'ban')
          - Nhà cho thuê (loai_hinh = 'cho_thue')
          - Dự án      (aggregated from listings linked to projects)

        Args:
            province: Filter by tỉnh/thành phố (e.g. "TP Hồ Chí Minh")
            district: Filter by quận/huyện (e.g. "Quận 7")
            listing_type: 'ban', 'cho_thue', or None for all
            property_type: e.g. "Căn hộ chung cư", "Đất", or None for all
            months: How many past months to include (default 12)

        Returns:
            List of stat row dicts with keys:
            period, province, district, property_type, listing_type,
            listing_count, median_price_vnd, avg_price_vnd,
            median_price_per_m2_vnd, avg_area_m2, min_price_vnd, max_price_vnd
        """
        conditions = ["period >= (date_trunc('month', CURRENT_DATE) - INTERVAL '%(months)s months')"]
        params: Dict[str, Any] = {"months": f"{months} months"}

        if province:
            conditions.append("province ILIKE %(province)s")
            params["province"] = f"%{province}%"
        if district:
            conditions.append("district ILIKE %(district)s")
            params["district"] = f"%{district}%"
        if listing_type:
            conditions.append("listing_type = %(listing_type)s")
            params["listing_type"] = listing_type
        if property_type:
            conditions.append("property_type ILIKE %(property_type)s")
            params["property_type"] = f"%{property_type}%"

        where = " AND ".join(conditions)
        q = f"""
            SELECT
                to_char(period, 'YYYY-MM') AS period,
                province,
                district,
                property_type,
                listing_type,
                listing_count,
                median_price_vnd,
                avg_price_vnd,
                median_price_per_m2_vnd,
                avg_area_m2,
                min_price_vnd,
                max_price_vnd
            FROM market_snapshots
            WHERE {where}
            ORDER BY period DESC, listing_count DESC
            LIMIT 200
        """

        rows = []
        try:
            with self.get_cursor() as cur:
                cur.execute(q, params)
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(cols, row)))
        except Exception as e:
            log.warning(f"fetch_market_stats failed: {e}")
        return rows

    def refresh_market_snapshots(self, period: Optional[date] = None):
        """
        Rebuild market report aggregates from normalized listing facts.

        This gives the assistant deterministic numbers for report generation
        instead of asking the LLM to infer statistics from retrieved snippets.
        """
        snapshot_period = period or date.today().replace(day=1)
        q = """
            INSERT INTO market_snapshots (
                id, period, province, district, ward, property_type, listing_type,
                listing_count, median_price_vnd, avg_price_vnd, median_price_per_m2_vnd,
                avg_area_m2, min_price_vnd, max_price_vnd
            )
            SELECT
                md5(
                    %(period)s::text || '|' ||
                    coalesce(province, '') || '|' ||
                    coalesce(district, '') || '|' ||
                    coalesce(ward, '') || '|' ||
                    coalesce(loai_nha_dat, '') || '|' ||
                    coalesce(loai_hinh, '')
                ) AS id,
                %(period)s::date AS period,
                province,
                district,
                ward,
                loai_nha_dat AS property_type,
                loai_hinh AS listing_type,
                count(*)::integer AS listing_count,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY price_vnd)::bigint AS median_price_vnd,
                avg(price_vnd)::bigint AS avg_price_vnd,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_m2_vnd)::bigint AS median_price_per_m2_vnd,
                avg(dien_tich_m2) AS avg_area_m2,
                min(price_vnd)::bigint AS min_price_vnd,
                max(price_vnd)::bigint AS max_price_vnd
            FROM listings
            WHERE price_vnd IS NOT NULL
            GROUP BY province, district, ward, loai_nha_dat, loai_hinh
            ON CONFLICT (id) DO UPDATE SET
                listing_count = EXCLUDED.listing_count,
                median_price_vnd = EXCLUDED.median_price_vnd,
                avg_price_vnd = EXCLUDED.avg_price_vnd,
                median_price_per_m2_vnd = EXCLUDED.median_price_per_m2_vnd,
                avg_area_m2 = EXCLUDED.avg_area_m2,
                min_price_vnd = EXCLUDED.min_price_vnd,
                max_price_vnd = EXCLUDED.max_price_vnd,
                generated_at = CURRENT_TIMESTAMP;
        """
        with self.get_cursor() as cur:
            cur.execute(q, {"period": snapshot_period})

    def close(self):
        """Close connection cleanly."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            log.info("PostgreSQL database connection closed.")
