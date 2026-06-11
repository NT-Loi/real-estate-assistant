# Real Estate Assistant

Crawler and ingestion toolkit for a real-estate RAG assistant. The project collects property listings, projects, market/wiki content, social review evidence, and nearby amenities.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Optional API keys can be placed in `.env`:

```bash
YOUTUBE_API_KEY=your_youtube_key
GOOGLE_MAPS_API_KEY=optional_legacy_google_key
```

`YOUTUBE_API_KEY` enables YouTube comment/transcript crawling. Google Maps is optional legacy functionality; POI/amenity crawling uses OpenStreetMap by default and does not need a paid key.

## Data Crawl Order

Recommended full workflow:

```bash
# 1. Crawl property listings and projects
python run.py --type ban --pages 3
python run.py --type cho-thue --pages 3
python run.py --type du-an --pages 3

# 2. Crawl general market/knowledge content
python run.py --type tin-tuc --pages 3
python run.py --type wiki --pages 3

# 3. Add latitude/longitude to listings and projects
python run.py --type geocode --geocode-limit 0

# 4. Crawl nearby amenities within 2 km for every geocoded listing/project
python run.py --type osm-poi

# 5. Crawl related review/social evidence
python run.py --type youtube --pages 3
python run.py --type tiktok --pages 3
python run.py --type voz --pages 3
```

You can also run the consolidated flow:

```bash
python run.py --type all --pages 3 --geocode-limit 0
```

In `--type all`, review keywords are generated lazily after fresh listings and projects have been crawled and saved. This means YouTube, TikTok, and VOZ searches use the newly crawled `dia_chi` and `ten_du_an` values, not stale keywords loaded at startup.

For long OSM amenity crawls, prefer running `geocode` first, then `osm-poi` separately. The OSM crawler caches results in `data/.osm_cache.json` and checkpoints nearby amenities as it goes.

## Listings And Projects

```bash
python run.py --type ban --pages 5
python run.py --type cho-thue --pages 5
python run.py --type all-listings --pages 5
python run.py --type du-an --pages 3
```

Outputs:

- `data/listings_ban.json`
- `data/listings_cho_thue.json`
- `data/projects.json`

Use `--no-details` for faster crawls with less structured data. Use `--resume` to continue from crawler checkpoints.

## Related Reviews

When `--keywords` is omitted, review crawlers generate search keywords from crawled data:

- listings: `review {dia_chi}`
- projects: `review {ten_du_an}`

Commands:

```bash
python run.py --type youtube
python run.py --type tiktok
python run.py --type voz --pages 3
```

Default review limits:

- YouTube: 20 videos per keyword, up to 50 comments per video.
- TikTok: 20 videos per keyword, up to 50 comments per video.
- VOZ: 20 threads per keyword, across `--pages` search pages.

Outputs:

- `data/youtube_comments.json`
- `data/tiktok_comments.json`
- `data/voz_discussions.json`

YouTube requires `YOUTUBE_API_KEY` in `.env`. TikTok and VOZ do not require API keys. Google reviews are not scraped; `google-reviews` is optional legacy functionality and requires `GOOGLE_MAPS_API_KEY`.

You can override generated keywords:

```bash
python run.py --type youtube --keywords "review Vinhomes Ocean Park 3,review The Global City"
python run.py --type voz --keywords "review Phu My Hung"
```

## Geocoding

Geocoding uses Nominatim/OpenStreetMap and writes coordinates back into:

- `data/listings_ban.json`
- `data/listings_cho_thue.json`
- `data/projects.json`

Run all records:

```bash
python run.py --type geocode --geocode-limit 0
```

`--geocode-limit 0` means no limit, so all records are scanned. A positive value limits the number of records scanned per file:

```bash
python run.py --type geocode --geocode-limit 10
```

Added fields include:

- `latitude`
- `longitude`
- `geo_source`
- `geo_confidence`

## Nearby Amenities

Nearby amenities use OpenStreetMap Overpass and default to a 2 km radius.

```bash
python run.py --type osm-poi
```

This uses the geocoded `dia_chi` coordinates from sale listings, rental listings, and projects. Each source record gets:

- `nearby_amenities`
- `nearby_amenities_radius_m`
- `nearby_amenities_source`
- `nearby_amenities_target_latitude`
- `nearby_amenities_target_longitude`

Each amenity includes:

- `name`
- `category`
- `address`
- `latitude`
- `longitude`
- `distance_m`
- `place_id`
- `osm_type`
- `osm_id`

Global and per-record outputs:

- `data/pois.json`: deduplicated global POI catalog
- `data/nearby_amenities.json`: source-record to nearby-amenities mapping
- source files also get embedded `nearby_amenities`

Supported amenity categories:

- `school`
- `hospital`
- `transit_station`
- `park`
- `shopping_mall`
- `supermarket`

Manual coordinate search:

```bash
python run.py --type osm-poi --coords 10.7769,106.7009 --radius 2000
```

## Ingestion

After JSON data exists, ingest into the database/vector pipeline:

```bash
python -m db.ingest --source all
```

POIs are ingested as structured Postgres facts from `data/pois.json`; they are not embedded into Qdrant by default.

## Notes

- `data/.osm_cache.json` stores OSM geocode and POI cache entries.
- Public OSM endpoints can rate-limit. Rerun `python run.py --type osm-poi`; cached records will be reused.
- `.env`, `data/`, cache files, and local generated artifacts should stay out of git.
