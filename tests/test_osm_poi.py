import json
import tempfile
import unittest
from pathlib import Path

from crawlers.dynamic.osm_poi import OSMPOI
from run import save_nearby_amenities_to_source_files


class OSMPOITest(unittest.TestCase):
    def test_build_overpass_query_contains_category_tags(self):
        query = OSMPOI.build_overpass_query(10.7769, 106.7009, 1000, "school")

        self.assertIn('node["amenity"="school"](around:1000,10.7769,106.7009);', query)
        self.assertIn('way["amenity"="kindergarten"](around:1000,10.7769,106.7009);', query)
        self.assertIn('relation["amenity"="university"](around:1000,10.7769,106.7009);', query)
        self.assertTrue(query.startswith("[out:json]"))

    def test_normalize_geocode_result(self):
        result = OSMPOI.normalize_geocode_result(
            "Ben Thanh, Ho Chi Minh",
            {
                "lat": "10.7769",
                "lon": "106.7009",
                "display_name": "Ben Thanh, Ho Chi Minh City, Vietnam",
                "importance": 0.82,
            },
        )

        self.assertEqual(result["geo_source"], "osm_nominatim")
        self.assertEqual(result["latitude"], 10.7769)
        self.assertEqual(result["longitude"], 106.7009)
        self.assertEqual(result["geo_confidence"], 0.82)

    def test_normalize_poi_element(self):
        poi = OSMPOI.normalize_poi_element(
            {
                "type": "node",
                "id": 123,
                "lat": 10.77,
                "lon": 106.70,
                "tags": {
                    "name": "Cho Ben Thanh",
                    "addr:street": "Le Loi",
                    "addr:city": "Ho Chi Minh City",
                },
            },
            "shopping_mall",
        )

        self.assertEqual(poi["place_id"], "osm:node:123")
        self.assertEqual(poi["name"], "Cho Ben Thanh")
        self.assertEqual(poi["source"], "osm_overpass")
        self.assertEqual(poi["category"], "shopping_mall")

    def test_geocode_cache_hit_does_not_call_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            cache_key = OSMPOI._cache_key("Ben Thanh")
            cached_result = {
                "query": "Ben Thanh",
                "latitude": 10.7769,
                "longitude": 106.7009,
                "geo_source": "osm_nominatim",
                "geo_confidence": 0.8,
            }
            cache_path.write_text(
                json.dumps({"geocode": {cache_key: cached_result}, "poi": {}}),
                encoding="utf-8",
            )

            service = OSMPOI(cache_file=cache_path, output_file=Path(tmp) / "pois.json")
            service.session.get = lambda *args, **kwargs: self.fail("network should not be called")

            self.assertEqual(service.geocode_address("Ben Thanh"), cached_result)

    def test_save_pois_deduplicates_by_place_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "pois.json"
            service = OSMPOI(cache_file=Path(tmp) / "cache.json", output_file=output_path)
            service.save_pois(
                {
                    "school": [
                        {"place_id": "osm:node:1", "name": "A", "category": "school"},
                        {"place_id": "osm:node:1", "name": "A updated", "category": "school"},
                    ]
                }
            )

            records = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["name"], "A updated")

    def test_amenities_for_target_keeps_coordinates_and_sorts_by_distance(self):
        service = OSMPOI(cache_file=Path("/tmp/nonexistent-cache.json"), output_file=Path("/tmp/nonexistent-pois.json"))
        payload = service.amenities_for_target(
            {
                "lat": 10.0,
                "lng": 106.0,
                "source_type": "listing_ban",
                "source_file": "listings_ban.json",
                "source_index": 0,
                "label": "Listing A",
                "dia_chi": "Address A",
            },
            {
                "school": [
                    {
                        "place_id": "osm:node:far",
                        "osm_type": "node",
                        "osm_id": 2,
                        "name": "Far School",
                        "category": "school",
                        "address": "B",
                        "latitude": 10.02,
                        "longitude": 106.0,
                        "source": "osm_overpass",
                    },
                    {
                        "place_id": "osm:node:near",
                        "osm_type": "node",
                        "osm_id": 1,
                        "name": "Near School",
                        "category": "school",
                        "address": "A",
                        "latitude": 10.001,
                        "longitude": 106.0,
                        "source": "osm_overpass",
                    },
                ]
            },
            2000,
        )

        schools = payload["amenities"]["school"]
        self.assertEqual(payload["radius_m"], 2000)
        self.assertEqual(schools[0]["name"], "Near School")
        self.assertEqual(schools[0]["latitude"], 10.001)
        self.assertEqual(schools[0]["longitude"], 106.0)
        self.assertLess(schools[0]["distance_m"], schools[1]["distance_m"])

    def test_save_nearby_amenities_to_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            source_path = data_dir / "listings_ban.json"
            source_path.write_text(json.dumps([{"dia_chi": "Address A"}]), encoding="utf-8")

            summary = save_nearby_amenities_to_source_files(
                data_dir,
                [
                    {
                        "source_file": "listings_ban.json",
                        "source_index": 0,
                        "radius_m": 2000,
                        "target_latitude": 10.0,
                        "target_longitude": 106.0,
                        "amenities": {
                            "school": [
                                {
                                    "name": "Near School",
                                    "latitude": 10.001,
                                    "longitude": 106.0,
                                    "distance_m": 111.2,
                                }
                            ]
                        },
                    }
                ],
            )

            records = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["listings_ban.json"]["updated"], 1)
            self.assertEqual(records[0]["nearby_amenities_radius_m"], 2000)
            self.assertEqual(records[0]["nearby_amenities"]["school"][0]["name"], "Near School")
            self.assertEqual(records[0]["nearby_amenities"]["school"][0]["latitude"], 10.001)


if __name__ == "__main__":
    unittest.main()
