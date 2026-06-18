import math
import logging
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calculate_poi_distances")

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def main():
    conn_params = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "dbname": "real_estate"
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        log.info("Connected to PostgreSQL successfully")
    except Exception as e:
        log.error(f"Connection failed: {e}")
        return
        
    cur = conn.cursor()
    
    # 1. Fetch listings
    cur.execute("SELECT id, latitude, longitude FROM listings WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    listings = cur.fetchall()
    log.info(f"Fetched {len(listings)} listings from DB")
    
    # 2. Fetch projects
    cur.execute("SELECT id, latitude, longitude FROM projects WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    projects = cur.fetchall()
    log.info(f"Fetched {len(projects)} projects from DB")
    
    # 3. Fetch POIs
    cur.execute("SELECT id, latitude, longitude FROM pois WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    pois = cur.fetchall()
    log.info(f"Fetched {len(pois)} POIs from DB")
    
    # Clear previous distances
    cur.execute("TRUNCATE TABLE entity_poi_distances")
    log.info("Cleared old distances from entity_poi_distances")
    
    # Calculate for listings
    distance_records = []
    
    for lid, l_lat, l_lon in listings:
        for pid, p_lat, p_lon in pois:
            dist = haversine(l_lat, l_lon, p_lat, p_lon)
            if dist <= 2000: # 2km radius
                distance_records.append(("listing", lid, pid, dist, "straight_line"))
                
    # Calculate for projects
    for prid, pr_lat, pr_lon in projects:
        for pid, p_lat, p_lon in pois:
            dist = haversine(pr_lat, pr_lon, p_lat, p_lon)
            if dist <= 2000: # 2km radius
                distance_records.append(("project", prid, pid, dist, "straight_line"))
                
    log.info(f"Calculated {len(distance_records)} distance pairs <= 2km")
    
    # Batch insert into entity_poi_distances
    if distance_records:
        insert_query = """
            INSERT INTO entity_poi_distances (entity_type, entity_id, poi_id, distance_m, travel_mode)
            VALUES %s
            ON CONFLICT (entity_type, entity_id, poi_id, travel_mode) DO NOTHING
        """
        execute_values(cur, insert_query, distance_records)
        log.info(f"Successfully inserted distances into database")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
