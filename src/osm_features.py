import osmnx as ox


def safe_count(gdf):
    if gdf is None:
        return 0
    try:
        return int(len(gdf))
    except Exception:
        return 0


def get_pois_near_point(latitude, longitude, radius_m=500):
    """
    Extract nearby OSM POI counts around a latitude-longitude point.

    Parameters
    ----------
    latitude : float
    longitude : float
    radius_m : int

    Returns
    -------
    dict
        Dictionary of POI count features.
    """

    point = (latitude, longitude)

    poi_queries = {
        "fuel_station_count": {"amenity": "fuel"},

        "restaurant_count": {
            "amenity": ["restaurant", "fast_food", "cafe"]
        },

        "bus_stop_count": {"highway": "bus_stop"},

        "parking_count": {"amenity": "parking"},

        "construction_count": {"landuse": "construction"},

        "industrial_count": {"landuse": "industrial"},

        "factory_count": {"man_made": "works"},

        "warehouse_count": {"building": "warehouse"},

        "marketplace_count": {"amenity": "marketplace"},

        "park_count": {"leisure": "park"},

        "school_college_count": {
            "amenity": ["school", "college", "university"]
        },

        "hospital_count": {"amenity": "hospital"},

        "commercial_count": {"landuse": "commercial"},

        "retail_count": {"landuse": "retail"},
    }

    features = {}

    for feature_name, tags in poi_queries.items():
        output_name = f"{feature_name}_{radius_m}m"

        try:
            gdf = ox.features_from_point(
                point,
                tags=tags,
                dist=radius_m
            )
            features[output_name] = safe_count(gdf)

        except Exception:
            features[output_name] = 0

    return features


def get_road_features_near_point(latitude, longitude, radius_m=500):
    """
    Extract road network features around a point.

    Features include:
    - road segment count
    - total road length
    - road type counts
    """

    features = {}

    default_features = {
        f"road_segment_count_{radius_m}m": 0,
        f"total_road_length_{radius_m}m": 0.0,
        f"motorway_count_{radius_m}m": 0,
        f"trunk_road_count_{radius_m}m": 0,
        f"primary_road_count_{radius_m}m": 0,
        f"secondary_road_count_{radius_m}m": 0,
        f"tertiary_road_count_{radius_m}m": 0,
        f"residential_road_count_{radius_m}m": 0,
        f"service_road_count_{radius_m}m": 0,
        f"living_street_count_{radius_m}m": 0,
        f"unclassified_road_count_{radius_m}m": 0,
    }

    try:
        graph = ox.graph_from_point(
            (latitude, longitude),
            dist=radius_m,
            network_type="drive"
        )

        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

        features[f"road_segment_count_{radius_m}m"] = int(len(edges))

        if "length" in edges.columns:
            features[f"total_road_length_{radius_m}m"] = float(edges["length"].sum())
        else:
            features[f"total_road_length_{radius_m}m"] = 0.0

        if "highway" in edges.columns:
            highway_series = edges["highway"].astype(str)

            road_types = {
                "motorway": "motorway_count",
                "trunk": "trunk_road_count",
                "primary": "primary_road_count",
                "secondary": "secondary_road_count",
                "tertiary": "tertiary_road_count",
                "residential": "residential_road_count",
                "service": "service_road_count",
                "living_street": "living_street_count",
                "unclassified": "unclassified_road_count",
            }

            for road_key, feature_prefix in road_types.items():
                features[f"{feature_prefix}_{radius_m}m"] = int(
                    highway_series.str.contains(road_key, case=False, na=False).sum()
                )

        for key, value in default_features.items():
            features.setdefault(key, value)

    except Exception:
        features = default_features

    return features


def extract_osm_features(latitude, longitude, radii=(250,)):
    """
    Extract multi-radius OSM features.

    For each radius, we extract:
    - nearby POI counts
    - road network features
    """

    all_features = {}

    for radius in radii:
        poi_features = get_pois_near_point(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius
        )

        road_features = get_road_features_near_point(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius
        )

        all_features.update(poi_features)
        all_features.update(road_features)

    return all_features