def estimate_density_placeholder(row):
    """
    Temporary placeholder density formula.

    This is NOT the final scientific formula.
    It is only for testing the full image -> features -> density workflow.

    Later, replace this with your partner's BAM-calibrated density formula.
    """

    total_vehicles = row.get("total_vehicles", 0)
    car_count = row.get("car_count", 0)
    motorcycle_count = row.get("motorcycle_count", 0)
    bus_count = row.get("bus_count", 0)
    truck_count = row.get("truck_count", 0)
    bicycle_count = row.get("bicycle_count", 0)
    road_dust_score = row.get("road_dust_score", 0)

    # If these columns are not already present in CSV, calculate them here
    heavy_vehicle_count = row.get("heavy_vehicle_count", bus_count + truck_count)

    traffic_load_score = row.get(
        "traffic_load_score",
        (
            1.0 * car_count
            + 0.6 * motorcycle_count
            + 3.0 * bus_count
            + 3.5 * truck_count
            + 0.1 * bicycle_count
        )
    )

    dust_traffic_interaction_score = row.get(
        "dust_traffic_interaction_score",
        traffic_load_score * road_dust_score
    )

    density = (
        1.0
        + 0.02 * total_vehicles
        + 0.10 * heavy_vehicle_count
        + 0.50 * road_dust_score
        + 0.03 * dust_traffic_interaction_score
    )

    return density