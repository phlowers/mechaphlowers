import json
import os
import requests
import time
from typing import List, Dict

def get_elevation(locations: List[Dict[str, float]]) -> List[float]:
    """
    Get elevation data for a list of locations using Open-Elevation API
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    # Format locations for the API
    payload = {
        "locations": [{"latitude": loc["lat"], "longitude": loc["lon"]} for loc in locations]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return [result["elevation"] for result in data["results"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation data: {e}")
        return [0.0] * len(locations)  # Return zeros if request fails

def process_elevations(data: List[Dict], batch_size: int = 100) -> List[Dict]:
    """
    Process elevation data in batches to avoid overwhelming the API
    """
    processed_data = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        locations = [{"lat": pylon['geo_point_2d']['lat'], 
                     "lon": pylon['geo_point_2d']['lon']} for pylon in batch]
        
        # Get elevations for the batch
        elevations = get_elevation(locations)
        
        # Add elevation to each pylon in the batch
        for pylon, elevation in zip(batch, elevations):
            pylon['elevation'] = elevation
            processed_data.append(pylon)
        
        # Add a small delay between batches to be nice to the API
        if i + batch_size < len(data):
            time.sleep(1)
    
    return processed_data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "./reseau-aerien-haute-tension-htb-corse.json")
    output_file = os.path.join(current_dir, "./reseau-aerien-haute-tension-htb-corse-with-elevation.json")
    
    # Read the input data
    with open(input_file, "r") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} points...")
    
    # Process elevations in batches
    processed_data = process_elevations(data)
    
    # Save the processed data
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"Elevation data saved to {output_file}")
