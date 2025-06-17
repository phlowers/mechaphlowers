import json
import os
import math

def sort_by_consecutive_distance(data):
    if not data:
        return data
    
    # Start with the first point
    sorted_data = [data[0]]
    remaining = data[1:]
    
    while remaining:
        current = sorted_data[-1]
        # Find the closest point to the current point
        closest_idx = min(range(len(remaining)), 
                         key=lambda i: haversine_distance(
                             current['geo_point_2d']['lat'],
                             current['geo_point_2d']['lon'],
                             remaining[i]['geo_point_2d']['lat'],
                             remaining[i]['geo_point_2d']['lon']
                         ))
        sorted_data.append(remaining.pop(closest_idx))
    
    return sorted_data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "./reseau-aerien-haute-tension-htb.json"), "r") as f:
        data = json.load(f)
        new_data = [{'geo_point_2d': pylon['geo_point_2d'], 
                    'nom_ligne': pylon['nom_ligne'], 
                    'start': int(pylon['nom_ligne'].replace("porte ", "").split('-')[0]) if 'SAGON' not in pylon['nom_ligne'] else 0, 
                    'end': int(pylon['nom_ligne'].replace("porte ", "").split('-')[1]) if 'SAGON' not in pylon['nom_ligne'] else 0} 
                   for pylon in data 
                   if pylon['region'] == "Corse" and pylon['code_epci'] == "200067049" and not 'SAGON' in pylon['nom_ligne']]
        new_data = sorted(new_data, key=lambda x: x['start'])
        # Sort the data by consecutive point distances
        new_data = sort_by_consecutive_distance(new_data)
        
        print("data is", len(data))
        print("new data is", len(new_data))
        
        #pretty print
        with open(os.path.join(current_dir, "./reseau-aerien-haute-tension-htb-corse.json"), "w") as f:
            json.dump(new_data, f, indent=4)

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



def main():
        # Define the app layout
    np.random.seed(142)
    # np.random.seed(42)
    size_multiple = 5
    size_section = size_multiple*5
    df = generate_section_array(size_section)
    app.layout = html.Div([
        # dcc.Graph(
        #     id='map',
        #     figure=create_map2(df, size_section, size_multiple),
        #     config={
        #         'displayModeBar': True,
        #         'displaylogo': False,
        #         'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
        #     }
        # ),
        dcc.Graph(
            id='map2',
            figure=create_line_map(df),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        ),
        dcc.Graph(
            id='map3',
            figure=create_elevation_profile(df, size_section, size_multiple),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        )
    ], style={'width': '100%', 'height': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})

    app.run(debug=True)

# result = main()
if __name__ == '__main__':
    main()