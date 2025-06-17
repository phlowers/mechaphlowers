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
