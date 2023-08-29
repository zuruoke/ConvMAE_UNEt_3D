import json
import os

def save_to_json(list_data, root_dir,name):
    file_name = f'{name}.json'
    file_path = os.path.join(root_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(list_data, f)