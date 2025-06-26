import os
import json

class ModelHelper:

    def __init__(self):
        pass

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            print("Warning: config for " + config_path + " was not found.\nUsing default config...")
            return {}
        json_data = open(config_path, "r")
        config = json.load(json_data)
        return config