import json
import os

class ConfigLoader:
    def __init__(self, config_path="config/config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            config = json.load(f)
        
        return config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_all(self):
        return self.config