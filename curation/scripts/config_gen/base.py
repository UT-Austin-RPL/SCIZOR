import os
import time

class ConfigGenerator:
    args = None
    def __init__(self):
        self.scripts = []

    def generate_script(self):
        raise NotImplementedError
        
    def save_config(self, save_dir="./exps"):
        raise NotImplementedError
    
    def parse_args(self):
        raise NotImplementedError