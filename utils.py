import yaml

def load_config():
    with open('config.yaml', 'r') as file:
        loaded_data = yaml.safe_load(file)

    return loaded_data
