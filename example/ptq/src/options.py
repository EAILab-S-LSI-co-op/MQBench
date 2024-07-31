import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    return args