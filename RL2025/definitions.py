import configparser
import os

PACKAGE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
PROJECT_ROOT_DIR = os.path.dirname(PACKAGE_ROOT_DIR)
CACHE_FOLDER = f"{PROJECT_ROOT_DIR}/cache/"
SEPARATOR = "\u0001"
SEPARATOR_LINE = "\u0002"
SEPARATOR_N = "\u0001\n"
SEPARATOR_LINE_N = "\u0002\n"

CONFIG = configparser.ConfigParser()
CONFIG.read(f"{PROJECT_ROOT_DIR}/config.txt")
