import os

DEBUG = os.getenv("R3KIT_DEBUG", "False").lower() in ["true", "1"]
INFO = os.getenv("R3KIT_INFO", "False").lower() in ["true", "1"]
