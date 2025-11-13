import logging

from dotenv import load_dotenv
from rich import print

from . import agent
from .config import ServiceConfig

load_dotenv()
__version__ = "0.0.0"


def main() -> None:
    print(ServiceConfig.get_or_create_instance().config)
