import logging

from run_servers import run_agent_executor

logger = logging.getLogger(__name__)

__version__ = "0.0.1"


def main() -> None:
    run_agent_executor()


if __name__ == "__main__":
    main()
