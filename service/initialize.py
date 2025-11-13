import functools
import logging
import os

import dotenv
import yaml

from validations import validate_env_variables


@functools.lru_cache(maxsize=1)
def initialize() -> None:
    """
    Initialize the application with logging and environment validation.
    This function is decorated with @functools.lru_cache(maxsize=1) to ensure
    it only runs once per application lifecycle, even if called multiple times.
    """
    dotenv.load_dotenv()

    LOG_CONFIG_PATH = os.getenv("LOG_CONFIG_PATH", DEFAULT_LOG_CONFIG_PATH)
    setup_logging(LOG_CONFIG_PATH)

    validate_env_variables()


DEFAULT_LOG_CONFIG_PATH = "resources/logging.yaml"


def setup_logging(config_path: str) -> None:
    """
    Load logging configuration from YAML file and apply LOG_LEVEL override.
    Falls back to default config if specified file fails, then to basicConfig if both fail.
    """
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Apply env var to root level to override the default level
        config.setdefault("root", {})["level"] = LOG_LEVEL
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info(
            f"Logging is configured based on the config file: {config_path}"
        )
    except (
        FileNotFoundError,
        yaml.YAMLError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ) as e:
        if config_path != DEFAULT_LOG_CONFIG_PATH:
            # Retry setup with the default config path
            logging.getLogger(__name__).warning(
                f"Failed to load logging config from {config_path}: {e.__repr__()}. Falling back to {DEFAULT_LOG_CONFIG_PATH}",
            )
            setup_logging(DEFAULT_LOG_CONFIG_PATH)
        else:
            # Fallback to basicConfig with LOG_LEVEL
            logging.basicConfig(level=LOG_LEVEL)
            logging.getLogger(__name__).warning(
                f"Failed to load logging config from {config_path}: {repr(e)}. Falling back to basicConfig with LOG_LEVEL={LOG_LEVEL}"
            )
