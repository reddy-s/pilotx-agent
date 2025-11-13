import logging
import os

logger = logging.getLogger(__name__)


def validate_env_variables():
    """
    Validates the presence of required environment variables. This function checks for the presence of
    a set of environment variables essential for the application's proper functioning. If any of the
    required variables are missing, it raises an exception to signal that the environment is improperly
    configured. Otherwise, it logs a success message indicating all variables are set.

    Optional Environment Variables enable the usage of integrations. When these variables are not set, certain features may be disabled.

    :raises EnvironmentVariableNotFound: If any of the required environment variables are not present.

    :return: None
    """
    env_vars = [
        "OPENAI_API_KEY",
        "CONFIG_PATH",
        "CONFIG_SCHEMA_PATH",
        "LOG_CONFIG_PATH",
        "AGENT_HOST",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "TAVILY_API_KEY"
    ]

    optional_env_vars = ["DEV_MODE", "STATE_PATH"]

    missing_vars = [var for var in env_vars if os.getenv(var) is None]
    missing_optional_vars = [var for var in optional_env_vars if os.getenv(var) is None]

    if missing_vars:
        raise EnvironmentVariableNotFound(
            f"Environment variables missing: {missing_vars}"
        )
    else:
        logger.info("All required environment variables are set.")

    if missing_optional_vars:
        logger.info(f"Optional environment variables not set: {missing_optional_vars}")
        logger.info("Some features may be disabled.")
    else:
        logger.info("All optional environment variables are set.")


class EnvironmentVariableNotFound(Exception):
    def __init__(self, env_var_name: str):
        self.message = f"[ASE:000] Environment variable ${env_var_name} not found"
        super().__init__(self.message)
