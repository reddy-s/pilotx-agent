import asyncio
import logging
import signal
import threading
import time
from typing import List, Tuple

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

from initialize import initialize

initialize()

from pilotx_agent.a2a import make_pilotx_a2a_server


def make_adk_web_server() -> uvicorn.Server:
    """Run the ADK web service on port 8888 for accessing the agents UI."""
    config = uvicorn.Config(
        get_fast_api_app(
            agents_dir="../src",
            web=True,
            host="0.0.0.0",
            port=8888,
            reload_agents=True,
        ),
        host="0.0.0.0",
        port=8888,
        log_config=None,  # Rely on the current logging config
    )
    server = uvicorn.Server(config)
    return server


def block_until_server_interrupt_requested(sig=None):
    # Flag used to wake up main thread when shutting down
    stop_gate = threading.Event()
    # Handle Ctrl+C (SIGINT) and kill (SIGTERM): log and set stop flag

    for sig in uvicorn.server.HANDLED_SIGNALS:
        signal.signal(
            sig,
            lambda signalnum, handler, sig=sig: (
                logging.info(f"Interrupt signal received {sig.__repr__()}"),
                stop_gate.set(),
            ),
        )
    # Main thread blocks here until a signal handler sets stop_gate
    stop_gate.wait()


def run_servers_in_threads(
    servers: List[Tuple[str, uvicorn.Server]], service_description: str
):
    """Run multiple servers in separate threads and handle graceful shutdown."""
    threads = []

    # Start all servers in separate threads
    for name, server in servers:
        logging.info(
            f"Starting {name} on http://{server.config.host}:{server.config.port}"
        )
        thread = threading.Thread(
            target=lambda s=server: asyncio.run(s.serve()), name=name
        )
        thread.start()
        threads.append(thread)

    time.sleep(1)
    logging.info(f"Press CTRL+C to stop {service_description}")
    block_until_server_interrupt_requested()

    # Tell all uvicorn servers to shut down gracefully after stop flag is unblocked
    for name, server in servers:
        server.should_exit = True

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    logging.info(f"{service_description} stopped cleanly.")


def run_agent_executor() -> None:
    servers = [("ai_data_analyst", make_pilotx_a2a_server())]
    run_servers_in_threads(servers, "ai_data_analyst")
