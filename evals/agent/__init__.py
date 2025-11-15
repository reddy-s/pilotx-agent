import logging

from .orchestrator import OrchestratorEvaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

evaluators = [
    ("orchestrator", OrchestratorEvaluation()),
]

def run_evaluation():
    for e, runner in evaluators:
        logger.info(f"Evaluating {e}")
        runner.run()


__all__ = [run_evaluation]