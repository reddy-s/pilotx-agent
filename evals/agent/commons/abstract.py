import logging
import os
from abc import ABC
from functools import partial
from typing import Any, List, Union
from uuid import uuid4

import mlflow
from mlflow.genai import Scorer
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import Guidelines

from pilotx_agent.agents.abstract import (
    AbstractAgent,
    AbstractLoopAgent,
    AbstractParallelAgent,
    AbstractSequentialAgent,
)

from .utils import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    def __init__(
        self,
        instance: Union[
            AbstractAgent
            | AbstractSequentialAgent
            | AbstractParallelAgent
            | AbstractLoopAgent
        ],
        experiment: str = "pilotx",
    ):
        logging.info(
            f"Starting Agent evaluation with experiment: {experiment}"
        )
        self.instance = instance
        self.experiment = experiment

        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        if not self.tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI must be set, it must point to a valid MLflow 3.0 tracking server."
            )

    def run_eval(self, scorers: List[Union[Scorer, Any]], dataset: list[dict]) -> EvaluationResult:
        mlflow.openai.autolog()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        run_id = f"{self.instance.config.name}_{str(uuid4())}"
        logging.info(
            f"Generated run ID: {run_id} for agent: {self.instance.config.name}. Test set size: {len(dataset)}"
        )

        try:
            all_results = mlflow.genai.evaluate(
                data=dataset,
                predict_fn=partial(
                    run_agent,
                    instance=self.instance,
                    user_id="eval-user",
                    session_id=str(uuid4()),
                ),
                scorers=scorers,
            )

            logger.info(
                f"Agent evaluation completed successfully for agent: {self.instance.config.name}"
            )
            return all_results
        except Exception as e:
            logging.error(f"Campaign Agent evaluation failed: {e}")
            raise


class AbstractEvaluationRunner(ABC):
    def __init__(
        self,
        instance: Union[
            AbstractAgent
            | AbstractSequentialAgent
            | AbstractParallelAgent
            | AbstractLoopAgent
        ],
        experiment: str = "pilotx",
    ):
        self.agent_evaluator = AgentEvaluator(instance=instance, experiment=experiment)

    def get_scorers(self) -> List[Union[Guidelines, Scorer]]:
        raise NotImplementedError

    def get_dataset(self) -> list[dict]:
        raise NotImplementedError

    def run(self) -> EvaluationResult:
        scorers = self.get_scorers()
        dataset = self.get_dataset()
        results = self.agent_evaluator.run_eval(scorers=scorers, dataset=dataset)
        return results