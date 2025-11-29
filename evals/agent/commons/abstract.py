import logging
import os
from abc import ABC
from functools import partial
from typing import Any, List, Union, Tuple
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
            "AbstractAgent",
            "AbstractSequentialAgent",
            "AbstractParallelAgent",
            "AbstractLoopAgent",
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

    def _get_or_create_experiment(self) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        exp = mlflow.get_experiment_by_name(self.experiment)
        if exp is None:
            raise RuntimeError(
                f"Failed to get or create MLflow experiment '{self.experiment}'"
            )

        logger.info(
            "Using MLflow experiment '%s' (id=%s, artifact_location=%s)",
            exp.name,
            exp.experiment_id,
            exp.artifact_location,
        )
        return exp.experiment_id

    def _get_or_create_run(self, run_name: str, tag_name: str = "agentEval") -> Tuple[str, str]:
        experiment_id = self._get_or_create_experiment()

        # Look for an existing run with this logical ID
        existing = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.{tag_name} = '{run_name}'",
            max_results=1,
        )

        if len(existing) > 0:
            # Run exists → just return its ID, do NOT start it here
            run_id = existing.iloc[0].run_id
        else:
            # No run -> create a new one and tag it with the logical ID,
            # then end it so it's not active anymore.
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                mlflow.set_tag(tag_name, run_name)
                run_id = run.info.run_id

            # at this point the run is ENDED; we can reopen it later by run_id

        return experiment_id, run_id

    def run_eval(
        self,
        scorers: List[Scorer],
        dataset: list[dict],
    ) -> EvaluationResult:
        # Enable OpenAI / GenAI autologging
        mlflow.openai.autolog()

        run_name = f"{self.instance.config.name}-eval"
        experiment_id, run_id = self._get_or_create_run(run_name=run_name)
        logger.info(
            f"Using MLflow experiment_id={experiment_id}, run_id={run_id} for agent evaluation"
        )

        # Now we start the run exactly once and keep it active during evaluation
        with mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
        ) as run:
            logger.info(
                "Started MLflow run %s for agent '%s' in experiment '%s' "
                "(experiment_id=%s). Test set size: %d",
                run.info.run_id,
                self.instance.config.name,
                self.experiment,
                experiment_id,
                len(dataset),
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

                eval_run_id = getattr(all_results, "run_id", run_id)

                logger.info(
                    "Agent evaluation completed successfully for agent '%s'. "
                    "Evaluation run_id=%s",
                    self.instance.config.name,
                    eval_run_id,
                )
                return all_results

            except Exception as e:
                logger.error(
                    "Campaign Agent evaluation failed for agent '%s' in run %s: %s",
                    self.instance.config.name,
                    run_id,
                    e,
                )
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