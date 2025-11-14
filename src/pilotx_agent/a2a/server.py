import os

import uvicorn

from starlette.middleware.cors import CORSMiddleware
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from ..agent import WorkflowAgent
from ..agents.entities import SessionType
from ..executors.agent_executor import PilotXAgentExecutor


def make_pilotx_a2a_server() -> uvicorn.Server:
    """Run the A2A service on port 9999."""
    agent_host = os.getenv("AGENT_HOST", "http://localhost:9999")

    data_analyst_skill = AgentSkill(
        id="data_analyst_definition",
        name="AI Data Analyst",
        description=(
            "Helps analysts explore and understand data stored in a Postgres database. "
            "Generates efficient SQL queries, runs them via available tools, and interprets the results to provide clear insights. "
            "Can answer questions about metrics, trends, anomalies, segment performance, and other analytical needs."
        ),
        tags=[
            "Data Analysis",
            "SQL",
            "Postgres",
            "Business Intelligence",
            "Analytics",
            "Reporting",
        ],
        examples=[
            "What was our total revenue by month for the last 12 months?",
            "Find the top 10 customers by lifetime value from the transactions table.",
            "Calculate the churn rate for subscriptions over the past 3 months.",
            "Show the daily active users and weekly active users for the last quarter.",
            "Identify any anomalies in orders by day over the past year.",
            "Segment users by country and device type and show average order value.",
            "Compare conversion rates between users who received the new feature and those who did not.",
            "What are the most common reasons for failed payments in the last 30 days?",
            "Give me a retention cohort analysis by signup month.",
            "Summarize key trends in our product usage events over the last 6 months.",
        ],
    )

    data_analyst_agent_card = AgentCard(
        name="Data Analyst",
        description=(
            "Data Analyst is a conversational AI assistant that helps users explore and understand data stored in a "
            "Postgres database. It follows a structured workflow: understand the question → (optionally) clarify "
            "requirements and definitions → generate SQL → execute queries via available tools → interpret and summarize "
            "the results. It can compute metrics, trends, cohorts, segment performance, and anomaly detection, and "
            "provides concise, decision-ready explanations in plain language. It avoids making up data, clearly states "
            "assumptions and limitations, and does not provide legal, tax, or regulated financial advice."
        ),
        url=agent_host,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[data_analyst_skill],
        supports_authenticated_extended_card=True,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PilotXAgentExecutor(
            agent=WorkflowAgent(session_type=SessionType.Firestore),
            streaming=True,
        ),
        task_store=InMemoryTaskStore(),
    )

    application = A2AStarletteApplication(
        agent_card=data_analyst_agent_card, http_handler=request_handler
    )

    app = application.build()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(app, host="0.0.0.0", port=9999, log_config=None)
    server = uvicorn.Server(config)

    return server
