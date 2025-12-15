import logging
import os
import mlflow
import json

from abc import ABC
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import (
    InMemorySessionService,
    Session,
    DatabaseSessionService,
)
from google.adk.tools import load_memory, FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content, Part
from mlflow.entities import SpanType
from pydantic import BaseModel
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .entities import AgentConfig, ContentRoles, SessionType, AgentType
from ..storage import FirestoreSessionService

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Represents a runner responsible for managing agents, sessions, and operations execution.

    This class is designed to facilitate interactions between agents and users.
    It manages sessions, memory, and operations tied to user-agent interactions,
    providing utilities for asynchronous execution and streaming functionality.

    :ivar runner: Instance of the Runner used for executing operations.
    :type runner: Runner
    :ivar session: Service object for session management.
    :type session: InMemorySessionService | DatabaseSessionService | MongoDBSessionService | FirestoreSessionService
    :ivar memory: Service object for managing memory storage.
    :type memory: InMemoryMemoryService
    """

    runner = None
    session = None
    memory = InMemoryMemoryService()

    def __init__(
        self,
        agent: Agent | SequentialAgent | ParallelAgent | LoopAgent,
        session_type: SessionType = SessionType.InMemory,
        plugins: Optional[List[BasePlugin]] = None,
    ) -> None:
        if session_type == SessionType.Database:
            state_path_prefix = os.environ.get("STATE_PATH", None)
            if state_path_prefix is None:
                state_path_prefix = "."
            else:
                os.makedirs(state_path_prefix, exist_ok=True)

            db_path = (
                f"sqlite:///{os.path.join(state_path_prefix, 'agent_session_state.db')}"
            )
            self.session = DatabaseSessionService(db_path)
        elif session_type == SessionType.Firestore:
            try:
                self.session = FirestoreSessionService()
                logger.info(
                    f"Initialized Firestore service with config: {self.session}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Firestore session service: {e}")
                logger.warning("Falling back to InMemory session service")
                self.session = InMemorySessionService()
        else:
            self.session = InMemorySessionService()

        self.runner = Runner(
            app_name=agent.name,
            agent=agent,
            session_service=self.session,
            memory_service=self.memory,
            plugins=plugins,
        )

    @retry(
        retry=retry_if_exception_type(ValidationError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=1),
        reraise=True,
    )
    async def stream(
        self, prompt: str, user_id: str, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        # ensure session exists
        await self.get_or_create_session(
            user_id=user_id, app_name=self.runner.app_name, session_id=session_id, user_prompt=prompt
        )

        user_content = Content(role=ContentRoles.User.value, parts=[Part(text=prompt)])

        streaming_mode = RunConfig(streaming_mode=StreamingMode.SSE)

        # You need a run config set to streaming mode to stream
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
            run_config=streaming_mode,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        yield {
                            "agent": event.author,
                            "type": "function_call",
                            "content": f"Running '{part.function_call.name}'...",
                            "function_name": part.function_call.name,
                            "lastResponse": False,
                        }

                    # Handle function responses
                    elif hasattr(part, "function_response") and part.function_response:
                        yield {
                            "agent": event.author,
                            "type": "function_response",
                            "content": f"Finished running '{part.function_response.name}'.",
                            "function_name": part.function_response.name,
                            "lastResponse": False,
                        }

                    # Handle regular text content (streaming tokens)
                    elif hasattr(part, "text") and part.text and event.partial:
                        yield {
                            "agent": event.author,
                            "type": "text",
                            "content": part.text,
                            "function_name": None,
                            "lastResponse": False,
                        }

            if event.is_final_response() and event.content and event.content.parts:
                current_state = await self.get_current_session_state(
                    app_name=self.runner.app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

                final_response_content = None
                response_type = "text"
                if event.content and event.content.parts:
                    try:
                        final_response_content = json.loads(event.content.parts[0].text)
                        response_type = "json"
                    except json.JSONDecodeError as e:
                        final_response_content = "".join(
                            [p.text for p in event.content.parts if p.text]
                        )

                yield {
                    "agent": event.author,
                    "type": response_type,
                    "content": final_response_content,
                    "function_name": None,
                    "lastResponse": False,
                    "state": current_state,
                }

    @mlflow.trace(span_type=SpanType.AGENT, name="agent_invoke")
    async def invoke(
        self, prompt: str, user_id: str, session_id: str = None
    ) -> List[dict]:
        # ensure session exists
        await self.get_or_create_session(user_id=user_id, app_name=self.runner.app_name, session_id=session_id, user_prompt=prompt)
        res = []
        user_content = Content(role=ContentRoles.User.value, parts=[Part(text=prompt)])
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        logger.info(f"Function call: {part.function_call}")
                        res.append(
                            {
                                "type": "function_call",
                                "args": part.function_call.args,
                                "content": f"Running '{part.function_call.name}'...",
                                "function_name": part.function_call.name,
                                "done": False,
                            }
                        )
                    elif hasattr(part, "function_response") and part.function_response:
                        logger.info(f"Function response: {part.function_response}")
                        res.append(
                            {
                                "type": "function_response",
                                "content": f"Finished running '{part.function_response.name}'.",
                                "tool_response": part.function_response.response,
                                "function_name": part.function_response.name,
                                "done": False,
                            }
                        )

            if event.is_final_response() and event.content and event.content.parts:
                final_response_content = ""
                if event.content and event.content.parts:
                    final_response_content = "".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                current_state = await self.get_current_session_state(
                    app_name=self.runner.app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                logger.info(f"Final response: {final_response_content}")
                res.append(
                    {
                        "done": True,
                        "type": "text",
                        "content": final_response_content,
                        "function_name": None,
                        "state": current_state,
                    }
                )
        return res

    async def get_or_create_session(
        self, app_name: str, user_id: str, session_id: str, user_prompt: str = "New Conversation"
    ) -> Session:
        """
        Retrieves an existing session or creates a new one if it does not exist. If the session is new,
        it initializes the session state with a title and a turn counter.

        :param app_name: The name of the application associated with the session.
        :type app_name: str
        :param user_id: The unique identifier for the user.
        :type user_id: str
        :param session_id: The unique identifier for the session.
        :type session_id: str
        :param user_prompt: The initial prompt for the user's conversation. Defaults to "New Conversation".
        :type user_prompt: str, optional
        :return: The retrieved or newly created session object.
        :rtype: Session
        """
        session = await self.session.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if not session:
            session = await self.session.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id, state={"conversationTitle": user_prompt, "turn": 0}
            )
        return session

    async def get_current_session_state(
        self, app_name: str, user_id: str, session_id: str
    ) -> dict[str, Any]:
        """
        Retrieves the updated session state for a given user and session.

        This asynchronous method fetches the current session state from the session
        service based on the provided application name, user ID, and session ID.
        It returns the session object if found, or None if no session exists.

        :param app_name: The name of the application associated with the session.
        :type app_name: str
        :param user_id: The unique identifier of the user whose session is to be retrieved.
        :type user_id: str
        :param session_id: The unique identifier of the session to be retrieved.
        :type session_id: str
        :return: The updated Session object if found, otherwise None.
        :rtype: Optional[Session]
        """
        session = await self.session.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        return session.state if session else {}


class AbstractAgent(ABC):
    """
    Represents an abstract base class for building agents.

    This class is designed to provide a structured implementation for managing agents,
    their configurations, and the interaction mechanisms with tools, callbacks, and
    language models. The `AbstractAgent` serves as a blueprint for specialized agents
    that require extensive configurability and extendable behavior through pre-defined
    callback functions. The integration with tools and models allows it to support
    complex workflows seamlessly.

    :ivar _agent: The internal agent instance being managed.
    :type _agent: Agent
    :ivar config: Configurations and settings specifying agent behavior.
    :type config: AgentConfig
    :ivar _runner: Internal instance of the agent runner, which handles the execution
        logic with the associated session type.
    :type _runner: AgentRunner
    """

    _agent = None
    config = None
    _runner = None

    def __init__(
        self,
        agent_type: AgentType,
        config: AgentConfig,
        tools: list[FunctionTool] | None = None,
        output_schema: type[BaseModel] = None,
        global_instruction: str = None,
        include_memory_tool: bool = False,
        sub_agents: (
            list[Agent | SequentialAgent | ParallelAgent | LoopAgent] | None
        ) = None,
        session_type: SessionType = SessionType.Database,
        plugins: Optional[List[BasePlugin]] = None,
    ):
        self.agent_type = agent_type
        self.config = config
        self._session_type = session_type
        self._plugins = plugins
        if not tools:
            tools = []

        if not sub_agents:
            sub_agents = []

        if include_memory_tool:
            tools.append(load_memory)

        self.model = LiteLlm(model=self.config.modelName)

        self._agent = Agent(
            name=self.config.name,
            model=self.model,
            description=self.config.description,
            instruction=self.config.instruction,
            global_instruction=global_instruction,
            tools=tools,
            output_schema=output_schema,
            output_key=self.config.outputKey,
            before_model_callback=self._before_model_callback,
            after_model_callback=self._after_model_callback,
            before_tool_callback=self._before_tool_callback,
            after_tool_callback=self._after_tool_callback,
            sub_agents=sub_agents,
        )

    def _before_model_callback(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """
        Executes the callback function prior to the interaction with the language model
        (Llm) to allow preprocessing or preparation. It is called with the current
        callback context and the request being sent to the model. The function can
        be used to modify, log, or interact with the request data before it is handled
        by the language model.

        :param callback_context: Provides contextual information about the current
            interaction, including the agent name and other metadata.
        :type callback_context: CallbackContext
        :param llm_request: Represents the request being sent to the language model.
            Includes details of the input, options, or configurations for processing
            the request.
        :type llm_request: LlmRequest
        :return: Can optionally return a response object which will short-circuit
            the normal flow, bypassing the usual processing of the request by
            the language model. If no response is provided, the normal flow proceeds.
        :rtype: Optional[LlmResponse]
        """
        logger.debug(f"Before Model Callback triggered: {callback_context.agent_name}")
        return None

    def _after_model_callback(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """
        Handles the callback function to execute after a language model response is
        processed.

        This method is designed to be invoked as part of a callback mechanism. It
        performs necessary operations using the provided callback context and the
        language model response. By default, this function currently does not
        modify the response and returns `None`.

        :param callback_context: Context object containing information about the
            callback process, such as agent-specific details.
        :type callback_context: CallbackContext
        :param llm_response: The response object generated by the language model,
            encapsulating output details.
        :type llm_response: LlmResponse
        :return: Optionally returns an updated or modified language model response.
            Returns `None` if no changes are applied.
        :rtype: Optional[LlmResponse]
        """
        logger.debug(f"After Model Callback triggered: : {callback_context.agent_name}")
        return None

    def _before_tool_callback(
        self, tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
    ) -> Optional[Dict]:
        """
        Executes a callback function that is triggered before a tool is invoked. This function
        allows for logging or additional preprocessing of arguments or context prior to the
        execution of the specified tool.

        :param tool: The tool instance being invoked.
        :type tool: BaseTool
        :param args: Dictionary of arguments provided to the tool.
        :type args: Dict[str, Any]
        :param tool_context: Contextual information specific to the tool's execution.
        :type tool_context: ToolContext
        :return: Optionally, returns a dictionary containing modified arguments or additional
            data for the tool execution. If no modifications or additional data are required,
            it returns None.
        :rtype: Optional[Dict]
        """
        logger.debug(
            f"Before Tool Callback triggered for tool {tool.name}: {tool_context.agent_name}"
        )
        return None

    def _after_tool_callback(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict,
    ) -> Optional[dict]:
        """
        Handles post-execution logic for a tool by triggering a callback.

        This function is invoked after the execution of a specific tool. It logs the
        triggering of the callback event, including details about the tool and the
        agent invoking it. The function can optionally return a dictionary, if additional
        processing or data manipulation is required after the execution.

        :param tool: The tool instance that was executed.
        :type tool: BaseTool
        :param args: A dictionary containing arguments passed to the tool during execution.
        :type args: Dict[str, Any]
        :param tool_context: Contextual information associated with the execution of the tool,
            including related metadata and agent details.
        :type tool_context: ToolContext
        :param tool_response: The response or output returned by the tool after its execution.
        :type tool_response: Any
        :return: An optional dictionary with additional data to use post tool execution,
            or None if no further processing is required.
        :rtype: Optional[CallToolResult]
        """
        logger.debug(
            f"After Tool Callback triggered for tool {tool.name}: {tool_context.agent_name}"
        )
        return None

    @property
    def agent(self) -> Agent:
        """
        Gets the agent associated with the instance.

        This property is used to retrieve the `Agent` object that represents the
        agent associated with the current instance. It encapsulates and returns
        the `_agent` attribute, allowing external access while keeping the
        underlying attribute private.

        :return: The `Agent` object linked to the instance.
        :rtype: Agent
        """
        return self._agent

    @property
    def runner(self) -> AgentRunner:
        """
        Provides access to the AgentRunner instance associated with the current agent. The
        property initializes the AgentRunner if it has not been created already, using the
        agent and session type provided during the creation of the instance.

        :rtype: AgentRunner
        :return: Returns the AgentRunner instance associated with the current agent.
        """
        if self._runner is None:
            self._runner = AgentRunner(
                agent=self._agent,
                session_type=self._session_type,
                plugins=self._plugins,
            )
        return self._runner


class AbstractSequentialAgent(ABC):
    """
    Defines an abstract base class for a sequential agent which serves as a wrapper
    around a SequentialAgent with additional functionality, specifically lazy initialization
    of an AgentRunner and encapsulation of properties.

    This class is designed primarily to facilitate sequential execution of sub-agents,
    grouped under a common agent with meaningful metadata such as `name` and
    `description`. It provides an abstract structure for managing and interacting
    with the agent and its corresponding runner.

    :ivar _agent: Internal reference to the sequential agent, encapsulating its name,
        description, and sub-agents. It is initialized during the instantiation of the class.
    :type _agent: SequentialAgent
    :ivar _runner: Stores the runner instance managing the sequential agent. Lazily initialized
        upon first access to avoid unnecessary resource usage.
    :type _runner: AgentRunner or None
    """

    _agent = None
    _runner = None

    def __init__(
        self,
        name: str,
        description: str,
        sub_agents: list[Agent | AbstractAgent],
        session_type: SessionType = SessionType.InMemory,
        plugins: Optional[List[BasePlugin]] = None,
    ) -> None:
        self._session_type = session_type
        self._agent = SequentialAgent(
            name=name,
            description=description,
            sub_agents=sub_agents,
        )
        self._plugins = plugins

    @property
    def agent(self) -> SequentialAgent:
        """
        Property representing the `agent` attribute.

        This property retrieves the `_agent` attribute, which is of type
        `SequentialAgent`.

        :return: Instance of the `SequentialAgent` type.
        :rtype: SequentialAgent
        """
        return self._agent

    @property
    def runner(self) -> AgentRunner:
        """
        Provides access to the `runner` property, which initializes and returns
        an instance of `AgentRunner` if it has not been previously created.

        Attributes:
            runner (AgentRunner): The `runner` property representing an
            instance of the `AgentRunner` class. It is initialized using
            the `_agent` instance and a session type `SessionType.MongoDB`.

        :return: Instance of `AgentRunner`
        :rtype: AgentRunner
        """
        if self._runner is None:
            self._runner = AgentRunner(
                agent=self._agent, session_type=self._session_type, plugins=self._plugins
            )
        return self._runner


class AbstractLoopAgent(ABC):
    """
    Defines an abstract base class for loop agent functionality.

    This class serves as a blueprint for creating loop-based agent implementations. It uses
    an internal `LoopAgent` object to handle its operations and supports maintaining sub-agents,
    providing descriptions, and controlling execution iterations.

    :ivar _agent: Stores an instance of the `LoopAgent` that encapsulates the main
       functionalities of the loop agent.
    :type _agent: LoopAgent
    :ivar _runner: Represents a placeholder for an execution runner or environment, which
       may control or supervise the agent's operational loop.
    :type _runner: Any
    """

    _agent = None
    _runner = None

    def __init__(
        self,
        name: str,
        description: str,
        sub_agents: list[Agent],
        session_type: SessionType = SessionType.InMemory,
        max_iterations: int = 3,
        plugins: Optional[List[BasePlugin]] = None,
    ) -> None:
        """
        Initializes an instance of the LoopAgent wrapper class with the specified
        name, description, sub-agents, and maximum iterations for processing.

        :param name: The name of the LoopAgent instance.
        :type name: str
        :param description: A detailed description of the LoopAgent's purpose.
        :type description: str
        :param sub_agents: A list of sub-agents associated with this LoopAgent.
        :type sub_agents: list[Agent]
        :param max_iterations: The maximum number of iterations to run for the
            LoopAgent. Defaults to 3.
        :type max_iterations: int
        """
        self._plugins = plugins
        self._session_type = session_type
        self._agent = LoopAgent(
            name=name,
            description=description,
            sub_agents=sub_agents,
            max_iterations=max_iterations,
        )

    @property
    def agent(self) -> LoopAgent:
        """
        Provides a read-only access property for the `_agent` attribute, ensuring
        encapsulation of the `LoopAgent` instance.

        :return: The `LoopAgent` instance associated with this object.
        :rtype: LoopAgent
        """
        return self._agent

    @property
    def runner(self) -> AgentRunner:
        """
        Provides access to the AgentRunner instance associated with the current agent. The
        property initializes the AgentRunner if it has not been created already, using the
        agent and session type provided during the creation of the instance.

        :rtype: AgentRunner
        :return: Returns the AgentRunner instance associated with the current agent.
        """
        if self._runner is None:
            self._runner = AgentRunner(
                agent=self._agent,
                session_type=self._session_type,
                plugins=self._plugins,
            )
        return self._runner


class AbstractParallelAgent(ABC):
    """
    Abstract base class for creating parallel agents.

    This class serves as a base class for managing parallel agents. It provides
    properties to retrieve the underlying parallel agent instance and its
    corresponding runner. The purpose of the class is to simplify the creation
    and management of such entities and ensure the proper initialization of the
    parallel agent and runner components.

    The class initializes a ParallelAgent instance upon creation and includes a
    lazy initialization mechanism for the corresponding runner.

    :ivar _agent: The initialized ParallelAgent instance associated with the
        abstract parallel agent.
    :type _agent: ParallelAgent
    :ivar _runner: The lazy-initialized AgentRunner associated with the
        abstract parallel agent.
    :type _runner: AgentRunner
    """

    _agent = None
    _runner = None

    def __init__(
        self,
        name: str,
        description: str,
        sub_agents: list[Agent],
        session_type: SessionType = SessionType.InMemory,
        plugins: Optional[List[BasePlugin]] = None,
    ) -> None:
        self._session_type = session_type
        self._plugins = plugins
        self._agent = ParallelAgent(
            name=name,
            description=description,
            sub_agents=sub_agents,
        )

    @property
    def agent(self) -> ParallelAgent:
        return self._agent

    @property
    def runner(self) -> AgentRunner:
        if self._runner is None:
            self._runner = AgentRunner(
                agent=self._agent, session_type=self._session_type, plugins=self._plugins
            )
        return self._runner
