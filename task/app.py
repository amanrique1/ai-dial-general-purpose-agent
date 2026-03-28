import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.py_interpreter.python_code_interpreter_tool import (
    PythonCodeInterpreterTool,
)
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

# ---------------------------------------------------------------------------
# Configuration – all tunables are read from environment variables with
# sensible local-development defaults.
# ---------------------------------------------------------------------------

DIAL_ENDPOINT: str = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME: str = os.getenv("DEPLOYMENT_NAME", "gpt-4o")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class GeneralPurposeAgentApplication(ChatCompletion):
    """
    DIAL chat-completion implementation that delegates to a
    :class:`GeneralPurposeAgent` backed by a configurable set of tools.

    Tools are created lazily on the first request so that the event loop
    is guaranteed to be running when async initialisation (e.g. MCP
    handshakes) takes place.
    """

    def __init__(self) -> None:
        # Populated once on the first call to ``chat_completion``.
        self.tools: list[BaseTool] = []

    # ------------------------------------------------------------------
    # MCP discovery
    # ------------------------------------------------------------------

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        """
        Connect to an MCP server at *url* and return one ``MCPTool``
        wrapper for every tool the server advertises.

        Raises:
            Exception: Re-raised after logging if the MCP handshake or
                tool listing fails (e.g. server unreachable).
        """
        try:
            mcp_client = await MCPClient.create(url)
            tool_models = await mcp_client.get_tools()

            return [
                MCPTool(client=mcp_client, mcp_tool_model=model)
                for model in tool_models
            ]
        except Exception as exc:
            print(f"Warning: Could not load MCP tools from {url}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    async def _create_tools(self) -> list[BaseTool]:
        """
        Instantiate the full tool-belt used by the agent.

        The list currently includes:
        * **ImageGenerationTool** – generates images via a DIAL deployment.
        * **FileContentExtractionTool** – extracts text from uploaded files.
        * **RagTool** – retrieval-augmented generation over cached documents.
        * **PythonCodeInterpreterTool** – sandboxed Python execution (MCP).
        * **DuckDuckGo search tools** – web search exposed through MCP.
        """
        py_interpreter_mcp_url = os.getenv(
            "PYINTERPRETER_MCP_URL", "http://localhost:8050/mcp"
        )
        print(f"PYINTERPRETER_MCP_URL {py_interpreter_mcp_url}")

        # -- Built-in tools (direct implementations) ----------------------
        tools: list[BaseTool] = [
            ImageGenerationTool(endpoint=DIAL_ENDPOINT),
            FileContentExtractionTool(endpoint=DIAL_ENDPOINT),
            RagTool(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                document_cache=DocumentCache.create(),
            ),
            await PythonCodeInterpreterTool.create(
                mcp_url=py_interpreter_mcp_url,
                tool_name="execute_code",
                dial_endpoint=DIAL_ENDPOINT,
            ),
        ]

        # -- MCP-based tools (dynamically discovered) ----------------------
        ddg_mcp_url = os.getenv("DDG_MCP_URL", "http://localhost:8051/mcp")
        print(f"DDG_MCP_URL {ddg_mcp_url}")
        tools.extend(await self._get_mcp_tools(ddg_mcp_url))

        return tools

    # ------------------------------------------------------------------
    # DIAL SDK entry point
    # ------------------------------------------------------------------

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        """
        Handle an incoming chat-completion request.

        On the very first invocation the tool list is initialised (lazy
        init).  Every request then gets its own ``GeneralPurposeAgent``
        instance so that per-turn state (tool-call history) is isolated.
        """
        # Lazy tool initialisation — safe because chat_completion is
        # always called inside a running event loop.
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


# ---------------------------------------------------------------------------
# DIAL application wiring
# ---------------------------------------------------------------------------

app: DIALApp = DIALApp()
agent_app = GeneralPurposeAgentApplication()

# Register the agent under a logical deployment name so that DIAL can
# route ``/chat/completions`` requests to it.
app.add_chat_completion(deployment_name="general-purpose-agent", impl=agent_app)

# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")