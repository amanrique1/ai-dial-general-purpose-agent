import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:
    """
    An agent that orchestrates LLM interactions with tool-calling capabilities.

    The agent sends messages to a chat completion endpoint, interprets any
    tool-call requests from the model, executes the corresponding tools,
    and recursively re-invokes the model until a final text response is
    produced (i.e. no more tool calls are requested).
    """

    def __init__(
        self,
        endpoint: str,
        system_prompt: str,
        tools: list[BaseTool],
    ):
        """
        Args:
            endpoint:      Base URL of the DIAL chat-completion service.
            system_prompt:  System-level instruction prepended to every request.
            tools:          Available tools the model may invoke.
        """
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools

        # Build a lookup table so we can resolve tools by name in O(1).
        self._tools_dict: dict[str, BaseTool] = {
            tool.name: tool for tool in tools
        }

        # Mutable conversation state persisted across recursive calls.
        # TOOL_CALL_HISTORY_KEY accumulates assistant + tool messages produced
        # during the current turn so they can be replayed on the next iteration.
        self.state: dict[str, Any] = {
            TOOL_CALL_HISTORY_KEY: [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        deployment_name: str,
        choice: Choice,
        request: Request,
        response: Response,
    ) -> Message:
        """
        Send the conversation to the LLM and return its final assistant
        message.  If the model requests tool calls, they are executed and
        the model is called again — repeating until it produces a plain
        text answer.

        Args:
            deployment_name: Model deployment to target.
            choice:          SDK choice object used for streaming content back.
            request:         Incoming SDK request (carries messages, auth, etc.).
            response:        SDK response handle (unused directly but forwarded
                             for potential downstream needs).

        Returns:
            The final assistant ``Message`` containing the model's text reply.
        """
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )

        # Request a streaming completion with the current conversation history
        # and the set of available tool definitions.
        chunks = await client.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            stream=True,
            deployment_name=deployment_name,
        )

        # Accumulate streamed content and tool-call deltas.
        content, tool_calls = await self._consume_stream(chunks, choice)

        # Build the assistant message that mirrors what the model produced.
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            custom_content=CustomContent(attachments=[]),
            tool_calls=[
                ToolCall.validate(tc) for tc in tool_calls.values()
            ],
        )

        # If the model requested tool calls, execute them and recurse.
        if assistant_message.tool_calls:
            tool_messages = await self._execute_tool_calls(
                tool_calls=assistant_message.tool_calls,
                choice=choice,
                api_key=request.api_key,
                conversation_id=request.headers["x-conversation-id"],
            )

            # Persist the assistant message and every tool response so that
            # the next iteration includes them in the conversation history.
            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            # Recurse — the model will see the tool results and can either
            # answer or request additional tool calls.
            return await self.handle_request(
                deployment_name=deployment_name,
                choice=choice,
                request=request,
                response=response,
            )

        # No tool calls — we have the final answer.  Persist state and return.
        choice.set_state(self.state)
        return assistant_message

    # ------------------------------------------------------------------
    # Stream consumption
    # ------------------------------------------------------------------

    @staticmethod
    async def _consume_stream(
        chunks: Any,
        choice: Choice,
    ) -> tuple[str, dict[int, Any]]:
        """
        Iterate over SSE chunks, forwarding text content to the caller in
        real time and reassembling any tool-call deltas.

        Returns:
            A tuple of (accumulated_text, tool_call_index_map) where the
            map keys are chunk-level indices and values are the merged
            ``ToolCall`` objects.
        """
        content = ""
        tool_call_index_map: dict[int, Any] = {}

        async for chunk in chunks:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta is None:
                continue

            # --- Text content ---
            if delta.content:
                choice.append_content(delta.content)
                content += delta.content

            # --- Tool-call deltas ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.id:
                        # First fragment for this index — store the skeleton.
                        tool_call_index_map[tc_delta.index] = tc_delta
                    else:
                        # Subsequent fragment — append argument tokens.
                        existing = tool_call_index_map[tc_delta.index]
                        if tc_delta.function:
                            existing.function.arguments += (
                                tc_delta.function.arguments or ""
                            )

        return content, tool_call_index_map

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def _prepare_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """
        Build the full message list sent to the LLM.

        Steps:
            1. Unpack user/assistant messages from the SDK format, merging
               in any tool-call history accumulated during this turn.
            2. Prepend the system prompt.
            3. Log the resulting history for debugging.

        Returns:
            A list of plain dicts ready for the DIAL client.
        """
        unpacked_messages = unpack_messages(
            messages, self.state[TOOL_CALL_HISTORY_KEY]
        )

        # Ensure the system prompt is always the first message.
        unpacked_messages.insert(
            0,
            {
                "role": Role.SYSTEM.value,
                "content": self.system_prompt,
            },
        )

        # Debug logging — helpful for inspecting the conversation window.
        self._log_history(unpacked_messages)

        return unpacked_messages

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        choice: Choice,
        api_key: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        Run every requested tool call concurrently and return the
        serialised tool-response messages.
        """
        tasks = [
            self._process_tool_call(
                tool_call=tc,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            )
            for tc in tool_calls
        ]
        return list(await asyncio.gather(*tasks))

    async def _process_tool_call(
        self,
        tool_call: ToolCall,
        choice: Choice,
        api_key: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """
        Execute a single tool call inside a UI stage and return the
        resulting message as a plain dict.

        Args:
            tool_call:        The tool call descriptor from the model.
            choice:           SDK choice used to create visual stages.
            api_key:          Forwarded to the tool for authenticated calls.
            conversation_id:  Current conversation identifier.

        Returns:
            Serialised tool message (``dict``) with ``None`` values stripped.
        """
        tool_name = tool_call.function.name
        tool = self._tools_dict[tool_name]

        # Open a collapsible stage in the UI so the user can inspect
        # the tool's request and response.
        stage = StageProcessor.open_stage(choice, tool_name)

        if tool.show_in_stage:
            # Render the arguments and a response header inside the stage.
            formatted_args = json.dumps(
                json.loads(tool_call.function.arguments), indent=2
            )
            stage.append_content("## Request arguments: \n")
            stage.append_content(f"```json\n\r{formatted_args}\n\r```\n\r")
            stage.append_content("## Response: \n")

        # Delegate to the concrete tool implementation.
        tool_message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            )
        )

        # Always close the stage, even if the tool raised earlier
        # (StageProcessor.close_stage_safely is idempotent).
        StageProcessor.close_stage_safely(stage)

        return tool_message.dict(exclude_none=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_history(messages: list[dict[str, Any]]) -> None:
        """Print the full conversation history for debugging purposes."""
        print("\nHistory:")
        for msg in messages:
            print(f"     {json.dumps(msg)}")
        print(f"{'-' * 100}\n")