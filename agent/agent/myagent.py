# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AI Travel Planner Agent.

Multi-agent LangGraph workflow with:
  - intake_node      : extracts structured trip fields from user message
  - clarify_node     : interrupts to ask missing-field questions (Human-in-the-Loop)
  - supervisor_node  : conditional routing hub that decides which specialist runs next
  - research_node    : weather, country info, flight search
  - budget_node      : currency conversion + budget breakdown
  - planner_node     : itinerary builder, datetime, PII remover
"""
# ruff: noqa: I001
from datetime import datetime
from typing import Any, Literal, Optional, Union

from ag_ui.core import RunAgentInput
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.tools.budget_tools import calculate_budget_breakdown, convert_currency
from agent.tools.planner_tools import build_itinerary, get_current_datetime, remove_pii
from agent.tools.research_tools import get_country_info, get_destination_weather, search_flights
from agent.config import Config


# ---------------------------------------------------------------------------
# Shared typed state
# ---------------------------------------------------------------------------


class TravelState(MessagesState):
    """Shared state passed between all nodes in the travel planner workflow."""

    # Extracted trip parameters
    destination: str
    origin: str
    departure_date: str   # ISO date or human-readable, e.g. "2025-06-10"
    return_date: str      # ISO date or human-readable, e.g. "2025-06-15"
    num_days: int         # Computed from departure_date → return_date
    budget_usd: float
    travel_style: str
    currency: str

    # Sub-agent result buckets
    research_results: dict[str, Any]
    budget_results: dict[str, Any]
    itinerary: dict[str, Any]

    # Workflow control
    needs_clarification: bool
    clarification_question: str
    needs_confirmation: Optional[bool]  # None=not shown yet, True=waiting, False=confirmed
    next_agent: str  # "research" | "budget" | "planner" | "FINISH"
    completed_steps: list[str]

    # Guardrail control — set by _guardrail_node, consumed by _rejection_node
    guardrail_blocked: bool      # True when input failed safety/topic check
    guardrail_rejection: str     # Friendly rejection text to stream to the user


# ---------------------------------------------------------------------------
# Pydantic model for structured intake extraction
# ---------------------------------------------------------------------------


class TripIntakeModel(BaseModel):
    """Structured fields extracted from the user's travel request.

    Every field is required in the JSON schema so DataRobot's response_format
    validation passes. Use empty string / 0 / False as sentinel values for
    fields the user has not yet provided.
    """

    model_config = {"populate_by_name": True}

    destination: str = Field(description="Travel destination extracted from the message, e.g. 'Rome, Italy'. Use empty string '' if not mentioned.")
    origin: str = Field(description="Departure city or airport, e.g. 'London'. Use empty string '' if not mentioned.")
    departure_date: str = Field(description="The date the user will depart / start travelling. ALWAYS resolve to ISO format YYYY-MM-DD, e.g. '2025-06-10'. Resolve relative expressions like 'next Monday' or '7th April' using today's date provided in the prompt. Use empty string '' if not mentioned.")
    return_date: str = Field(description="The date the user will return / arrive back home. ALWAYS resolve to ISO format YYYY-MM-DD, e.g. '2025-06-15'. Resolve relative expressions like 'next Friday' or 'in 2 weeks' using today's date provided in the prompt. Use empty string '' if not mentioned.")
    budget_usd: float = Field(description="Total trip budget converted to USD as a float, e.g. 1200.0. Use 0.0 if not mentioned.")
    travel_style: str = Field(description="Travel preferences, e.g. 'culture, food, adventure, relaxation'. Use empty string '' if not mentioned.")
    currency: str = Field(description="Preferred local currency code ISO 4217, e.g. 'EUR'. Use 'USD' if not mentioned.")
    needs_clarification: bool = Field(description="Set to true if any of these required fields are still empty/zero after extraction: destination, origin, departure_date, return_date, budget_usd.")
    clarification_question: str = Field(description="A single friendly question asking for only the single most critical missing field. Ask for departure_date before return_date. Use empty string '' if needs_clarification is false.")
    confirmation_was_shown: bool = Field(description="Set to true if the conversation history contains an Assistant message that presented a structured trip summary and explicitly asked the user to confirm or correct the details (e.g. 'does everything look correct?', 'please confirm', 'reply to proceed'). Set to false if no such confirmation prompt appears in the history.")
    user_confirmed: bool = Field(description="Only relevant when confirmation_was_shown is true. Set to true if the user's most recent message expresses agreement or approval of the trip details shown (e.g. 'yes', 'looks good', 'it\\'s great', 'I\\'m good with these', 'let\\'s go', 'all correct', 'sure', 'proceed'). Set to false if the user is requesting a change, expressing doubt, or if confirmation_was_shown is false.")


# ---------------------------------------------------------------------------
# Pydantic model for LLM-based input guardrails
# ---------------------------------------------------------------------------


class GuardrailResult(BaseModel):
    """Result of LLM-based input guardrail classification."""

    is_travel_related: bool = Field(
        description=(
            "True if the user message is related to travel planning, trip enquiries, "
            "destinations, flights, hotels, budgets, itineraries, or general follow-up "
            "questions within an ongoing travel-planning conversation. "
            "Set to false only when the message is entirely unrelated to travel."
        )
    )
    is_safe: bool = Field(
        description=(
            "True if the message contains no harmful, illegal, abusive, violent, "
            "sexually explicit, or otherwise inappropriate content. False otherwise."
        )
    )
    rejection_reason: str = Field(
        description=(
            "A short, friendly explanation addressed to the user explaining why the "
            "request cannot be fulfilled. Use empty string '' when is_travel_related "
            "and is_safe are both true."
        )
    )


# ---------------------------------------------------------------------------
# Conditional edge routing functions
# ---------------------------------------------------------------------------


def _route_guardrail(state: TravelState) -> Literal["rejection_node", "intake_node"]:
    """After guardrail_node: route to rejection_node when blocked, intake_node otherwise."""
    if state.get("guardrail_blocked", False):
        return "rejection_node"
    return "intake_node"


def _route_intake(
    state: TravelState,
) -> Literal["clarify_node", "confirm_node", "supervisor_node"]:
    if state.get("needs_clarification", False):
        return "clarify_node"
    # needs_confirmation=False means the user explicitly confirmed → skip confirmation
    if state.get("needs_confirmation") is False:
        return "supervisor_node"
    return "confirm_node"


def _route_confirm(state: TravelState) -> Literal["supervisor_node", "__end__"]:
    """After confirm_node: proceed to supervisor if confirmed, otherwise stop and wait."""
    if state.get("needs_confirmation", True):
        # Still waiting for the user's yes/no — graph ends so the next user
        # message (with their answer) re-enters through intake_node.
        return "__end__"
    return "supervisor_node"


def _route_supervisor(
    state: TravelState,
) -> Literal["research_node", "budget_node", "planner_node", "presenter_node"]:
    next_agent = state.get("next_agent", "research")
    mapping: dict[str, Literal["research_node", "budget_node", "planner_node", "presenter_node"]] = {
        "research": "research_node",
        "budget": "budget_node",
        "planner": "planner_node",
        "FINISH": "presenter_node",
    }
    return mapping.get(next_agent, "presenter_node")


# ---------------------------------------------------------------------------
# MyAgent
# ---------------------------------------------------------------------------


class MyAgent(LangGraphAgent):
    """AI Travel Planner — multi-agent LangGraph workflow.

    Orchestrates three specialist sub-agents (Research, Budget, Planner) via a
    Supervisor node that uses conditional edges. Supports both single-shot and
    multi-turn conversations with Human-in-the-Loop clarification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[Union[bool, str]] = True,
        timeout: Optional[int] = 120,
        *,
        llm: Optional[BaseChatModel] = None,
        workflow_tools: Optional[list[BaseTool]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self._nat_llm = llm
        self._workflow_tools = workflow_tools or []
        self.config = Config()
        self.default_model = self.config.llm_default_model
        if model in ("unknown", "datarobot-deployed-llm"):
            self.model = self.default_model

    # ------------------------------------------------------------------
    # LLM factory — MUST NOT be modified (DataRobot requirement)
    # ------------------------------------------------------------------

    def llm(
        self,
        auto_model_override: bool = True,
    ) -> BaseChatModel:
        """Returns the LLM to use for agent nodes.

        In NAT mode, returns the pre-configured LLM provided at construction.
        In DRUM mode, creates a ChatLiteLLM using the configured API credentials.

        Args:
            auto_model_override: Optional[bool]: If True, it will try and use the model
                specified in the request but automatically back out if the LLM Gateway is
                not available.

        Returns:
            BaseChatModel: The model to use.
        """
        if self._nat_llm is not None:
            return self._nat_llm

        api_base = self.litellm_api_base(self.config.llm_deployment_id)
        model = self.model or self.default_model
        if auto_model_override and not self.config.use_datarobot_llm_gateway:
            model = self.default_model
        if self.verbose:
            print(f"Using model: {model}")

        config = {
            "model": model,
            "api_base": api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "streaming": True,
            "max_retries": 3,
        }

        if not self.config.use_datarobot_llm_gateway and self._identity_header:
            config["model_kwargs"] = {"extra_headers": self._identity_header}  # type: ignore[assignment]

        return ChatLiteLLM(**config)

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert AI Travel Planner. Help the user plan their perfect trip.",
                ),
                ("user", "{user_prompt_content}"),
            ]
        )

    def convert_input_message(self, run_agent_input: RunAgentInput) -> Command:
        """Convert the full AG-UI message history into LangGraph state messages.

        Converts every prior turn in run_agent_input.messages into the
        appropriate HumanMessage / AIMessage so that _intake_node sees the
        complete conversation history, not just the latest user turn.
        """
        history_messages: list[Any] = []
        all_messages = list(run_agent_input.messages or [])

        # Prior messages become raw history entries.
        # The last user message is formatted through the prompt template so
        # the system prompt is included.
        prior_messages = all_messages[:-1] if len(all_messages) > 1 else []

        for msg in prior_messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
            if role == "user":
                history_messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                history_messages.append(AIMessage(content=str(content)))

        raw = extract_user_prompt_content(run_agent_input)
        user_prompt_str = raw if isinstance(raw, str) else str(raw)
        current_messages = self.prompt_template.invoke(
            {"user_prompt_content": user_prompt_str}
        ).to_messages()

        return Command(update={"messages": history_messages + current_messages})  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Node: guardrail_node
    # LLM-based input guardrail — runs before intake to block off-topic
    # or harmful requests before any trip processing begins.
    # ------------------------------------------------------------------

    def _guardrail_node(self, state: TravelState) -> dict[str, Any]:
        """Classify the latest user message with an LLM before any processing.

        Sets guardrail_blocked=True and stores the rejection text in
        guardrail_rejection when the input fails either check. The actual
        streaming of the rejection message is handled by _rejection_node so
        that the base-class streaming layer sees real AIMessageChunk events.

        Returns an empty dict (no state changes) when the message passes so
        that the graph continues normally to intake_node.
        """
        messages = state.get("messages", [])
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human is None:
            return {"guardrail_blocked": False, "guardrail_rejection": ""}

        user_text = last_human.content if isinstance(last_human.content, str) else str(last_human.content)

        guardrail_llm = self.llm().with_structured_output(GuardrailResult)
        result: GuardrailResult = guardrail_llm.invoke(
            "You are a content moderation assistant for a travel-planning chatbot.\n\n"
            "Evaluate the user message below on two criteria:\n"
            "1. Is it related to travel planning (destinations, flights, hotels, budgets, "
            "itineraries, trip styles, dates, currencies, or general follow-up questions "
            "within an ongoing travel conversation)?\n"
            "2. Is it free of harmful, illegal, abusive, or inappropriate content?\n\n"
            f"User message:\n\"{user_text}\"\n\n"
            "Respond with is_travel_related, is_safe, and rejection_reason."
        )

        if self.verbose:
            print(
                f"[guardrail] is_travel_related={result.is_travel_related}, "
                f"is_safe={result.is_safe}"
            )

        if not result.is_safe or not result.is_travel_related:
            rejection = result.rejection_reason or (
                "I'm sorry, I can only help with travel planning requests. "
                "Feel free to ask me about destinations, flights, hotels, or itineraries!"
            )
            return {"guardrail_blocked": True, "guardrail_rejection": rejection}

        return {"guardrail_blocked": False, "guardrail_rejection": ""}

    # ------------------------------------------------------------------
    # Node: rejection_node
    # Streams the guardrail rejection through a real LLM call so the
    # base-class streaming layer emits proper TEXT_MESSAGE_* events.
    # ------------------------------------------------------------------

    @property
    def _rejection_agent(self) -> Any:
        """Single-shot agent that re-states a guardrail rejection naturally."""
        return create_agent(
            self.llm(),
            tools=[],
            system_prompt=make_system_prompt(
                "You are a friendly AI travel planner assistant. "
                "When a user's request falls outside your travel-planning scope or contains "
                "inappropriate content, politely decline and invite them to ask about travel instead. "
                "Keep your response brief, warm, and non-judgmental. "
                "Never output JSON or structured data."
            ),
            name="rejection_agent",
        )

    def _rejection_node(self, state: TravelState) -> dict[str, Any]:
        """Stream the guardrail rejection via a real LLM call.

        Using create_agent (like _clarify_node does) ensures the base-class
        streaming infrastructure sees live AIMessageChunk events and emits
        proper TEXT_MESSAGE_* events to the frontend.
        """
        raw_rejection = state.get("guardrail_rejection", "")
        prompt_text = (
            f"Please communicate the following to the user in a friendly, natural way:\n"
            f"{raw_rejection}"
        )
        result = self._rejection_agent.invoke({"messages": [HumanMessage(content=prompt_text)]})
        friendly_rejection = _last_ai_content(result)

        return {
            "messages": [AIMessage(content=friendly_rejection)],
            "guardrail_blocked": False,
            "guardrail_rejection": "",
        }

    # ------------------------------------------------------------------
    # Node: intake_node
    # Extracts structured trip fields from the user's latest message.
    # ------------------------------------------------------------------


    def _intake_node(self, state: TravelState) -> dict[str, Any]:
        messages = state.get("messages", [])
        print("messages", messages)

        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        today_weekday = today.strftime("%A")  # e.g. "Monday"

        conversation_lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_lines.append(f"{role}: {content}")

        conversation_context = "\n".join(conversation_lines) if conversation_lines else "(no conversation yet)"

        extraction_prompt = (
            f"Today's date is {today_str} ({today_weekday}). Use this to resolve any relative "
            f"date references such as 'next Monday', 'this Friday', '7th April', 'in 2 weeks', etc.\n\n"
            "You are a travel intake assistant. Extract ALL travel details mentioned ANYWHERE "
            "in the conversation history below — not just the latest message.\n\n"
            "CRITICAL RULES FOR FIELD EXTRACTION:\n"
            "- Read every User message from top to bottom before deciding what is known.\n"
            "- A field is 'known' if it was mentioned in ANY prior User message.\n"
            "- NEVER leave a field empty/zero if it was mentioned in an earlier turn.\n"
            "- Extract BOTH the departure date (when the user leaves) and the return date "
            "(when the user comes back). These are separate fields.\n"
            "- Always resolve dates to the ISO format YYYY-MM-DD (e.g. '2025-06-10'). "
            "Use today's date above as the reference point for all relative expressions.\n"
            "- For example, if the user said 'I leave on June 10 and return on June 15', "
            "set departure_date='2025-06-10' and return_date='2025-06-15'.\n\n"
            "CRITICAL RULES FOR CONFIRMATION DETECTION:\n"
            "- Set confirmation_was_shown=true ONLY if the conversation contains an Assistant "
            "message that presented the trip details as a structured summary AND explicitly asked "
            "the user to confirm (e.g. 'does everything look correct?', 'please confirm', "
            "'reply with yes', 'let me know what to change').\n"
            "- Set user_confirmed=true ONLY when confirmation_was_shown=true AND the user's "
            "most recent message  expresses acceptance — this includes ANY positive or agreeable "
            "reply such as: 'yes', 'sure', 'ok', 'great', 'it\\'s cool', 'it\\'s very great', "
            "'I\\'m good with these', 'all correct', 'let\\'s go', 'proceed', 'sounds good', "
            "or any similar expression of approval. "
            "Set user_confirmed=false if the user is asking for a change or expressing doubt.\n\n"
            f"Full conversation history:\n{conversation_context}\n\n"
            "Now extract all fields. Required fields are: destination, origin, departure_date, "
            "return_date, budget_usd. If any required field is still missing after reading the "
            "entire conversation, set needs_clarification=True and ask for ONLY the single most "
            "critical missing field (ask for departure_date before return_date). "
            "Do not ask for multiple things at once."
        )

        structured_llm = self.llm().with_structured_output(TripIntakeModel)
        extracted: TripIntakeModel = structured_llm.invoke(extraction_prompt)

        # Compute num_days from the two explicit dates when possible
        num_days = _compute_num_days(extracted.departure_date, extracted.return_date)

        # The LLM already determined from the full conversation history whether:
        #   - a confirmation prompt was previously shown (extracted.confirmation_was_shown)
        #   - the user's latest reply is an approval    (extracted.user_confirmed)
        # Use those two flags directly — no state persistence or keyword matching needed.
        if extracted.confirmation_was_shown and extracted.user_confirmed:
            needs_confirmation: Optional[bool] = False  # _route_intake → supervisor_node
        else:
            needs_confirmation = None  # _route_intake → confirm_node (show/re-show summary)

        print(f"[intake] confirmation_was_shown={extracted.confirmation_was_shown}, user_confirmed={extracted.user_confirmed} → needs_confirmation={needs_confirmation}")

        return {
            "destination": extracted.destination,
            "origin": extracted.origin,
            "departure_date": extracted.departure_date,
            "return_date": extracted.return_date,
            "num_days": num_days,
            "budget_usd": extracted.budget_usd,
            "travel_style": extracted.travel_style,
            "currency": extracted.currency or "USD",
            "needs_clarification": extracted.needs_clarification,
            "clarification_question": extracted.clarification_question,
            "needs_confirmation": needs_confirmation,
            "research_results": state.get("research_results", {}),
            "budget_results": state.get("budget_results", {}),
            "itinerary": state.get("itinerary", {}),
            "completed_steps": state.get("completed_steps", []),
            "next_agent": state.get("next_agent", "research"),
        }

    # ------------------------------------------------------------------
    # Node: clarify_node
    # Runs a streaming clarify_agent so the base-class streaming layer
    # emits proper TEXT_MESSAGE_* events to the frontend, then resets
    # the clarification flag before the graph terminates.
    # ------------------------------------------------------------------

    @property
    def _clarify_agent(self) -> Any:
        """Single-shot clarification agent — no tools, streams its response naturally."""
        return create_agent(
            self.llm(),
            tools=[],
            system_prompt=make_system_prompt(
                "You are a friendly AI travel planner having a warm conversation with a potential traveller. "
                "When asked to clarify a missing detail, write a single short, encouraging, conversational "
                "message. Never list multiple questions. Never output JSON or structured data."
            ),
            name="clarify_agent",
        )

    def _clarify_node(self, state: TravelState) -> dict[str, Any]:
        """Stream a warm clarification question via the clarify_agent so that the
        base-class streaming infrastructure picks up the AIMessageChunk events
        and the frontend receives a proper TEXT_MESSAGE_* sequence.
        """
        raw_question = state.get("clarification_question", "")
        already_know = {
            k: v for k, v in {
                "destination": state.get("destination", ""),
                "origin": state.get("origin", ""),
                "departure_date": state.get("departure_date", ""),
                "return_date": state.get("return_date", ""),
                "budget_usd": state.get("budget_usd", 0.0),
                "travel_style": state.get("travel_style", ""),
            }.items() if v and v not in ("", 0, 0.0)
        }

        prompt_text = (
            f"So far you know about the user's trip: {already_know if already_know else 'nothing yet'}.\n"
            f"Please ask the user (in a friendly, natural way): {raw_question}"
        )

        result = self._clarify_agent.invoke({"messages": [HumanMessage(content=prompt_text)]})
        friendly_question = _last_ai_content(result)

        return {
            "needs_clarification": False,
            "clarification_question": "",
            "messages": [AIMessage(content=friendly_question)],
        }

    # ------------------------------------------------------------------
    # Node: confirm_node
    # Presents the extracted trip details to the user and waits for
    # confirmation before any expensive API calls are made.
    # ------------------------------------------------------------------

    @property
    def _confirm_agent(self) -> Any:
        """Confirmation agent — streams a summary of extracted trip data."""
        return create_agent(
            self.llm(),
            tools=[],
            system_prompt=make_system_prompt(
                "You are a friendly AI travel planner assistant. "
                "Your task is to present a clean summary of the trip details you have understood "
                "and ask the user to confirm they are correct before you start planning. "
                "Format the summary clearly using bullet points. "
                "End with a short, friendly question asking the user to reply with 'yes' to confirm "
                "or to let you know what needs to be changed. "
                "Never output JSON or technical details."
            ),
            name="confirm_agent",
        )

    def _confirm_node(self, state: TravelState) -> dict[str, Any]:
        """Present the extracted trip data and ask the user to confirm before
        any expensive sub-agent API calls are triggered.

        On the first pass (needs_confirmation is not yet set), this node
        renders a human-readable summary and terminates the graph turn so
        the user can reply.  When the user's next message indicates approval
        (detected in intake_node via the conversation history), intake sets
        needs_confirmation=False and the graph routes to supervisor_node.
        """
        destination = state.get("destination", "")
        origin = state.get("origin", "")
        departure_date = state.get("departure_date", "")
        return_date = state.get("return_date", "")
        num_days = state.get("num_days", 0)
        budget_usd = state.get("budget_usd", 0.0)
        travel_style = state.get("travel_style", "not specified")
        currency = state.get("currency", "USD")

        summary_prompt = (
            "Please confirm the following trip details with the user:\n\n"
            f"- **Destination**: {destination}\n"
            f"- **Departing from**: {origin}\n"
            f"- **Departure date**: {departure_date}\n"
            f"- **Return date**: {return_date}\n"
            f"- **Duration**: {num_days} days\n"
            f"- **Total budget**: ${budget_usd:,.0f} USD\n"
            f"- **Preferred currency**: {currency}\n"
            f"- **Travel style**: {travel_style}\n\n"
            "Present these details in a warm, friendly way and ask the user to confirm "
            "everything looks correct, or to let you know what to change."
        )

        result = self._confirm_agent.invoke({"messages": [HumanMessage(content=summary_prompt)]})
        confirmation_message = _last_ai_content(result)

        return {
            "needs_confirmation": True,  # flip to True → _route_confirm will send to END
            "messages": [AIMessage(content=confirmation_message)],
        }

    # ------------------------------------------------------------------
    # Node: supervisor_node
    # Reads completed_steps and state to decide which sub-agent runs next.
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: TravelState) -> dict[str, Any]:
        """Orchestrate sub-agent execution order using deterministic rule-based routing.

        Reads completed_steps and result buckets to decide which specialist runs
        next. Fully rule-based to avoid any risk of LLM-driven infinite loops.
        """
        completed = state.get("completed_steps", [])
        research_done = bool(state.get("research_results"))
        budget_done = bool(state.get("budget_results"))
        itinerary_done = bool(state.get("itinerary"))

        # Rule-based fast path avoids an LLM call when the order is clear
        if not research_done:
            next_agent = "research"
            reasoning = "Research must run first to gather destination info, weather, and flights."
        elif not budget_done:
            next_agent = "budget"
            reasoning = "Budget analysis runs after research to use flight costs and currency info."
        elif not itinerary_done:
            next_agent = "planner"
            reasoning = "Planner builds the day-by-day itinerary using research and budget results."
        else:
            next_agent = "FINISH"
            reasoning = "All sub-agents have completed. The travel plan is ready."

        if self.verbose:
            print(f"[supervisor] → {next_agent}: {reasoning}")

        return {
            "next_agent": next_agent,
            "completed_steps": completed,
        }

    # ------------------------------------------------------------------
    # Node factories for specialist sub-agents
    # ------------------------------------------------------------------

    @property
    def _research_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[get_destination_weather, get_country_info, search_flights] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Travel Research Agent. Your job is to gather accurate destination intelligence.\n"
                "\n"
                "Use your tools to:\n"
                "1. Fetch current weather for the destination (get_destination_weather)\n"
                "2. Retrieve country facts: capital, language, currency, timezone (get_country_info)\n"
                "3. Search for available flights from the origin to the destination (search_flights)\n"
                "\n"
                "Always call all three tools. Return a comprehensive JSON summary of your findings.\n"
                "Include best flight option price so the budget agent can use it."
            ),
            name="research_agent",
        )

    @property
    def _budget_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[convert_currency, calculate_budget_breakdown] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Travel Budget Agent. Your job is to produce a clear financial plan for the trip.\n"
                "\n"
                "Use your tools to:\n"
                "1. Convert the total budget to the destination's local currency (convert_currency)\n"
                "2. Build an itemised cost breakdown across flights, hotel, food, activities and transport "
                "(calculate_budget_breakdown)\n"
                "\n"
                "Use the flight cost from the research results when provided. Estimate hotel at ~$100/night "
                "and activities at 15% of total budget if not specified.\n"
                "Return a JSON summary with converted budget and full breakdown table."
            ),
            name="budget_agent",
        )

    @property
    def _planner_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[build_itinerary, get_current_datetime, remove_pii] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Travel Planner Agent. Your job is to create a rich, day-by-day itinerary.\n"
                "\n"
                "Use your tools to:\n"
                "1. Get the current date/time if travel dates were not specified (get_current_datetime)\n"
                "2. Build a structured day-by-day itinerary using the destination, trip style, "
                "research results and budget (build_itinerary)\n"
                "3. If the user's message contained any personal data (emails, phone numbers), "
                "clean it before including in the output (remove_pii)\n"
                "\n"
                "Tailor activities to the travel style (e.g. food tours for 'food lover', "
                "museums for 'culture', hiking for 'adventure').\n"
                "Return a beautifully structured itinerary in JSON + a friendly markdown summary."
            ),
            name="planner_agent",
        )

    # ------------------------------------------------------------------
    # Sub-agent node wrappers
    # These run the specialist react agent and store results in state.
    # These run the specialist react agent and store results in state.
    # ------------------------------------------------------------------

    def _research_node(self, state: TravelState) -> dict[str, Any]:
        """Run the research sub-agent and persist results into shared state."""
        context_msg = HumanMessage(
            content=(
                f"Research the following trip:\n"
                f"- Destination: {state.get('destination', 'unknown')}\n"
                f"- Origin: {state.get('origin', 'unknown')}\n"
                f"- Departure date: {state.get('departure_date', 'flexible')}\n"
                f"- Return date: {state.get('return_date', 'flexible')}\n"
                f"- Duration: {state.get('num_days', 1)} days\n"
                f"- Style: {state.get('travel_style', 'general')}\n"
                f"\nGather weather, country info, and flight options. "
                f"Use the exact departure and return dates when searching for flights."
            )
        )
        result = self._research_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)
        completed = list(state.get("completed_steps", []))
        if "research" not in completed:
            completed.append("research")
        return {
            "research_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    def _budget_node(self, state: TravelState) -> dict[str, Any]:
        """Run the budget sub-agent and persist results into shared state."""
        context_msg = HumanMessage(
            content=(
                f"Calculate the budget for this trip:\n"
                f"- Total budget: ${state.get('budget_usd', 0)} USD\n"
                f"- Destination currency: {state.get('currency', 'USD')}\n"
                f"- Duration: {state.get('num_days', 1)} days\n"
                f"- Research results: {state.get('research_results', {}).get('summary', 'N/A')}\n"
                f"\nConvert currency and produce an itemised cost breakdown."
            )
        )
        result = self._budget_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)
        completed = list(state.get("completed_steps", []))
        if "budget" not in completed:
            completed.append("budget")
        return {
            "budget_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    def _planner_node(self, state: TravelState) -> dict[str, Any]:
        """Run the planner sub-agent and persist the final itinerary into shared state."""
        context_msg = HumanMessage(
            content=(
                f"Create a detailed travel itinerary:\n"
                f"- Destination: {state.get('destination', 'unknown')}\n"
                f"- Departure date: {state.get('departure_date', 'flexible')}\n"
                f"- Return date: {state.get('return_date', 'flexible')}\n"
                f"- Duration: {state.get('num_days', 1)} days\n"
                f"- Travel style: {state.get('travel_style', 'general')}\n"
                f"- Research findings: {state.get('research_results', {}).get('summary', 'N/A')}\n"
                f"- Budget plan: {state.get('budget_results', {}).get('summary', 'N/A')}\n"
                f"\nBuild a day-by-day itinerary anchored to the departure and return dates. "
                f"Use get_current_datetime only if dates are unclear."
            )
        )
        result = self._planner_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)
        completed = list(state.get("completed_steps", []))
        if "planner" not in completed:
            completed.append("planner")
        return {
            "itinerary": {"summary": last_ai},
            "completed_steps": completed,
        }

    # ------------------------------------------------------------------
    # Node: presenter_node
    # Final node — synthesizes all sub-agent results into a single
    # beautiful, user-facing markdown response. No tools needed.
    # ------------------------------------------------------------------

    def _presenter_node(self, state: TravelState) -> dict[str, Any]:
        """Synthesize all collected data into a polished, user-facing travel plan.

        This dedicated presenter agent reads the research, budget, and itinerary
        results and produces a single, cohesive markdown response. It never
        exposes raw JSON, internal state, or agent prefixes to the user.
        """
        destination = state.get("destination", "your destination")
        origin = state.get("origin", "your city")
        departure_date = state.get("departure_date", "")
        return_date = state.get("return_date", "")
        num_days = state.get("num_days", 0)
        budget_usd = state.get("budget_usd", 0.0)
        travel_style = state.get("travel_style", "general")
        research = state.get("research_results", {}).get("summary", "")
        budget = state.get("budget_results", {}).get("summary", "")
        itinerary = state.get("itinerary", {}).get("summary", "")

        date_range = ""
        if departure_date and return_date:
            date_range = f"{departure_date} → {return_date}"
        elif departure_date:
            date_range = f"from {departure_date}"

        synthesis_prompt = (
            f"You are a friendly, expert travel consultant presenting a complete trip plan to a client.\n"
            f"\n"
            f"Trip details:\n"
            f"- Destination: {destination}\n"
            f"- Departing from: {origin}\n"
            f"- Travel dates: {date_range if date_range else 'flexible'} ({num_days} days)\n"
            f"- Total budget: ${budget_usd:,.0f} USD\n"
            f"- Travel style: {travel_style}\n"
            f"\n"
            f"Research findings:\n{research}\n"
            f"\n"
            f"Budget analysis:\n{budget}\n"
            f"\n"
            f"Itinerary:\n{itinerary}\n"
            f"\n"
            f"Write a warm, engaging, well-formatted markdown response presenting this as a complete "
            f"travel plan. Use clear sections with headers. Do NOT expose raw JSON, internal agent "
            f"labels, or technical details. Speak directly to the traveller in second person ('you'). "
            f"End with 2-3 practical travel tips specific to the destination."
        )

        response = self.llm().invoke(synthesis_prompt)
        final_text = response.content if isinstance(response.content, str) else str(response.content)

        return {"messages": [AIMessage(content=final_text)]}

    # ------------------------------------------------------------------
    # Workflow graph
    # ------------------------------------------------------------------

    @property
    def workflow(self) -> StateGraph[TravelState]:  # type: ignore[override]
        graph: StateGraph[TravelState] = StateGraph(TravelState)

        # Register nodes
        graph.add_node("guardrail_node", self._guardrail_node)
        graph.add_node("rejection_node", self._rejection_node)
        graph.add_node("intake_node", self._intake_node)
        graph.add_node("clarify_node", self._clarify_node)
        graph.add_node("confirm_node", self._confirm_node)
        graph.add_node("supervisor_node", self._supervisor_node)
        graph.add_node("research_node", self._research_node)
        graph.add_node("budget_node", self._budget_node)
        graph.add_node("planner_node", self._planner_node)
        graph.add_node("presenter_node", self._presenter_node)

        # Entry point — guardrail runs first, blocks unsafe/off-topic input
        graph.add_edge(START, "guardrail_node")
        graph.add_conditional_edges(
            "guardrail_node",
            _route_guardrail,
            {"rejection_node": "rejection_node", "intake_node": "intake_node"},
        )
        # Rejection streams the decline message then terminates
        graph.add_edge("rejection_node", END)

        # Intake → clarify (missing fields) | confirm (show summary) | supervisor (confirmed)
        graph.add_conditional_edges(
            "intake_node",
            _route_intake,
            {
                "clarify_node": "clarify_node",
                "confirm_node": "confirm_node",
                "supervisor_node": "supervisor_node",
            },
        )

        # Clarify resets flags then terminates — next user message starts a fresh run
        # that flows through intake_node again with the accumulated state.
        graph.add_edge("clarify_node", END)

        # Confirm presents the trip summary and waits — next turn re-enters via intake_node
        graph.add_conditional_edges(
            "confirm_node",
            _route_confirm,
            {"supervisor_node": "supervisor_node", "__end__": END},
        )

        # Supervisor → specialist or presenter (conditional)
        graph.add_conditional_edges(
            "supervisor_node",
            _route_supervisor,
            {
                "research_node": "research_node",
                "budget_node": "budget_node",
                "planner_node": "planner_node",
                "presenter_node": "presenter_node",
            },
        )

        # All specialist nodes return to supervisor after completion
        graph.add_edge("research_node", "supervisor_node")
        graph.add_edge("budget_node", "supervisor_node")
        graph.add_edge("planner_node", "supervisor_node")

        # Presenter is the final node — delivers the polished response then ends
        graph.add_edge("presenter_node", END)

        return graph  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_ai_content(result: Any) -> str:
    """Extract the last AI message content from a react-agent result dict."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return str(result)


def _compute_num_days(departure_date: str, return_date: str) -> int:
    """Return the number of trip days derived from departure and return dates.

    Tries several common date formats. Falls back to 0 when parsing fails so
    the supervisor can still ask for clarification.
    """
    if not departure_date or not return_date:
        return 0

    formats = [
        "%Y-%m-%d",
        "%B %d %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]

    dep_dt = ret_dt = None
    for fmt in formats:
        try:
            dep_dt = datetime.strptime(departure_date.strip(), fmt)
            break
        except ValueError:
            continue

    for fmt in formats:
        try:
            ret_dt = datetime.strptime(return_date.strip(), fmt)
            break
        except ValueError:
            continue

    if dep_dt and ret_dt and ret_dt > dep_dt:
        return (ret_dt - dep_dt).days

    return 0
