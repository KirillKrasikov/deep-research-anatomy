from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    query: str
    brief: str
    draft: str
    messages: Annotated[list[BaseMessage], add_messages]
    notes: Annotated[list[str], add]
    completed_supervisor_tool_rounds: int
    final_report: str


class ResearcherState(TypedDict, total=False):
    task: str
    messages: Annotated[list[BaseMessage], add_messages]
    notes: str
    completed_tool_rounds: int
