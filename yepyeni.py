from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt.tool_node import ToolNode
from typing import List, TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages
import json

# ---------------------------------------------------------------------
# 1. Plan verisi
# ---------------------------------------------------------------------
plan = {
    
}

# ---------------------------------------------------------------------
# 2. Tool'lar
# ---------------------------------------------------------------------
@tool
def AddEventToPlan(input: dict) -> str:
    """
    Adds a new item to the daily plan.

    Input:
        A JSON DICT with two keys:
            - "time": a string representing the time (e.g., "16.00")
            - "event": a string describing the event (e.g., "yemek ye")

    Example input:
        '{"time": "16.00", "event": "yemek ye"}'

    Output:
        A confirmation message like:
        "16.00 → yemek ye eklendi."
    """
    data = json.loads(input) if isinstance(input, str) else input
    time = data["time"]
    event = data["event"]
    plan[time] = event
    return f"{time} → {event} eklendi."

@tool
def ShowPlan(_: str = "") -> str:
    """
    Returns the current daily plan in a sorted format.

    Input:
        (Ignored) Can be an empty string.

    Output:
        A multi-line string showing the current plan, sorted by time.
        Example:
            12.00 → Spor yap
            13.10 → Namaz kıl
            16.00 → yemek ye
    """
    return "\n".join(f"{k} → {v}" for k, v in sorted(plan.items()))

tools = [AddEventToPlan, ShowPlan]

# ---------------------------------------------------------------------
# 3. LLM + system prompt
# ---------------------------------------------------------------------

          # Ollama REST endpoint
llm = ChatOllama(
    model="qwen3:4b",
    base_url="http://localhost:11434",
    temperature=0.2
)
llm = llm.bind_tools(tools)

system_prompt = SystemMessage(content="""
Sen bir günlük plan asistanısın.

Sadece gerektiğinde araç çağır.
""")

# ---------------------------------------------------------------------
# 4. State şeması
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ---------------------------------------------------------------------
# 5. Agent düğümü (LLM step)
# ---------------------------------------------------------------------
def call_llm(state: AgentState) -> AgentState:
    msgs = [system_prompt] + state["messages"]
    response = llm.invoke(msgs)
    print (response)
    return {"messages": [response]}


# ---------------------------------------------------------------------
# 6. Tool node (hazır ToolNode ile)
# ---------------------------------------------------------------------
tool_node = ToolNode(tools)

# ---------------------------------------------------------------------
# 7. Akış kontrolü – agent tool çağırmış mı?
# ---------------------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tool", "__end__"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool"
    return "__end__"

# ---------------------------------------------------------------------
# 8. Graph oluştur
# ---------------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", call_llm)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "tool": "tool",
    "__end__": END
})
graph.add_edge("tool", "agent")

app = graph.compile()

# ---------------------------------------------------------------------
# 9. Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    user_input = HumanMessage(content="bugünkü planımı göster", name="user")
    final_state = app.invoke({"messages": [system_prompt, user_input]})

    print("🔹 Asistanın cevabı:")
    print(final_state["messages"][-1].content)
    print(final_state)
    print(plan)
