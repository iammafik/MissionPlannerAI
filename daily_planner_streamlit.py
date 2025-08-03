# daily_planner_streamlit.py
"""
Streamlit arayüzü × LangGraph ajanı (Mafiks Bot)
================================================
Bu sürüm, LangGraph ajanını bozmadan Streamlit’e bağlar **ve**
`st.session_state.chat_input` hatasını çözer:
- Girdi kutusunu temizlemek için artık `_clear_input` bayrağı + `del
  st.session_state["chat_input"]` kullanıyoruz; doğrudan assignment yok.

Kurulum
-------
```bash
pip install streamlit langchain_ollama langgraph
ollama pull qwen3:4b       # model yüklü değilse
streamlit run daily_planner_streamlit.py
```
"""

from __future__ import annotations

import json, os, re, streamlit as st
from typing import List, Dict, Tuple, Annotated, TypedDict, Literal

# ------------------------------------------------------------------
# 1. (DEĞİŞMEDEN) LANGGRAPH AJANI
# ------------------------------------------------------------------
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import add_messages

# 1.1 Plan verisi
plan: Dict[str, str] = {}

# 1.2 Tool'lar
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
    time, event = data["time"], data["event"]
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

TOOLS = [AddEventToPlan, ShowPlan]

# 1.3 LLM + bind_tools
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm_raw = ChatOllama(model="qwen3:4b", base_url=OLLAMA_URL, temperature=0.2)
llm = llm_raw.bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(content="""
Sen bir günlük plan asistanısın. Kullanıcının talimatını anla ve gerektiğinde
araç çağır. Aksi hâlde kullanıcıya normal yanıt ver.
""")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Agent düğümü

def call_llm(state: AgentState) -> AgentState:
    msgs = [SYSTEM_PROMPT] + state["messages"]
    return {"messages": [llm.invoke(msgs)]}

# Tool node & kontrol
TOOL_NODE = ToolNode(TOOLS)

def should_continue(state: AgentState) -> Literal["tool", "__end__"]:
    return "tool" if getattr(state["messages"][-1], "tool_calls", None) else "__end__"

# Graph
GRAPH = StateGraph(AgentState)
GRAPH.add_node("agent", call_llm)
GRAPH.add_node("tool", TOOL_NODE)
GRAPH.set_entry_point("agent")
GRAPH.add_conditional_edges("agent", should_continue, {"tool": "tool", "__end__": END})
GRAPH.add_edge("tool", "agent")
APP = GRAPH.compile()

# ------------------------------------------------------------------
# 2. Yardımcı – ajanı çağır
# ------------------------------------------------------------------

def agent_response(user_text: str) -> str:
    fs = APP.invoke({"messages": [HumanMessage(content=user_text, name="user")]})
    print (fs["messages"])
    return fs["messages"][-1].content

# ------------------------------------------------------------------
# 3. Streamlit UI – chat_input temizleme fixli
# ------------------------------------------------------------------

st.set_page_config(page_title="Daily Planner – Mafiks Bot", layout="wide")

# ― input temizleme bayrağı
if "_clear_input" not in st.session_state:
    st.session_state._clear_input = False
if st.session_state._clear_input:
    try:
        del st.session_state["chat_input"]  # widget yeniden başlatılır
    except KeyError:
        pass
    st.session_state._clear_input = False

left, mid, right = st.columns([1, 4, 1])

# --- Sol Araçlar ---
with left:
    st.header("🔧 Araçlar (LLM)")

    with st.expander("➕ Ekle"):
        at = st.text_input("Saat", key="add_time")
        ad = st.text_input("Açıklama", key="add_desc")
        if st.button("Ekle") and at and ad:
            st.toast(agent_response(f"{at} {ad} ekle"))

    with st.expander("✏️ Güncelle"):
        ut = st.text_input("Saat", key="upd_time")
        ud = st.text_input("Yeni açıklama", key="upd_desc")
        if st.button("Güncelle") and ut and ud:
            st.toast(agent_response(f"{ut} {ud} güncelle"))

    with st.expander("🗑️ Sil"):
        dt = st.text_input("Saat", key="del_time")
        if st.button("Sil") and dt:
            st.toast(agent_response(f"{dt} sil"))

    st.markdown("---")
    if st.button("🧹 Planı Temizle"):
        st.toast(agent_response("planı temizle"))

# --- Orta: Sohbet ---
with mid:
    st.header("🤖 Mafiks Bot")
    if "chatlog" not in st.session_state:
        st.session_state.chatlog: List[Tuple[str, str]] = []

    for role, msg in st.session_state.chatlog:
        icon = "👤" if role == "user" else "🤖"
        st.markdown(f"**{icon} {role}:** {msg}")

    user_msg = st.text_input("Komut yaz…", key="chat_input", label_visibility="collapsed")
    if st.button("➡️", use_container_width=True) and user_msg.strip():
        st.session_state.chatlog.append(("user", user_msg))
        st.session_state.chatlog.append(("Mafiks Bot", agent_response(user_msg)))
        st.session_state._clear_input = True
        st.rerun()

# --- Sağ: Plan ---
with right:
    st.header("🗓️ Plan")
    if plan:
        for t, ev in sorted(plan.items()):
            st.write(f"**{t}** – {ev}")
    else:
        st.write("(Henüz etkinlik yok)")
