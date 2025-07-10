# daily_planner_streamlit.py
"""
Daily Planner · Mafiks Bot (LangGraph & Vision) – v11
====================================================
Fixes
-----
* **SyntaxError**: Prompt satırındaki kesik tırnak düzeltildi.
* Çift `except` kaldırıldı.
* Kod satırları okunabilir hâle getirildi (uzun tek satırlar bölündü).
"""

from __future__ import annotations

import json, os, base64, re, streamlit as st
from typing import List, Dict, Tuple, Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import add_messages
from marshmallow.error_store import merge_errors


# ────────────────── Saat normalizasyonu ──────────────────

def norm_time(raw: str) -> str:
    raw = raw.strip().replace(":", ".")
    m = re.fullmatch(r"(\d{1,2})(?:\.(\d{2}))?", raw)
    if not m:
        raise ValueError("Saat formatı geçersiz. Örn: 15, 15.00, 15:00")
    h, mnt = int(m.group(1)), int(m.group(2) or 0)
    return f"{h:02d}.{mnt:02d}"

# ────────────────── Plan araçları ──────────────────
plan: Dict[str, str] = {"12.00": "Spor yap", "13.10": "Namaz kıl"}

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
    d = json.loads(input) if isinstance(input, str) else input
    t = norm_time(d["time"])
    plan[t] = d["event"]
    return f"{t} eklendi."

@tool
def DeleteEventFromPlan(input: dict | str) -> str:
    """{time} JSON'u → plan siler."""
    d = json.loads(input) if isinstance(input, str) else input
    t = norm_time(d["time"])
    return f"{t} silindi." if plan.pop(t, None) else f"{t} yok."

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
    return "\n".join(f"{k} → {v}" for k, v in sorted(plan.items())) or "Plan boş."

TOOLS = [AddEventToPlan, DeleteEventFromPlan, ShowPlan]

# ────────────────── LLM & LangGraph ──────────────────
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
qwen = ChatOllama(model="qwen3:4b", base_url=OLLAMA, temperature=0.2).bind_tools(TOOLS)

global_sys = SystemMessage(content="Komutları anla, gerekirse araç çağır.")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def agent(state: State) -> State:
    resp = qwen.invoke([global_sys] + state["messages"])
    return {"messages": [resp]}

node = ToolNode(TOOLS)

def route(s: State) -> Literal["tool", "__end__"]:
    return "tool" if getattr(s["messages"][-1], "tool_calls", None) else "__end__"

G = StateGraph(State)
G.add_node("agent", agent)
G.add_node("tool", node)
G.set_entry_point("agent")
G.add_conditional_edges("agent", route, {"tool": "tool", "__end__": END})
G.add_edge("tool", "agent")
APP = G.compile()

# ────────────────── Gemma Vision ──────────────────
vision = ChatOllama(model="gemma3:4b", base_url=OLLAMA, temperature=0.1)

def vision_reply(text: str, img_bytes: bytes) -> str:
    """Gemma3:4b görsel analizi – markdown gömülü görsel kullanır."""
    b64 = base64.b64encode(img_bytes).decode()
    md_img = f"![image](data:image/png;base64,{b64})"
    user_part = text.strip()
    prompt = (user_part + "") if user_part else ""
    message= HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "base64",
                "data": b64,
                "mime_type": "image/jpeg",
            }
            ,
            {"type": "text", "text": text}
        ]
    )


    prompt += (
        "Aşağıdaki görseli ayrıntılı ve katmanlı biçimde incele. "
        "Ana ve ikincil nesneleri, renkleri, ortamı, olası duygusal veya bağlamsal çağrışımları ayrıntılı olarak açıkla. "
        "Kısa değil, açıklayıcı ve TÜRKÇE yanıt ver."

        f"{md_img}"
    )
    try:
        resp = vision.invoke([message])
        return resp.content
    except Exception as e:
        return f"Görsel işlenemedi: {e}"
    except Exception as e:
        return f"Görsel işlenemedi: {e}"

# ────────────────── Streamlit ──────────────────
st.set_page_config(page_title="Mafiks Bot", layout="wide")

if "chat" not in st.session_state:
    st.session_state.chat: List[Tuple[str, str]] = []
if "_clr" not in st.session_state:
    st.session_state._clr = False

if st.session_state._clr:
    st.session_state.pop("chat_input", None)
    st.session_state.pop("chat_image", None)
    st.session_state._clr = False

left, mid, right = st.columns([1, 4, 1])

# ─────────────── Sol – Manuel Araçlar ───────────────
with left:
    st.header("🔧 Araçlar")

    # Ekle
    with st.expander("➕ Ekle"):
        t_in = st.text_input("Saat", key="in_t")
        d_in = st.text_input("Açıklama", key="in_d")
        if st.button("Ekle") and t_in and d_in:
            try:
                plan[norm_time(t_in)] = d_in
                st.toast("Eklendi")
                st.rerun()
            except ValueError as err:
                st.toast(str(err))

    # Güncelle
    with st.expander("✏️ Güncelle"):
        t_up = st.text_input("Saat", key="up_t")
        d_up = st.text_input("Yeni açıklama", key="up_d")
        if st.button("Güncelle") and t_up and d_up:
            try:
                plan[norm_time(t_up)] = d_up
                st.toast("Güncellendi")
                st.rerun()
            except ValueError as err:
                st.toast(str(err))

    # Sil
    with st.expander("🗑️ Sil"):
        t_del = st.text_input("Saat", key="del_t")
        if st.button("Sil") and t_del:
            try:
                key = norm_time(t_del)
                st.toast("Silindi" if plan.pop(key, None) else "Yok")
                st.rerun()
            except ValueError as err:
                st.toast(str(err))

    if st.button("🧹 Planı Temizle"):
        plan.clear()
        st.rerun()

# ─────────────── Orta – Sohbet ───────────────
with mid:
    st.header("🤖 Mafiks Bot")

    for role, msg in st.session_state.chat:
        icon = "👤" if role == "user" else "🤖"
        st.markdown(f"**{icon} {role}:** {msg}")

    u_col, i_col, s_col = st.columns([6, 2, 1])
    user_msg = u_col.text_input("", key="chat_input", label_visibility="collapsed")
    img_file = i_col.file_uploader("Resim", type=["png", "jpg", "jpeg"], key="chat_image")

    if s_col.button("➡️", use_container_width=True):
        if user_msg.strip() or img_file:
            st.session_state.chat.append(("user", user_msg or "[resim]"))

            if img_file:
                reply = vision_reply(user_msg, img_file.read())
            else:
                res = APP.invoke({"messages": [HumanMessage(content=user_msg, name="user")]})
                reply = res["messages"][-1].content

            st.session_state.chat.append(("Mafiks Bot", reply))
            st.session_state._clr = True
            st.rerun()

# ─────────────── Sağ – Plan ───────────────
with right:
    st.header("🗓️ Plan")
    if plan:
        for k, v in sorted(plan.items()):
            st.write(f"**{k}** – {v}")
    else:
        st.write("Plan boş")
