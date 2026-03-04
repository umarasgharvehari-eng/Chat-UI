import os
import uuid
import ast
import operator as op
import requests
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


st.set_page_config(page_title="Gemini Agent", page_icon="🤖")

api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if not api_key:
    st.error("Missing GOOGLE_API_KEY. Add it in Streamlit Cloud → App settings → Secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

MODEL = os.getenv("GEMINI_MODEL") or st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.3)

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body
    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        raise ValueError("Unsupported expression")
    return float(_eval(node))

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like: (18.5 * 3) / 2"""
    return str(_safe_eval(expression.strip()))

def _geocode(city: str):
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res:
        return None
    x = res[0]
    return {"name": x.get("name") or city, "country": x.get("country"), "lat": x["latitude"], "lon": x["longitude"]}

@tool
def weather(city: str) -> str:
    """Get live current weather for a city using Open-Meteo (no API key)."""
    city = city.strip()
    geo = _geocode(city)
    if not geo:
        return f"Couldn't find city: {city}"
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": geo["lat"],
            "longitude": geo["lon"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=20,
    )
    r.raise_for_status()
    cur = r.json().get("current") or {}
    loc = f'{geo["name"]}, {geo["country"]}' if geo.get("country") else geo["name"]
    return f"{loc}: {cur.get('temperature_2m')}°C, humidity {cur.get('relative_humidity_2m')}%, wind {cur.get('wind_speed_10m')} km/h"

search = DuckDuckGoSearchRun()

SYSTEM_PROMPT = (
    "You are a powerful, beginner-friendly AI agent.\n"
    "- Use tools when helpful.\n"
    "- For up-to-date info, use duckduckgo search.\n"
    "- For weather, call the weather tool.\n"
    "- Keep answers concise.\n"
    "- If you used search, include 2-4 sources as bullet points."
)

prompt_tmpl = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

@st.cache_resource
def build_agent():
    a = create_tool_calling_agent(llm, [calculator, weather, search], prompt_tmpl)
    return AgentExecutor(agent=a, tools=[calculator, weather, search], verbose=False)

agent = build_agent()

st.title("🤖 Gemini Agent (Tools + Memory + Search)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

for m in st.session_state.ui_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        out = agent.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
        answer = out.get("output", "")
    except Exception as e:
        answer = f"⚠️ Error: {type(e).__name__}: {e}"

    st.session_state.ui_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", answer))
