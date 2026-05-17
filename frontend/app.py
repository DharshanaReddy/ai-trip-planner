"""
AI Trip Planner — Streamlit Frontend
Powered by a LangGraph multi-agent pipeline with RAG and LangSmith observability.
"""

import streamlit as st
import httpx
import os
import json
from dotenv import load_dotenv
from components.charts import budget_donut, daily_cost_bar, budget_gauge

load_dotenv()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero { text-align:center; padding: 1.5rem 0 0.5rem; }
    .hero h1 { font-size:3rem; font-weight:800; background:linear-gradient(135deg,#7C3AED,#4ECDC4);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
    .hero p { color:#666; font-size:1.1rem; margin-top:0.3rem; }
    .agent-badge { display:inline-block; background:#ede9fe; color:#6d28d9;
                   border-radius:20px; padding:3px 12px; font-size:0.78rem;
                   font-weight:600; margin:2px; }
    .day-header { background:linear-gradient(90deg,#7C3AED15,transparent);
                  border-left:4px solid #7C3AED; padding:0.6rem 1rem;
                  border-radius:0 8px 8px 0; margin-bottom:0.8rem; }
    .time-card { background:#fafafa; border:1px solid #e5e7eb; border-radius:10px;
                 padding:1rem; height:100%; }
    .meal-card { background:#f5f3ff; border-radius:8px; padding:0.7rem; margin-bottom:0.4rem; }
    .stat-pill { background:#7C3AED; color:white; border-radius:20px;
                 padding:4px 14px; font-size:0.85rem; font-weight:600; }
    .insight-box { background:#f0fdf4; border:1px solid #86efac;
                   border-radius:10px; padding:1rem; }
    .phrase-card { background:#fff7ed; border-radius:8px; padding:0.6rem 1rem;
                   border-left:3px solid #f97316; margin-bottom:0.4rem; }
</style>
""", unsafe_allow_html=True)


def call_backend(payload: dict) -> dict | None:
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{BACKEND}/generate-itinerary", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        st.error("Cannot reach backend. Start the FastAPI server: `uvicorn backend.main:app --port 8000`")
    except httpx.HTTPStatusError as e:
        st.error(f"Backend error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def render_day(day: dict, idx: int):
    with st.expander(
        f"**Day {day['day']}** — {day.get('theme', '')}  ·  "
        f"${day.get('daily_total', 0):,.0f}",
        expanded=(idx == 0),
    ):
        st.markdown(
            f'<div class="day-header"><strong>🗓 Day {day["day"]}:</strong> {day.get("theme","")}'
            f' &nbsp;·&nbsp; 🚌 {day.get("transport","")}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        for col, period, emoji, bg in [
            (c1, "morning", "🌅", "#fff9f0"),
            (c2, "afternoon", "☀️", "#f0fff9"),
            (c3, "evening", "🌙", "#f5f0ff"),
        ]:
            seg = day.get(period, {})
            with col:
                st.markdown(f"""
<div class="time-card" style="background:{bg}">
  <div style="font-weight:700;font-size:0.9rem;margin-bottom:0.4rem">{emoji} {period.title()}</div>
  <div style="font-weight:600;margin-bottom:0.3rem">{seg.get('activity','')}</div>
  <div style="font-size:0.85rem;color:#555;margin-bottom:0.5rem">{seg.get('description','')}</div>
  <div style="font-size:0.78rem;color:#888">📍 {seg.get('location','')} &nbsp;·&nbsp;
       ⏱ {seg.get('duration','')} &nbsp;·&nbsp; 💵 ${seg.get('cost',0)}</div>
  {'<div style="font-size:0.78rem;color:#7C3AED;margin-top:0.4rem">💡 ' + seg.get('tips','') + '</div>' if seg.get('tips') else ''}
</div>""", unsafe_allow_html=True)

        st.markdown("**🍽️ Meals**")
        m1, m2, m3 = st.columns(3)
        for col, meal, emoji in [(m1, "breakfast", "☕"), (m2, "lunch", "🥗"), (m3, "dinner", "🍽️")]:
            m = day.get("meals", {}).get(meal, {})
            with col:
                st.markdown(f"""
<div class="meal-card">
  <strong>{emoji} {meal.title()}</strong><br>
  <span style="font-size:0.88rem">{m.get('name','')}</span><br>
  <span style="font-size:0.78rem;color:#888">{m.get('cuisine','')} · ${m.get('cost',0)}</span><br>
  <span style="font-size:0.78rem;color:#555">{m.get('why','')}</span>
</div>""", unsafe_allow_html=True)

        if day.get("insider_tip"):
            st.info(f"🔑 **Insider Tip:** {day['insider_tip']}")


def render_itinerary(data: dict, budget: float, gen_ms: int, cached: bool):
    st.success(
        f"{'⚡ Cached response' if cached else '✅ Generated in ' + str(gen_ms // 1000) + 's'} · "
        f"**{data.get('destination', '')}** — {len(data.get('days', []))} days planned"
    )

    # Agent pipeline badge
    st.markdown(
        '<div style="margin-bottom:1rem">'
        '<span class="agent-badge">🔍 ResearchAgent</span>'
        '<span style="color:#ccc;margin:0 4px">→</span>'
        '<span class="agent-badge">📋 ItineraryAgent</span>'
        '<span style="color:#ccc;margin:0 4px">→</span>'
        '<span class="agent-badge">💰 BudgetAgent</span>'
        '<span style="color:#ccc;margin:0 4px">→</span>'
        '<span class="agent-badge">✅ ReviewAgent</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_summary, col_stats = st.columns([3, 1])
    with col_summary:
        st.markdown("### Trip Overview")
        st.write(data.get("summary", ""))
        st.markdown(f"🕐 **Best time:** {data.get('best_time_to_visit','')}")
        st.markdown(f"🚌 **Getting around:** {data.get('getting_around','')}")
    with col_stats:
        ba = data.get("budget_analysis", {})
        status = ba.get("budget_status", "")
        color = {"under_budget": "🟢", "on_budget": "🟡", "over_budget": "🔴"}.get(status, "⚪")
        st.metric("Budget Status", f"{color} {status.replace('_',' ').title()}")
        st.metric("Daily Average", f"${ba.get('daily_average', 0):,.0f}")
        variance = ba.get("variance_usd", 0)
        st.metric("Variance", f"${abs(variance):,.0f} {'under' if variance >= 0 else 'over'}")

    # Charts
    st.markdown("---")
    st.markdown("### Budget Analytics")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.plotly_chart(budget_donut(ba), use_container_width=True)
    with bc2:
        st.plotly_chart(daily_cost_bar(data.get("days", [])), use_container_width=True)
    with bc3:
        st.plotly_chart(budget_gauge(budget, ba.get("total_estimate", 0)), use_container_width=True)

    # Days
    st.markdown("---")
    st.markdown("### Day-by-Day Itinerary")
    for i, day in enumerate(data.get("days", [])):
        render_day(day, i)

    # Bottom info
    st.markdown("---")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("### 🎒 What to Pack")
        for item in data.get("packing_essentials", []):
            st.write(f"• {item}")

    with p2:
        st.markdown("### 🗣️ Useful Phrases")
        for p in data.get("local_phrases", []):
            st.markdown(f"""
<div class="phrase-card">
  <strong>{p.get('phrase','')}</strong>
  <span style="color:#888;font-size:0.8rem"> ({p.get('pronunciation','')})</span><br>
  <span style="font-size:0.88rem">{p.get('meaning','')}</span>
</div>""", unsafe_allow_html=True)

    with p3:
        st.markdown("### 💰 Save Money")
        for tip in ba.get("savings_tips", []):
            st.write(f"• {tip}")
        if ba.get("splurge_picks"):
            st.markdown("**Worth splurging on:**")
            for s in ba.get("splurge_picks", []):
                st.write(f"⭐ {s}")

    ec = data.get("emergency_info", {})
    if ec:
        with st.expander("🆘 Emergency Information"):
            for k, v in ec.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")

    if data.get("research_insights"):
        with st.expander("🔍 Research Insights (RAG Context)"):
            st.write(data["research_insights"])


def main():
    st.markdown("""
<div class="hero">
  <h1>✈️ AI Trip Planner</h1>
  <p>Multi-agent itinerary generation · RAG-enhanced research · LangSmith observability</p>
</div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Plan Your Trip")
        st.markdown("Powered by 4 specialized AI agents working in sequence.")
        st.markdown("---")

        destination = st.text_input("🌍 Destination", placeholder="e.g., Kyoto, Japan")

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input("💵 Budget (USD)", min_value=200, max_value=50000, value=2000, step=100)
        with c2:
            duration = st.slider("📅 Days", 1, 21, 7)

        travel_style = st.selectbox(
            "🎒 Travel Style",
            ["budget", "balanced", "comfort"],
            index=1,
            format_func=lambda x: {"budget": "🎒 Budget", "balanced": "⚖️ Balanced", "comfort": "✨ Comfort"}[x],
        )

        interests = st.multiselect(
            "❤️ Interests",
            ["Culture & History", "Food & Cuisine", "Nature & Outdoors",
             "Arts & Entertainment", "Adventure & Sports", "Photography",
             "Music & Nightlife", "Wellness & Spa", "Shopping", "Architecture"],
            default=["Culture & History", "Food & Cuisine"],
        )

        go = st.button(
            "🚀 Generate Itinerary",
            type="primary",
            use_container_width=True,
            disabled=not destination or not interests,
        )

        st.markdown("---")
        st.markdown("**Agent Pipeline:**")
        st.markdown("1. 🔍 ResearchAgent (RAG)")
        st.markdown("2. 📋 ItineraryAgent")
        st.markdown("3. 💰 BudgetAgent")
        st.markdown("4. ✅ ReviewAgent")

    if go:
        with st.spinner("Running 4-agent pipeline... (Research → Plan → Budget → Review)"):
            result = call_backend({
                "destination": destination.strip(),
                "budget": budget,
                "duration": duration,
                "interests": interests,
                "travel_style": travel_style,
            })
        if result:
            st.session_state["result"] = result
            st.session_state["budget"] = budget
            render_itinerary(result.get("itinerary", result), budget, result.get("generation_time_ms", 0), result.get("cached", False))
    elif "result" in st.session_state:
        result = st.session_state["result"]
        render_itinerary(result.get("itinerary", result), st.session_state.get("budget", 0), result.get("generation_time_ms", 0), result.get("cached", False))
    else:
        st.markdown("---")
        cols = st.columns(4)
        features = [
            ("🔍", "RAG Research", "ChromaDB vector search enriches every itinerary"),
            ("🤖", "Multi-Agent", "4 specialized agents — research, plan, budget, review"),
            ("📊", "Analytics", "Budget breakdown, daily cost chart, variance gauge"),
            ("🔭", "Observable", "Every agent call traced in LangSmith"),
        ]
        for col, (icon, title, desc) in zip(cols, features):
            with col:
                st.markdown(f"### {icon} {title}")
                st.write(desc)


if __name__ == "__main__":
    main()
