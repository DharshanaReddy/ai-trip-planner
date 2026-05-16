import streamlit as st
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header { text-align: center; color: #888; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .day-card {
        background: #f8f9fa; border-left: 4px solid #4ECDC4;
        padding: 1rem; margin: 0.5rem 0; border-radius: 6px;
    }
    .feature-box {
        background: #f0fffe; border: 1px solid #4ECDC4;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("OPENAI_API_KEY not found. Add it to .env or Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=key)


def generate_itinerary(destination: str, budget: float, duration: int, interests: list) -> dict:
    client = get_client()
    prompt = f"""Create a detailed day-by-day travel itinerary.

Destination: {destination}
Total Budget: ${budget:.0f} USD
Duration: {duration} days
Interests: {', '.join(interests)}

Return ONLY valid JSON with this structure:
{{
  "destination": "City, Country",
  "summary": "2-3 sentence trip overview",
  "total_estimated_cost": <number>,
  "days": [
    {{
      "day": <number>,
      "theme": "Day theme",
      "morning": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "afternoon": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "evening": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "meals": {{"breakfast": "suggestion ~$X", "lunch": "suggestion ~$X", "dinner": "suggestion ~$X"}},
      "daily_cost_estimate": <number>,
      "pro_tip": "One practical tip"
    }}
  ],
  "packing_essentials": ["item1", "item2", "item3", "item4", "item5"],
  "best_time_to_visit": "Month range and why",
  "getting_around": "Local transport advice",
  "money_saving_tips": ["tip1", "tip2", "tip3"],
  "emergency_contacts": {{
    "local_emergency": "number",
    "tourist_helpline": "number or advice",
    "nearest_hospital": "general advice"
  }}
}}

Keep all costs within the ${budget:.0f} total budget across {duration} days."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert travel planner creating personalized, "
                    "budget-conscious itineraries. Always return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=4000,
    )
    return json.loads(response.choices[0].message.content)


def display_itinerary(data: dict):
    st.success(f"Your personalized itinerary for **{data.get('destination')}** is ready!")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Trip Overview")
        st.write(data.get("summary", ""))
    with col2:
        st.metric("Estimated Total", f"${data.get('total_estimated_cost', 0):,.0f}")
        st.caption(f"Best time: {data.get('best_time_to_visit', '')}")

    st.markdown("---")
    st.markdown("### Day-by-Day Itinerary")

    for day in data.get("days", []):
        with st.expander(
            f"**Day {day['day']}** — {day.get('theme', '')}",
            expanded=(day["day"] == 1),
        ):
            cols = st.columns(3)
            periods = [
                ("morning", "🌅 Morning", cols[0]),
                ("afternoon", "☀️ Afternoon", cols[1]),
                ("evening", "🌙 Evening", cols[2]),
            ]
            for key, label, col in periods:
                with col:
                    seg = day.get(key, {})
                    st.markdown(f"**{label}**")
                    st.markdown(f"**{seg.get('activity', '')}**")
                    st.write(seg.get("description", ""))
                    st.caption(f"⏱ {seg.get('duration', '')}  |  💵 ${seg.get('cost', 0)}")

            meals = day.get("meals", {})
            st.markdown("**Meals**")
            mc = st.columns(3)
            for i, (meal, emoji) in enumerate(
                [("breakfast", "☕"), ("lunch", "🥗"), ("dinner", "🍽️")]
            ):
                with mc[i]:
                    st.caption(f"{emoji} {meal.title()}")
                    st.write(meals.get(meal, ""))

            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric(f"Day {day['day']} Cost", f"${day.get('daily_cost_estimate', 0):,.0f}")
            with c2:
                st.info(f"💡 {day.get('pro_tip', '')}")

    st.markdown("---")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("### Packing Essentials")
        for item in data.get("packing_essentials", []):
            st.write(f"• {item}")
    with p2:
        st.markdown("### Getting Around")
        st.write(data.get("getting_around", ""))
    with p3:
        st.markdown("### Money-Saving Tips")
        for tip in data.get("money_saving_tips", []):
            st.write(f"• {tip}")

    ec = data.get("emergency_contacts", {})
    if ec:
        with st.expander("Emergency Contacts"):
            for k, v in ec.items():
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")


def main():
    st.markdown('<h1 class="main-header">✈️ AI Trip Planner</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate personalized day-by-day travel itineraries powered by AI</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Plan Your Trip")

        destination = st.text_input(
            "Destination",
            placeholder="e.g., Tokyo, Japan",
        )

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input(
                "Budget (USD)", min_value=100, max_value=50000, value=1500, step=100
            )
        with c2:
            duration = st.slider("Days", min_value=1, max_value=30, value=5)

        all_interests = [
            "Culture & History",
            "Food & Cuisine",
            "Nature & Outdoors",
            "Arts & Entertainment",
            "Shopping",
            "Beach & Relaxation",
            "Adventure & Sports",
            "Photography",
            "Music & Nightlife",
            "Wellness & Spa",
        ]
        interests = st.multiselect(
            "Your Interests",
            all_interests,
            default=["Food & Cuisine", "Culture & History"],
        )

        go = st.button(
            "Generate Itinerary",
            type="primary",
            use_container_width=True,
            disabled=not destination or not interests,
        )

    if go:
        if not destination.strip():
            st.error("Please enter a destination.")
            return
        with st.spinner(f"Planning your {duration}-day trip to {destination}..."):
            try:
                result = generate_itinerary(destination.strip(), budget, duration, interests)
                st.session_state["itinerary"] = result
                display_itinerary(result)
            except Exception as exc:
                st.error(f"Failed to generate itinerary: {exc}")
                st.info("Check that your OPENAI_API_KEY is valid and has sufficient credits.")

    elif "itinerary" in st.session_state:
        display_itinerary(st.session_state["itinerary"])

    else:
        st.markdown("---")
        cols = st.columns(3)
        features = [
            ("🎯", "Personalized", "Tailored to your budget, duration, and interests"),
            ("📋", "Detailed", "Morning, afternoon & evening plans every day"),
            ("💰", "Budget-Aware", "Cost estimates for every activity and meal"),
        ]
        for col, (icon, title, desc) in zip(cols, features):
            with col:
                st.markdown(f'<div class="feature-box"><h3>{icon} {title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
