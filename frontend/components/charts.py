import plotly.graph_objects as go
import plotly.express as px


def budget_donut(breakdown: dict) -> go.Figure:
    labels = ["Accommodation", "Food", "Activities", "Transport", "Misc"]
    values = [
        breakdown.get("accommodation_estimate", 0),
        breakdown.get("food_estimate", 0),
        breakdown.get("activities_estimate", 0),
        breakdown.get("transport_estimate", 0),
        breakdown.get("miscellaneous", 0),
    ]
    colors = ["#7C3AED", "#4ECDC4", "#FF6B6B", "#FFD93D", "#6BCB77"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:.0f}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"${sum(values):,.0f}<br><span style='font-size:10px'>estimated</span>",
            x=0.5, y=0.5, font_size=16, showarrow=False,
        )],
    )
    return fig


def daily_cost_bar(days: list) -> go.Figure:
    day_nums = [f"Day {d['day']}" for d in days]
    totals = [d.get("daily_total", 0) for d in days]
    themes = [d.get("theme", "") for d in days]

    fig = go.Figure(go.Bar(
        x=day_nums,
        y=totals,
        text=[f"${t:.0f}" for t in totals],
        textposition="outside",
        customdata=themes,
        hovertemplate="<b>%{x}</b><br>%{customdata}<br>Cost: $%{y:.0f}<extra></extra>",
        marker_color="#7C3AED",
        marker_line_color="#5B21B6",
        marker_line_width=1,
    ))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Cost (USD)",
        margin=dict(t=20, b=20, l=20, r=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def budget_gauge(budget: float, estimated: float) -> go.Figure:
    pct = min((estimated / budget) * 100, 150) if budget > 0 else 0
    color = "#6BCB77" if pct <= 90 else "#FFD93D" if pct <= 110 else "#FF6B6B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=estimated,
        delta={"reference": budget, "valueformat": "$.0f"},
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, budget * 1.5], "tickprefix": "$"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, budget * 0.9], "color": "#e8f5e9"},
                {"range": [budget * 0.9, budget * 1.1], "color": "#fff3e0"},
                {"range": [budget * 1.1, budget * 1.5], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#7C3AED", "width": 3},
                "thickness": 0.75,
                "value": budget,
            },
        },
        title={"text": "Estimated vs Budget"},
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
