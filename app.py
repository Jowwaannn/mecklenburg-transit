import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
import folium
from streamlit_folium import st_folium
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transit Access & Economic Opportunity — Mecklenburg County",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette — matches presentation exactly ─────────────────────────────────────
NAVY    = "#2D3554"
TEAL    = "#2A8C8C"
TEAL_LT = "#3AADAD"
GOLD    = "#F0A500"
GREY    = "#E8ECF0"
GREY_MD = "#C8D0DC"
GREY_DK = "#64748B"
WHITE   = "#FFFFFF"
CORAL   = "#C0392B"   # muted, gap tracts only

# ── CSS — light mode, presentation palette ─────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {WHITE};
    color: {NAVY};
}}

/* App background */
[data-testid="stAppViewContainer"] {{ background: #F7F9FC; }}
[data-testid="stHeader"] {{
    background: {WHITE};
    border-bottom: 1px solid {GREY_MD};
}}

/* Sidebar — light, minimal */
[data-testid="stSidebar"] {{
    background: {WHITE};
    border-right: 1px solid {GREY_MD};
}}
[data-testid="stSidebar"] * {{ color: {NAVY} !important; }}
[data-testid="stSidebar"] label {{
    font-size: 11px !important;
    font-weight: 500 !important;
    color: {GREY_DK} !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* KPI cards */
.kpi-card {{
    background: {WHITE};
    border: 1px solid {GREY_MD};
    border-top: 3px solid {TEAL};
    border-radius: 8px;
    padding: 14px 16px;
}}
.kpi-label {{
    font-size: 10px;
    color: {GREY_DK};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 600;
    color: {NAVY};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px;
    color: {GREY_DK};
    margin-top: 3px;
}}
.kpi-gold {{ border-top-color: {GOLD}; }}
.kpi-teal {{ border-top-color: {TEAL}; }}
.kpi-navy {{ border-top-color: {NAVY}; }}

/* Section headers */
.section-header {{
    font-size: 11px;
    font-weight: 600;
    color: {GREY_DK};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding-bottom: 6px;
    border-bottom: 1px solid {GREY};
    margin-bottom: 12px;
    margin-top: 4px;
}}

/* Story banner */
.story-banner {{
    background: {NAVY};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
}}
.story-banner-title {{
    font-size: 15px;
    font-weight: 600;
    color: {WHITE};
    margin-bottom: 4px;
}}
.story-banner-sub {{
    font-size: 12px;
    color: {TEAL_LT};
}}

/* Map legend pill */
.legend-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}}

/* Streamlit metric override */
div[data-testid="stMetric"] {{
    background: {WHITE};
    border: 1px solid {GREY_MD};
    border-top: 3px solid {TEAL};
    border-radius: 8px;
    padding: 12px 14px;
}}
div[data-testid="stMetric"] label {{
    color: {GREY_DK} !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

/* Checkbox styling */
[data-testid="stCheckbox"] label {{
    font-size: 12px !important;
    color: {NAVY} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(__file__)

@st.cache_data
def load_data():
    tracts = gpd.read_file(os.path.join(DATA_DIR, "meck_tracts_need.geojson"))
    routes = gpd.read_file(os.path.join(DATA_DIR, "cats_routes_labeled.geojson"))
    silver = gpd.read_file(os.path.join(DATA_DIR, "LYNX_Silver_Line_Stations_Proposed.geojson"))

    tracts = tracts.to_crs("EPSG:4326")
    routes = routes.to_crs("EPSG:4326")
    silver = silver.to_crs("EPSG:4326")

    tracts["constraint_pct"] = (tracts["pct_no_or_one_vehicle"] * 100).round(1)
    tracts["no_vehicle_pct"] = (tracts["pct_no_vehicle"] * 100).round(1)

    tracts["constraint_quartile"] = pd.qcut(
        tracts["pct_no_vehicle"], 4,
        labels=["Low", "Low-Mid", "High-Mid", "High"]
    )

    threshold = tracts["pct_no_vehicle"].quantile(0.85)
    tracts["priority"] = tracts["pct_no_vehicle"] >= threshold

    # Silver line proximity
    silver_proj = silver.to_crs("EPSG:2264")
    tracts_proj = tracts.to_crs("EPSG:2264")
    silver_buffer = silver_proj.geometry.buffer(2640).union_all()
    tracts_proj["near_silver"] = tracts_proj.geometry.intersects(silver_buffer)
    tracts["near_silver"] = tracts_proj["near_silver"].values

    # Bus corridor proximity for 5 priority routes
    routes_proj = routes.to_crs("EPSG:2264")
    tracts_proj2 = tracts.to_crs("EPSG:2264")
    for rid in ["35", "7", "3", "11", "21"]:
        r = routes_proj[routes_proj["route_short_name"] == rid]
        if not r.empty:
            rbuf = r.geometry.buffer(2640).union_all()
            tracts_proj2[f"near_rt{rid}"] = tracts_proj2.geometry.intersects(rbuf)
            tracts[f"near_rt{rid}"] = tracts_proj2[f"near_rt{rid}"].values
        else:
            tracts[f"near_rt{rid}"] = False

    return tracts, routes, silver

tracts, routes, silver = load_data()

# ── Derived county-level stats ─────────────────────────────────────────────────
total_hh      = tracts["total_households"].sum()
zero_veh_hh   = tracts["no_vehicle_households"].sum()
one_veh_hh    = tracts["one_vehicle_households"].sum()
constrained_hh = zero_veh_hh + one_veh_hh
priority_n    = int(tracts["priority"].sum())
priority_hh   = tracts[tracts["priority"]]["no_vehicle_households"].sum()
gap_tracts    = int((tracts["priority"] & ~tracts["near_silver"]).sum())
gap_hh        = tracts[tracts["priority"] & ~tracts["near_silver"]]["no_vehicle_households"].sum()

# ── Population growth data (from slide 6) ─────────────────────────────────────
YEARS_OBS = np.array([1970,1980,1990,2000,2010,2020,2025])
POP_OBS   = np.array([0.354,0.404,0.511,0.695,0.919,1.115,1.233])
COEFFS    = np.polyfit(YEARS_OBS, POP_OBS, 2)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 12px;'>
        <div style='font-size:10px;color:{GREY_DK};text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;'>Dashboard</div>
        <div style='font-size:16px;font-weight:600;color:{NAVY};line-height:1.35;'>
            Transit Access<br>Mecklenburg Co.
        </div>
        <div style='font-size:10px;color:{GREY_DK};margin-top:4px;'>DSBA 5122 · Group 6</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"<div style='font-size:10px;font-weight:600;color:{GREY_DK};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>Map layers</div>", unsafe_allow_html=True)
    show_priority  = st.checkbox("Highlight priority tracts",   value=True)
    show_silver    = st.checkbox("Silver Line stations",         value=True)
    show_bus       = st.checkbox("5 priority bus corridors",     value=True)
    show_blue      = st.checkbox("Existing Blue Line",           value=True)

    st.divider()

    st.markdown(f"<div style='font-size:10px;font-weight:600;color:{GREY_DK};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>Growth projection</div>", unsafe_allow_html=True)
    target_year = st.slider("Target year", 2026, 2050, 2040, step=1)

    st.divider()
    st.markdown(f"""
    <div style='font-size:10px;color:{GREY_DK};line-height:1.7;'>
        <b>Data sources</b><br>
        ACS 5-yr 2024 · B25044, B19013<br>
        CATS GTFS · Feb 2026<br>
        LYNX Silver Line GIS · Charlotte DOT<br>
        US Census TIGER 2024
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='story-banner'>
    <div class='story-banner-title'>
        Transit Access, Mobility Constraints & Economic Opportunity — Mecklenburg County
    </div>
    <div class='story-banner-sub'>
        44.8% of households have ≤1 vehicle · 46 priority tracts identified ·
        Silver Line reaches 14 · 5 bus corridors fill the remaining gap
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class='kpi-card kpi-teal'>
        <div class='kpi-label'>Households ≤1 vehicle</div>
        <div class='kpi-value'>{constrained_hh/total_hh*100:.1f}%</div>
        <div class='kpi-sub'>{constrained_hh:,.0f} of {total_hh:,.0f} households</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card kpi-navy'>
        <div class='kpi-label'>Zero-vehicle households</div>
        <div class='kpi-value'>{zero_veh_hh:,.0f}</div>
        <div class='kpi-sub'>{zero_veh_hh/total_hh*100:.1f}% of all households</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card kpi-gold'>
        <div class='kpi-label'>High-need priority tracts</div>
        <div class='kpi-value'>{priority_n}</div>
        <div class='kpi-sub'>{priority_hh:,.0f} zero-vehicle HH inside</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card kpi-teal'>
        <div class='kpi-label'>Silver Line gap</div>
        <div class='kpi-value'>{gap_tracts} tracts</div>
        <div class='kpi-sub'>{gap_hh:,.0f} zero-vehicle HH unserved</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THREE-COLUMN LAYOUT
# LEFT (55%): Interactive map
# RIGHT TOP (45%): Analytical technique — regression + equity
# RIGHT BOTTOM (45%): Growth projection
# ══════════════════════════════════════════════════════════════════════════════
col_map, col_right = st.columns([1.1, 0.9])

# ── LEFT: INTERACTIVE MAP ─────────────────────────────────────────────────────
with col_map:
    st.markdown("<div class='section-header'>Interactive map — vehicle constraint by census tract</div>", unsafe_allow_html=True)

    # Build Folium map
    m = folium.Map(
        location=[35.22, -80.84],
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Choropleth: no-vehicle % by tract
    import json
    geojson_data = json.loads(tracts.to_json())

    folium.Choropleth(
        geo_data=geojson_data,
        data=tracts,
        columns=["GEOID", "no_vehicle_pct"],
        key_on="feature.properties.GEOID",
        fill_color="YlOrRd",
        fill_opacity=0.65,
        line_opacity=0.3,
        line_color="white",
        legend_name="% Zero-vehicle households",
        nan_fill_color="lightgray",
        highlight=True,
    ).add_to(m)

    # Priority tract outlines
    if show_priority:
        priority_tracts = tracts[tracts["priority"]]
        folium.GeoJson(
            json.loads(priority_tracts.to_json()),
            style_function=lambda f: {
                "fillColor":   "transparent",
                "color":       TEAL,
                "weight":      2.5,
                "dashArray":   "4 2",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME", "no_vehicle_pct", "total_households"],
                aliases=["Tract:", "% Zero-vehicle:", "Total HH:"],
                localize=True,
            ),
            name="Priority tracts",
        ).add_to(m)

    # Existing Blue Line
    if show_blue:
        blue = routes[routes["route_short_name"] == "501"]
        if not blue.empty:
            folium.GeoJson(
                json.loads(blue.to_json()),
                style_function=lambda f: {
                    "color": NAVY, "weight": 3, "opacity": 0.7, "dashArray": "6 3"
                },
                name="Existing Blue Line",
            ).add_to(m)

    # 5 priority bus corridors
    if show_bus:
        BUS_COLORS = {
            "35": "#1A6B6B",
            "7":  "#2A8C8C",
            "3":  "#3AADAD",
            "11": "#4DCECC",
            "21": "#0F4F4F",
        }
        BUS_NAMES = {
            "35": "Rt 35 · Wilkinson-Amazon",
            "7":  "Rt 7 · Beatties Ford",
            "3":  "Rt 3 · The Plaza",
            "11": "Rt 11 · North Tryon",
            "21": "Rt 21 · Statesville Ave",
        }
        for rid, col in BUS_COLORS.items():
            r = routes[routes["route_short_name"] == rid]
            if not r.empty:
                folium.GeoJson(
                    json.loads(r.to_json()),
                    style_function=lambda f, c=col: {
                        "color": c, "weight": 3.5, "opacity": 0.85
                    },
                    tooltip=BUS_NAMES.get(rid, f"Route {rid}"),
                    name=BUS_NAMES.get(rid, f"Route {rid}"),
                ).add_to(m)

    # Silver Line stations
    if show_silver:
        for _, row in silver.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=NAVY,
                fill=True,
                fill_color=GOLD,
                fill_opacity=0.9,
                weight=1.5,
                tooltip=f"{row.get('Name','Station')} — {row.get('Phase','')}",
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, height=520, use_container_width=True)

    # Map legend
    st.markdown(f"""
    <div style='margin-top:6px;font-size:11px;color:{GREY_DK};'>
        <span class='legend-pill' style='background:{GREY};color:{NAVY};border:1px solid {GREY_MD};'>
            Yellow-Red = constraint level
        </span>
        <span class='legend-pill' style='background:#d4eeee;color:{TEAL};border:1px solid {TEAL};'>
            — — Teal border = priority tract
        </span>
        <span class='legend-pill' style='background:{GOLD};color:{NAVY};'>● Gold = Silver Line station</span>
        <span class='legend-pill' style='background:{TEAL};color:white;'>— Teal lines = 5 priority corridors</span>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT COLUMN ───────────────────────────────────────────────────────────────
with col_right:

    # ── TOP: ANALYTICAL TECHNIQUE — Regression + Composite Index ──────────────
    st.markdown("<div class='section-header'>Analytical techniques — regression &amp; composite priority index</div>", unsafe_allow_html=True)

    tab_reg, tab_equity = st.tabs(["Linear regression", "Equity analysis"])

    with tab_reg:
        # OLS: % ≤1 vehicle ~ % zero-vehicle
        df_sc = tracts[["no_vehicle_pct", "constraint_pct", "priority", "NAME"]].dropna()
        x = df_sc["no_vehicle_pct"].values
        y = df_sc["constraint_pct"].values
        m_coef, b_coef = np.polyfit(x, y, 1)
        ss_res = np.sum((y - (m_coef * x + b_coef))**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        x_line = np.linspace(x.min(), x.max(), 100)

        fig_reg = go.Figure()

        # Non-priority scatter
        mask = df_sc["priority"] == False
        fig_reg.add_trace(go.Scatter(
            x=df_sc[mask]["no_vehicle_pct"],
            y=df_sc[mask]["constraint_pct"],
            mode="markers",
            marker=dict(color=GREY_MD, size=5, opacity=0.7),
            name="Other tracts",
            text=df_sc[mask]["NAME"],
            hovertemplate="%{text}<br>Zero-vehicle: %{x:.1f}%<br>≤1 vehicle: %{y:.1f}%<extra></extra>",
        ))

        # Priority scatter
        mask2 = df_sc["priority"] == True
        fig_reg.add_trace(go.Scatter(
            x=df_sc[mask2]["no_vehicle_pct"],
            y=df_sc[mask2]["constraint_pct"],
            mode="markers",
            marker=dict(color=TEAL, size=8, opacity=0.9,
                        line=dict(color=NAVY, width=1)),
            name="Priority tracts (top 15%)",
            text=df_sc[mask2]["NAME"],
            hovertemplate="%{text}<br>Zero-vehicle: %{x:.1f}%<br>≤1 vehicle: %{y:.1f}%<extra>Priority</extra>",
        ))

        # Regression line
        fig_reg.add_trace(go.Scatter(
            x=x_line, y=m_coef * x_line + b_coef,
            mode="lines",
            line=dict(color=GOLD, width=2, dash="dot"),
            name=f"OLS trend  R²={r2:.2f}",
        ))

        fig_reg.update_layout(
            paper_bgcolor="white", plot_bgcolor="#F7F9FC",
            font=dict(family="Inter", color=NAVY, size=11),
            margin=dict(l=8, r=8, t=8, b=8),
            height=230,
            xaxis=dict(title="% Zero-vehicle HH", gridcolor=GREY,
                       ticksuffix="%", titlefont=dict(size=10)),
            yaxis=dict(title="% ≤1 vehicle HH", gridcolor=GREY,
                       ticksuffix="%", titlefont=dict(size=10)),
            legend=dict(font=dict(size=9), x=0.01, y=0.99,
                        bgcolor="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown(f"""
        <div style='background:{GREY};border-radius:6px;padding:8px 12px;
                    font-size:11px;color:{GREY_DK};line-height:1.6;'>
            <b style='color:{NAVY};'>OLS result:</b> R² = {r2:.2f} —
            zero-vehicle rate strongly predicts overall mobility constraint across all
            {len(df_sc)} Mecklenburg tracts. Priority tracts (teal) cluster at the
            upper end of both axes.
        </div>
        """, unsafe_allow_html=True)

    with tab_equity:
        # Stacked bar: vehicle distribution by income quartile
        import pandas as pd
        df_q = tracts.copy()
        df_q["constraint_quartile"] = pd.Categorical(
            df_q["constraint_quartile"],
            categories=["Low", "Low-Mid", "High-Mid", "High"], ordered=True
        )
        stk = df_q.groupby("constraint_quartile", observed=True).agg(
            no_v=("pct_no_vehicle", "mean"),
            one_v=("pct_one_vehicle", "mean"),
        ).reset_index()
        stk["two_plus"] = 1 - stk["no_v"] - stk["one_v"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Bar(
            name="Zero vehicle",
            x=stk["constraint_quartile"],
            y=(stk["no_v"]*100).round(1),
            marker_color=CORAL,
            text=(stk["no_v"]*100).round(1).astype(str)+"%",
            textposition="inside", textfont=dict(color="white", size=9),
        ))
        fig_eq.add_trace(go.Bar(
            name="One vehicle",
            x=stk["constraint_quartile"],
            y=(stk["one_v"]*100).round(1),
            marker_color=TEAL,
            text=(stk["one_v"]*100).round(1).astype(str)+"%",
            textposition="inside", textfont=dict(color="white", size=9),
        ))
        fig_eq.add_trace(go.Bar(
            name="2+ vehicles",
            x=stk["constraint_quartile"],
            y=(stk["two_plus"]*100).round(1),
            marker_color=GREY_MD,
        ))
        fig_eq.update_layout(
            barmode="stack",
            paper_bgcolor="white", plot_bgcolor="#F7F9FC",
            font=dict(family="Inter", color=NAVY, size=11),
            margin=dict(l=8, r=8, t=8, b=8),
            height=230,
            yaxis=dict(ticksuffix="%", gridcolor=GREY, title="Share of households",
                       titlefont=dict(size=10)),
            xaxis=dict(title="Income constraint quartile", titlefont=dict(size=10)),
            legend=dict(font=dict(size=9), orientation="h", y=-0.22),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        lo = stk[stk["constraint_quartile"]=="High"]["no_v"].values[0]*100
        hi = stk[stk["constraint_quartile"]=="Low"]["no_v"].values[0]*100
        st.markdown(f"""
        <div style='background:{GREY};border-radius:6px;padding:8px 12px;
                    font-size:11px;color:{GREY_DK};line-height:1.6;'>
            <b style='color:{NAVY};'>Composite index finding:</b>
            Highest-constraint quartile averages {lo:.1f}% zero-vehicle households
            vs {hi:.1f}% in the lowest — a {lo/hi:.1f}× gap. The priority index
            combines vehicle constraint, income vulnerability, rail access, and
            bus coverage to identify the 46 most underserved tracts.
        </div>
        """, unsafe_allow_html=True)

    # ── BOTTOM: GROWTH PROJECTION ──────────────────────────────────────────────
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Population growth projection — the urgency case</div>", unsafe_allow_html=True)

    years_proj = np.arange(2025, target_year + 1)
    pop_proj   = np.polyval(COEFFS, years_proj)
    pop_target = float(np.polyval(COEFFS, target_year))
    pop_2025   = float(np.polyval(COEFFS, 2025))
    delta_pop  = pop_target - pop_2025
    # Estimate households needing transit (roughly 44.8% of projected HH, avg 2.5 ppl/HH)
    est_hh_needing = (pop_target * 1_000_000 / 2.5) * 0.448

    fig_pop = go.Figure()

    # Observed
    fig_pop.add_trace(go.Scatter(
        x=YEARS_OBS, y=POP_OBS,
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=5, color=NAVY),
        name="Observed",
    ))

    # Projection
    fig_pop.add_trace(go.Scatter(
        x=years_proj, y=pop_proj,
        mode="lines",
        line=dict(color=CORAL, width=2.5, dash="dash"),
        name="Projected (quadratic fit)",
        fill="tonexty" if False else None,
    ))

    # Target year marker
    fig_pop.add_vline(
        x=target_year, line_dash="dot",
        line_color=GOLD, line_width=1.5,
    )
    fig_pop.add_annotation(
        x=target_year, y=pop_target,
        text=f"  {target_year}: {pop_target:.2f}M",
        showarrow=False, xanchor="left",
        font=dict(color=GOLD, size=10, family="Inter"),
    )

    fig_pop.update_layout(
        paper_bgcolor="white", plot_bgcolor="#F7F9FC",
        font=dict(family="Inter", color=NAVY, size=10),
        margin=dict(l=8, r=8, t=8, b=8),
        height=195,
        xaxis=dict(gridcolor=GREY, title="Year", titlefont=dict(size=10)),
        yaxis=dict(gridcolor=GREY, ticksuffix="M",
                   title="Population (millions)", titlefont=dict(size=10)),
        legend=dict(font=dict(size=9), x=0.01, y=0.99,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig_pop, use_container_width=True)

    # Projection callout — two metrics side by side
    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            label=f"Projected population by {target_year}",
            value=f"{pop_target:.2f}M",
            delta=f"+{delta_pop*1000:.0f}K vs today",
        )
    with m2:
        st.metric(
            label=f"Est. HH needing transit access",
            value=f"{est_hh_needing:,.0f}",
            delta="based on current 44.8% constraint rate",
            delta_color="off",
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;font-size:10px;color:{GREY_DK};padding:4px 0 8px;'>
    DSBA 5122 · Group 6 · Transit Access &amp; Economic Opportunity in Mecklenburg County ·
    Data: ACS 2024, CATS GTFS Feb 2026, LYNX Silver Line GIS, US Census TIGER 2024
</div>
""", unsafe_allow_html=True)
