"""
dashboard/app.py
----------------
Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys
from datetime import datetime

# Make project-root imports work when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import plotly.express as px

from config import settings
from database import db_manager

st.set_page_config(page_title="Campus Stakeholder Detection Dashboard",
                page_icon="🎓", 
                layout="wide",     
                initial_sidebar_state="expanded"
)
# CUSTOM CSS
##############################################################

# supress the default layout of streamlit
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

#MainMenu {
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

.block-container{
    padding-top:1.2rem;
    padding-bottom:2rem;
}

html, body, [class*="css"]{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

*{
    scrollbar-width: thin;
    scrollbar-color: rgba(124, 58, 237, 0.5) transparent;
}

/* Respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce){
    *{
        transition: none !important;
        animation: none !important;
    }
}

/* Visible keyboard focus for accessibility */
button:focus-visible, input:focus-visible, [tabindex]:focus-visible{
    outline: 2px solid #A5B4FC !important;
    outline-offset: 2px;
}

/* Full app gradient background (fixed, covers scroll area) + soft glow blobs */
.stApp{
    background:
    radial-gradient(circle at 15% 10%, rgba(124, 58, 237, 0.35) 0%, rgba(124, 58, 237, 0) 45%),
    radial-gradient(circle at 85% 0%, rgba(99, 102, 241, 0.30) 0%, rgba(99, 102, 241, 0) 40%),
    radial-gradient(circle at 90% 85%, rgba(165, 180, 252, 0.18) 0%, rgba(165, 180, 252, 0) 45%),
    linear-gradient(
    135deg,
    #020617 0%,
    #172554 30%,
    #4C1D95 65%,
    #7C3ABD 100%
);
    background-attachment: fixed;
    background-size: cover;
}

/* Make default text readable on the dark gradient */
.stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption {
    color: #E5E7EB;
}

::selection{
    background: rgba(124, 58, 237, 0.45);
    color: #FFFFFF;
}

/* Sidebar — glassmorphism panel instead of a flat gradient fill */
section[data-testid="stSidebar"]{
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(22px) saturate(140%);
    -webkit-backdrop-filter: blur(22px) saturate(140%);
    border-right: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow: 8px 0 32px rgba(2, 6, 23, 0.35);
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Dashboard Title */

h1 {
    background: linear-gradient(90deg, #C4B5FD, #7C3AED, #A5B4FC);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Poppins', sans-serif;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    filter: drop-shadow(0 2px 18px rgba(124, 58, 237, 0.35));
}

.title{
    font-size:36px;
    font-weight:bold;
}

.subtitle{
    color:#C7C9D9;
    margin-top:-10px;
    margin-bottom:25px;
}

[data-testid="stCaptionContainer"] p{
    color: #A5B4FC !important;
}

/* ---------------- KPI Metric Cards (glassmorphism) ---------------- */

[data-testid="stMetric"]{
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 20px;
    padding: 18px 16px 14px 16px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.40), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before{
    content:"";
    position:absolute;
    top:0; left:0; right:0;
    height:3px;
    background: linear-gradient(90deg, #7C3AED, #A5B4FC, #7C3AED);
    opacity: 0.9;
}

[data-testid="stMetric"]::after{
    content:"";
    position:absolute;
    top:-60%; left:-20%;
    width:60%; height:220%;
    background: linear-gradient(120deg, rgba(255,255,255,0.10), rgba(255,255,255,0) 60%);
    transform: rotate(20deg);
    pointer-events:none;
}

[data-testid="stMetric"]:hover{
    transform: translateY(-4px);
    border-color: rgba(124, 58, 237, 0.55);
    box-shadow: 0 14px 36px rgba(124, 58, 237, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.14);
}

[data-testid="stMetricLabel"] p{
    color: #C4B5FD !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"]{
    font-family: 'Poppins', sans-serif;
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}

.metric-card{
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(16px) saturate(150%);
    -webkit-backdrop-filter: blur(16px) saturate(150%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius:18px;
    padding:20px;
    text-align:center;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.10);
}

.metric-title{
    font-size:16px;
    color:#C4B5FD;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
    color:#FFFFFF;
}

/* ---------------- Streamlit Tabs Customization ---------------- */

div[data-baseweb="tab-list"]{
    gap: 6px;
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px) saturate(150%);
    -webkit-backdrop-filter: blur(18px) saturate(150%);
    padding: 6px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

button[data-baseweb="tab"] {
    margin-right: 4px !important;
    padding: 8px 18px !important;
    background-color: transparent !important;
    border-radius: 14px !important;
    transition: background 0.25s ease;
}

button[data-baseweb="tab"]:hover{
    background: rgba(124, 58, 237, 0.20) !important;
}

button[data-baseweb="tab"][aria-selected="true"]{
    background: linear-gradient(135deg, rgba(49, 46, 129, 0.85), rgba(124, 58, 237, 0.85)) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.50), inset 0 1px 0 rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

/* Targeting the text inside the tabs */
button[data-baseweb="tab"] p {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #D1D5DB !important;
}

button[data-baseweb="tab"][aria-selected="true"] p{
    color: #FFFFFF !important;
}

[data-baseweb="tab-highlight"]{
    display:none;
}

/* ---------------- Buttons (frosted glass, gradient sheen) ---------------- */

.stButton>button, .stDownloadButton>button{
    background: linear-gradient(135deg, rgba(49, 46, 129, 0.55), rgba(124, 58, 237, 0.55));
    backdrop-filter: blur(12px) saturate(160%);
    -webkit-backdrop-filter: blur(12px) saturate(160%);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.16);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover{
    transform: translateY(-2px);
    border-color: rgba(165, 180, 252, 0.55);
    box-shadow: 0 10px 26px rgba(124, 58, 237, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.20);
    color: white;
}

/* ---------------- Inputs ---------------- */

.stTextInput input, .stNumberInput input{
    background: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(12px) saturate(150%);
    -webkit-backdrop-filter: blur(12px) saturate(150%);
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextInput input:focus, .stNumberInput input:focus{
    border-color: rgba(165, 180, 252, 0.65) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25) !important;
}

.stCheckbox label p{
    color: #E5E7EB !important;
}

/* ---------------- Dataframes / tables ---------------- */

[data-testid="stDataFrame"]{
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px) saturate(150%);
    -webkit-backdrop-filter: blur(14px) saturate(150%);
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.30);
}

/* Images (captured face thumbnails, logo, live frame) get a glass frame */
[data-testid="stImage"] img{
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.35);
}

/* ---------------- Subheaders ---------------- */

h2, h3{
    font-family: 'Poppins', sans-serif;
    color: #EDE9FE !important;
}

hr{
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* ---------------- Footer (glass card) ---------------- */

.footer{
    text-align:center;
    color:#A5B4FC;
    margin-top:40px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px) saturate(150%);
    -webkit-backdrop-filter: blur(14px) saturate(150%);
    border-radius: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.25);
}

/* ---------------- Responsive tweaks for mobile ---------------- */

@media (max-width: 768px){
    .block-container{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0.6rem;
    }

    h1{
        font-size: 24px !important;
    }

    [data-testid="stCaptionContainer"] p{
        font-size: 12px !important;
    }

    [data-testid="stMetric"]{
        padding: 12px 10px 10px 10px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"]{
        font-size: 22px !important;
    }

    [data-testid="stMetricLabel"] p{
        font-size: 11px !important;
    }

    button[data-baseweb="tab"] p {
        font-size: 13px !important;
    }

    button[data-baseweb="tab"] {
        padding: 6px 10px !important;
    }

    div[data-baseweb="tab-list"]{
        flex-wrap: wrap;
    }
}

</style>
""", unsafe_allow_html=True)


# database initialization
db_manager.init_db()


# Helpers
# ---------------------------------------------------------------------------
def visits_dataframe(limit, name_filter=None):
    rows = db_manager.fetch_visits(limit=limit, name_filter=name_filter)
    return pd.DataFrame(rows, columns=[
        "Timestamp", "Name", "Role", "UID", "Camera Location", "Similarity"])


def unknowns_dataframe(limit, only_unverified):
    rows = db_manager.fetch_unknowns(limit=limit,
                                    only_unverified=only_unverified)
    return pd.DataFrame(rows, columns=[
        "ID", "Timestamp", "Camera Location", "Image", "Verified"])


def stakeholders_dataframe():
    rows = db_manager.list_stakeholders()
    return pd.DataFrame(rows, columns=[
        "ID", "UID", "Name", "Role", "Image", "Registered At"])


# Header + Logo + headline metrics
# ---------------------------------------------------------------------------

col1, col2 = st.columns([0.5, 9.5])

with col1:
    st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
    st.image("assets/logo.png", width=80)

with col2:
    st.title("Campus Stakeholder Detection Dashboard")
    st.caption("Real-Time Stakeholder Identification & Unknown Person Registration")
    
stats = db_manager.get_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registered Stakeholders", stats["stakeholders"])
c2.metric("Total Visits Logged", stats["visits"])
c3.metric("Visits Today", stats["visits_today"])
c4.metric("Unknown Persons", stats["unknowns"])

# Updated Tab Names and Emojis
tab_live, tab_visits, tab_unknown, tab_people, tab_reports = st.tabs(
    ["📡 Live Monitor", "🧾 Visit Logs", "🚨 Unknown Persons",
    "👩‍💻🧑‍💻 Stakeholders", "📊 Reports"])


# Live monitor — shows the latest annotated frame saved by the pipeline
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Latest processed frame")
    st.caption("Click refresh to update.")
    if st.button("🔄 Refresh"):
        st.rerun()
    if os.path.isfile(settings.LATEST_FRAME_PATH):
        st.image(settings.LATEST_FRAME_PATH, use_container_width=True)
    else:
        st.info("No live frame yet.")

# Visit logs
# ---------------------------------------------------------------------------
with tab_visits:
    st.subheader("Stakeholder visit history")
    colf, coln = st.columns([2, 1])
    name_filter = colf.text_input("Search by name", "")
    limit = coln.number_input("Rows", 10, 5000, 500, step=50)
    df = visits_dataframe(int(limit), name_filter or None)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.download_button("⬇️ Export CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "visit_logs.csv", "text/csv")

# Unknown persons review queue
# ---------------------------------------------------------------------------
with tab_unknown:
    st.subheader("Unknown person registrations")
    only_pending = st.checkbox("Show only unverified", value=False)
    udf = unknowns_dataframe(200, only_pending)

    if udf.empty:
        st.success("No unknown persons recorded. ✅")
    else:
        st.dataframe(
            udf.drop(columns=["Image"]),
            use_container_width=True, hide_index=True)

        st.markdown("#### Captured faces")
        cols = st.columns(4)
        for i, row in udf.head(24).iterrows():
            with cols[i % 4]:
                img = row["Image"]
                if img and os.path.isfile(img):
                    st.image(img, use_container_width=True)
                else:
                    st.write("*(image missing)*")
                st.caption(f"#{row['ID']} • {row['Timestamp']} • "
                        f"{row['Camera Location']}")
                if not row["Verified"]:
                    if st.button("Mark verified", key=f"verify_{row['ID']}"):
                        db_manager.mark_unknown_verified(int(row["ID"]))
                        st.rerun()
                else:
                    st.caption("✔ verified")

# Stakeholder registry
# ---------------------------------------------------------------------------
with tab_people:
    st.subheader("Registered stakeholders")
    sdf = stakeholders_dataframe()
    st.dataframe(sdf.drop(columns=["Image"]),
                use_container_width=True, hide_index=True)

# Reports / analytics
# ---------------------------------------------------------------------------
with tab_reports:
    st.subheader("Activity overview")
    df = visits_dataframe(5000)
    
    if df.empty:
        st.info("No visit data yet.")
    else:
        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
        
        # Prepare aggregated data
        daily_visits = df.groupby("Date").size().reset_index(name="Count")
        cam_visits = df.groupby("Camera Location").size().reset_index(name="Count")
        role_visits = df.groupby("Role").size().reset_index(name="Count")
        
        layout_config = dict(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True, rangemode="tozero", tickformat="d"), 
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(color="#E5E7EB"),
        )
        
        # Hide the Plotly toolbar menu for a cleaner dashboard look
        ui_config = {'displayModeBar': False}

        left, right = st.columns(2)
        
        with left:
            fig_day = px.bar(
                daily_visits, x="Date", y="Count",
                color="Count", color_continuous_scale="Sunsetdark",
                title="<b>Visits per Day</b>"
            )
            fig_day.update_traces(width=0.4) 
            
            # Hides the color bar (number indicator) completely
            fig_day.update_layout(
                **layout_config,
                coloraxis_showscale=False 
            )
            st.plotly_chart(fig_day, use_container_width=True, config=ui_config)
            
        with right:
            fig_cam = px.bar(
                cam_visits, x="Camera Location", y="Count",
                color="Camera Location", color_discrete_sequence=px.colors.qualitative.Vivid,
                title="<b>Visits per Camera Location</b>"
            )
            fig_cam.update_traces(width=0.4)
            fig_cam.update_layout(**layout_config, showlegend=False)
            st.plotly_chart(fig_cam, use_container_width=True, config=ui_config)
            
        # Dodgerblue color and 0.2 width to visually match the split-column charts above
        fig_role = px.bar(
            role_visits, x="Role", y="Count",
            color_discrete_sequence=["dodgerblue"],
            title="<b>Visits per Role</b>"
        )
        fig_role.update_traces(width=0.2)
        fig_role.update_layout(**layout_config)
        st.plotly_chart(fig_role, use_container_width=True, config=ui_config)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Trend line chart
        fig_trend = px.line(
            daily_visits, x="Date", y="Count",
            title="<b>Visitor Trend Over Time (Ups & Downs)</b>",
            markers=True, line_shape="spline"
        )
        
        fig_trend.update_traces(
            line=dict(color="#FF4B4B", width=4), 
            marker=dict(size=10, color="#1E88E5", symbol="circle", line=dict(width=2, color="white"))
        )
        fig_trend.update_layout(**layout_config)
        fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        
        st.plotly_chart(fig_trend, use_container_width=True, config=ui_config)

st.markdown(

    f"""
    <div class='footer'>
    🎓 Real-Time Campus Stakeholder Identification System
    <br>
    Developed using Streamlit | YOLOv8 | InsightFace | SQLite
    <br>
    {datetime.now().strftime("%d %B %Y")}
    </div>
    """,
    unsafe_allow_html=True
)