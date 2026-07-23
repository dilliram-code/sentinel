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
    padding-top:1rem;
}

html, body, [class*="css"]{
    font-family: 'Segoe UI';
    background-color: #0B0C3B;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background-color:#0B0C1B;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* KPI Cards */

.metric-card{
    background-color:white;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,.08);
}

.metric-title{
    font-size:16px;
    color:gray;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
}

/* Dashboard Title */

.title{
    font-size:36px;
    font-weight:bold;
}

.subtitle{
    color:gray;
    margin-top:-10px;
    margin-bottom:25px;
}

/* Streamlit Tabs Customization */
button[data-baseweb="tab"] {
    margin-right: 35px !important; /* Increased space between tabs */
    padding-left: 10px !important;
    padding-right: 10px !important;
    background-color: transparent !important;
}

/* Targeting the text inside the tabs to increase font size */
button[data-baseweb="tab"] p {
    font-size: 35px !important; /* Increased font size */
    font-weight: bold !important; /* Increased font weight */
}

/* Footer */

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
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
            margin=dict(l=20, r=20, t=40, b=20)
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