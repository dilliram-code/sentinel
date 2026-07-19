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

from config import settings
from database import db_manager

st.set_page_config(page_title="Campus Stakeholder Detection Dashboard",
                page_icon="🎓", 
                layout="wide",     
                initial_sidebar_state="expanded"
)
# CUSTOM CSS
##############################################################

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
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background-color:#0F172A;
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


# Header + headline metrics
# ---------------------------------------------------------------------------
st.title("🎓 Campus Stakeholder Detection Dashboard")
st.caption("Real-Time Stakeholder Identification & Unknown Person Registration")

stats = db_manager.get_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registered Stakeholders", stats["stakeholders"])
c2.metric("Total Visits Logged", stats["visits"])
c3.metric("Visits Today", stats["visits_today"])
c4.metric("Unknown Persons", stats["unknowns"])

tab_live, tab_visits, tab_unknown, tab_people, tab_reports = st.tabs(
    ["📡 Live Monitor", "🧾 Visit Logs", "❓ Unknown Persons",
     "👥 Stakeholders", "📊 Reports"])


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
        st.info("No live frame yet. Start the pipeline with:  "
                "`python main.py run --source 0 --location \"Main Gate\"`")

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
    st.caption("Enroll new stakeholders with:  "
               "`python main.py register --uid S001 --name \"Full Name\" "
               "--role Student --images path/to/photos/`")

# Reports / analytics
# ---------------------------------------------------------------------------
with tab_reports:
    st.subheader("Activity overview")
    df = visits_dataframe(5000)
    if df.empty:
        st.info("No visit data yet.")
    else:
        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
        left, right = st.columns(2)
        with left:
            st.markdown("**Visits per day**")
            st.bar_chart(df.groupby("Date").size())
        with right:
            st.markdown("**Visits per camera location**")
            st.bar_chart(df.groupby("Camera Location").size())
        st.markdown("**Visits per role**")
        st.bar_chart(df.groupby("Role").size())

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