"""
dashboard/app.py
----------------

Run from the project root:

    streamlit run dashboard/app.py

Dashboard for:
    - Real-Time Campus Stakeholder Identification
    - Unknown Person Registration
    - Visit Logs
    - Stakeholder Registry
    - Reports

IMPORTANT:
The surveillance/AI pipeline should run separately.

The dashboard reads the latest annotated frame produced by:
    data/live/latest.jpg

The Live Monitor updates only when the user clicks the Refresh
button — there is no background auto-refresh.
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from config import settings
from database import db_manager


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Campus Stakeholder Detection Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800'
    '&family=Inter:wght@400;500;600&display=swap'
);


/* ============================================================
   GLOBAL
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

html,
body,
[class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

* {
    scrollbar-width: thin;
    scrollbar-color: rgba(124, 58, 237, 0.5) transparent;
}


/* ============================================================
   APP BACKGROUND
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124, 58, 237, 0.35) 0%,
            rgba(124, 58, 237, 0) 45%
        ),

        radial-gradient(
            circle at 85% 0%,
            rgba(99, 102, 241, 0.30) 0%,
            rgba(99, 102, 241, 0) 40%
        ),

        radial-gradient(
            circle at 90% 85%,
            rgba(165, 180, 252, 0.18) 0%,
            rgba(165, 180, 252, 0) 45%
        ),

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


/* ============================================================
   TEXT
   ============================================================ */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stMarkdown,
.stCaption {
    color: #E5E7EB;
}

h1 {
    background:
        linear-gradient(
            90deg,
            #C4B5FD,
            #7C3AED,
            #A5B4FC
        );

    -webkit-background-clip: text;
    background-clip: text;

    -webkit-text-fill-color: transparent;

    font-family: 'Poppins', sans-serif;

    font-weight: 800 !important;

    letter-spacing: -0.5px;

    filter:
        drop-shadow(
            0 2px 18px
            rgba(124, 58, 237, 0.35)
        );
}

h2,
h3 {
    font-family: 'Poppins', sans-serif;

    color: #EDE9FE !important;
}

hr {
    border-color:
        rgba(255, 255, 255, 0.15) !important;
}

[data-testid="stCaptionContainer"] p {
    color: #A5B4FC !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        rgba(15, 23, 42, 0.45);

    backdrop-filter:
        blur(22px) saturate(140%);

    -webkit-backdrop-filter:
        blur(22px) saturate(140%);

    border-right:
        1px solid rgba(255, 255, 255, 0.10);

    box-shadow:
        8px 0 32px
        rgba(2, 6, 23, 0.35);
}

section[data-testid="stSidebar"] * {
    color: white;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {

    background:
        rgba(255, 255, 255, 0.07);

    backdrop-filter:
        blur(18px) saturate(160%);

    -webkit-backdrop-filter:
        blur(18px) saturate(160%);

    border:
        1px solid rgba(255, 255, 255, 0.16);

    border-radius:
        20px;

    padding:
        18px 16px 14px 16px;

    box-shadow:
        0 8px 28px
        rgba(15, 23, 42, 0.40),

        inset 0 1px 0
        rgba(255, 255, 255, 0.12);

    position: relative;

    overflow: hidden;
}

[data-testid="stMetric"]::before {

    content: "";

    position: absolute;

    top: 0;
    left: 0;
    right: 0;

    height: 3px;

    background:
        linear-gradient(
            90deg,
            #7C3AED,
            #A5B4FC,
            #7C3AED
        );

    opacity: 0.9;
}

[data-testid="stMetricLabel"] p {

    color: #C4B5FD !important;

    font-size: 14px !important;

    font-weight: 600 !important;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {

    font-family:
        'Poppins',
        sans-serif;

    font-size:
        30px !important;

    font-weight:
        700 !important;

    color:
        #FFFFFF !important;
}


/* ============================================================
   TABS
   ============================================================ */

div[data-baseweb="tab-list"] {

    gap: 6px;

    background:
        rgba(255, 255, 255, 0.06);

    backdrop-filter:
        blur(18px) saturate(150%);

    -webkit-backdrop-filter:
        blur(18px) saturate(150%);

    padding: 6px;

    border-radius: 18px;

    border:
        1px solid rgba(255, 255, 255, 0.14);

    box-shadow:
        0 6px 20px
        rgba(15, 23, 42, 0.30);
}

button[data-baseweb="tab"] {

    margin-right: 4px !important;

    padding: 8px 18px !important;

    background-color:
        transparent !important;

    border-radius:
        14px !important;
}

button[data-baseweb="tab"]:hover {

    background:
        rgba(124, 58, 237, 0.20) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {

    background:
        linear-gradient(
            135deg,
            rgba(49, 46, 129, 0.85),
            rgba(124, 58, 237, 0.85)
        ) !important;

    box-shadow:
        0 4px 18px
        rgba(124, 58, 237, 0.50);

    border:
        1px solid rgba(255, 255, 255, 0.16);
}

button[data-baseweb="tab"] p {

    font-size: 16px !important;

    font-weight: 600 !important;

    color:
        #D1D5DB !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {

    color:
        #FFFFFF !important;
}

[data-baseweb="tab-highlight"] {
    display: none;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button,
.stDownloadButton > button {

    background:
        linear-gradient(
            135deg,
            rgba(49, 46, 129, 0.55),
            rgba(124, 58, 237, 0.55)
        );

    backdrop-filter:
        blur(12px) saturate(160%);

    -webkit-backdrop-filter:
        blur(12px) saturate(160%);

    color:
        white;

    border:
        1px solid rgba(255, 255, 255, 0.22);

    border-radius:
        12px;

    padding:
        8px 18px;

    font-weight:
        600;

    box-shadow:
        0 4px 16px
        rgba(124, 58, 237, 0.30);
}

.stButton > button:hover,
.stDownloadButton > button:hover {

    border-color:
        rgba(165, 180, 252, 0.55);

    box-shadow:
        0 10px 26px
        rgba(124, 58, 237, 0.45);

    color:
        white;
}


/* ============================================================
   INPUTS
   ============================================================ */

.stTextInput input,
.stNumberInput input {

    background:
        rgba(255, 255, 255, 0.07) !important;

    color:
        white !important;

    border-radius:
        12px !important;

    border:
        1px solid
        rgba(255, 255, 255, 0.20)
        !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {

    background:
        rgba(255, 255, 255, 0.04);

    border-radius:
        16px;

    overflow:
        hidden;

    border:
        1px solid
        rgba(255, 255, 255, 0.14);

    box-shadow:
        0 8px 24px
        rgba(15, 23, 42, 0.30);
}


/* ============================================================
   IMAGES
   ============================================================ */

[data-testid="stImage"] img {

    border-radius:
        14px;

    border:
        1px solid
        rgba(255, 255, 255, 0.14);

    box-shadow:
        0 6px 18px
        rgba(15, 23, 42, 0.35);
}


/* ============================================================
   LIVE MONITOR
   ============================================================ */

.live-monitor {

    background:
        rgba(255, 255, 255, 0.055);

    border:
        1px solid
        rgba(255, 255, 255, 0.14);

    border-radius:
        20px;

    padding:
        12px;

    box-shadow:
        0 10px 30px
        rgba(2, 6, 23, 0.35);

    backdrop-filter:
        blur(15px);

    -webkit-backdrop-filter:
        blur(15px);
}


/* ============================================================
   STATUS
   ============================================================ */

.status-online {

    display: inline-block;

    padding:
        5px 12px;

    border-radius:
        20px;

    background:
        rgba(34, 197, 94, 0.15);

    border:
        1px solid
        rgba(34, 197, 94, 0.35);

    color:
        #86EFAC;

    font-size:
        13px;

    font-weight:
        600;
}

.status-offline {

    display: inline-block;

    padding:
        5px 12px;

    border-radius:
        20px;

    background:
        rgba(239, 68, 68, 0.15);

    border:
        1px solid
        rgba(239, 68, 68, 0.35);

    color:
        #FCA5A5;

    font-size:
        13px;

    font-weight:
        600;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align:
        center;

    color:
        #A5B4FC;

    margin-top:
        40px;

    padding:
        16px;

    background:
        rgba(255, 255, 255, 0.05);

    backdrop-filter:
        blur(14px);

    -webkit-backdrop-filter:
        blur(14px);

    border-radius:
        16px;

    border-top:
        1px solid
        rgba(255, 255, 255, 0.14);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {

        padding-left:
            1rem !important;

        padding-right:
            1rem !important;

        padding-top:
            0.6rem;
    }

    h1 {
        font-size:
            24px !important;
    }

    [data-testid="stMetricValue"] {

        font-size:
            22px !important;
    }

    [data-testid="stMetricLabel"] p {

        font-size:
            11px !important;
    }

    button[data-baseweb="tab"] p {

        font-size:
            13px !important;
    }

    button[data-baseweb="tab"] {

        padding:
            6px 10px !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# DATABASE
# ============================================================================

settings.ensure_directories()

db_manager.init_db()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def visits_dataframe(
    limit,
    name_filter=None
):

    rows = db_manager.fetch_visits(
        limit=limit,
        name_filter=name_filter
    )

    return pd.DataFrame(
        rows,
        columns=[
            "Timestamp",
            "Name",
            "Role",
            "UID",
            "Camera Location",
            "Similarity",
        ]
    )


def unknowns_dataframe(
    limit,
    only_unverified
):

    rows = db_manager.fetch_unknowns(
        limit=limit,
        only_unverified=only_unverified
    )

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Timestamp",
            "Camera Location",
            "Image",
            "Verified",
        ]
    )


def stakeholders_dataframe():

    rows = db_manager.list_stakeholders()

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "UID",
            "Name",
            "Role",
            "Image",
            "Registered At",
        ]
    )


# ============================================================================
# READ LIVE FRAME SAFELY
# ============================================================================

def read_live_frame():

    path = settings.LATEST_FRAME_PATH

    if not os.path.isfile(path):

        return None

    try:

        # ----------------------------------------------------
        # Read binary data directly.
        #
        # This avoids Streamlit repeatedly resolving the
        # same filename as the image source.
        # ----------------------------------------------------

        with open(path, "rb") as file:

            data = file.read()

        if not data:

            return None

        return data

    except (
        OSError,
        IOError
    ):

        return None


# ============================================================================
# LIVE MONITOR
# ============================================================================

def render_live_monitor():

    """
    Render the live camera monitor.

    The frame only updates when the user clicks "Refresh" — no
    background polling, no auto-rerun.
    """

    st.subheader(
        "Live Camera Monitor"
    )

    st.caption(
        "Latest annotated frame from the surveillance pipeline. "
        "Click Refresh to pull the newest frame."
    )

    # --------------------------------------------------------
    # Refresh button
    # --------------------------------------------------------

    col_btn, col_status = st.columns(
        [1, 4]
    )

    with col_btn:

        st.button(
            "🔄 Refresh",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Persistent containers
    # --------------------------------------------------------

    status_placeholder = col_status.empty()

    image_placeholder = st.empty()

    time_placeholder = st.empty()

    # --------------------------------------------------------
    # Read frame
    #
    # This function body only re-executes when the user clicks
    # the Refresh button (or on any other rerun-triggering
    # interaction elsewhere on the page) — never on a timer.
    # --------------------------------------------------------

    image_data = read_live_frame()

    if image_data is None:

        status_placeholder.markdown(
            '<span class="status-offline">'
            '● Camera feed unavailable'
            '</span>',
            unsafe_allow_html=True
        )

        image_placeholder.info(
            "Waiting for the surveillance pipeline..."
        )

        return

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    image_placeholder.image(
        image_data,
        width='content'
    )

    # --------------------------------------------------------
    # Last update time
    # --------------------------------------------------------

    try:

        modified = os.path.getmtime(
            settings.LATEST_FRAME_PATH
        )

        modified_time = datetime.fromtimestamp(
            modified
        ).strftime(
            "%H:%M:%S"
        )

        time_placeholder.caption(
            f"Last frame update: {modified_time}"
        )

    except OSError:

        pass


# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns(
    [0.5, 9.5]
)


with col1:

    st.markdown(
        "<div style='padding-top:18px'></div>",
        unsafe_allow_html=True
    )

    logo_path = os.path.join(
        PROJECT_ROOT,
        "assets",
        "logo.png"
    )

    if os.path.isfile(logo_path):

        st.image(
            logo_path,
            width=80
        )


with col2:

    st.title(
        "Campus Stakeholder Detection Dashboard"
    )

    st.caption(
        "Real-Time Stakeholder Identification & "
        "Unknown Person Registration"
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        "## 🎓 System"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "**Camera**"
    )

    st.code(
        str(settings.DEFAULT_SOURCE)
    )

    st.markdown(
        "**Location**"
    )

    st.write(
        settings.DEFAULT_CAMERA_LOCATION
    )

    st.markdown(
        "**AI processing**"
    )

    st.write(
        f"Every {settings.FRAME_PROCESS_EVERY_N} frames"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### System information"
    )

    st.caption(
        "YOLOv8 person detection"
    )

    st.caption(
        "InsightFace recognition"
    )

    st.caption(
        "SQLite database"
    )

    st.caption(
        "Apple M2 optimized pipeline"
    )

    st.markdown(
        "---"
    )

    st.info(
        "The camera/AI pipeline should run separately. "
        "This dashboard displays its latest annotated frame."
    )


# ============================================================================
# DATABASE STATISTICS
# ============================================================================

stats = db_manager.get_stats()


# c1, c2, c3, c4 = st.columns(4)


# c1.metric(
#     "Registered Stakeholders",
#     stats["stakeholders"]
# )

# c2.metric(
#     "Total Visits Logged",
#     stats["visits"]
# )

# c3.metric(
#     "Visits Today",
#     stats["visits_today"]
# )

# c4.metric(
#     "Unknown Persons",
#     stats["unknowns"]
# )


# ============================================================================
# METRIC CARDS
# ============================================================================

st.markdown(
    """
    <style>

    /* ============================================================
        METRIC HOVER EFFECT
       ============================================================ */

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {

        transition:
            transform 0.25s ease,
            filter 0.25s ease;

    }

    /* Individual metric card */

    [data-testid="stMetric"] {

        transition:
            transform 0.28s ease,
            box-shadow 0.28s ease,
            border-color 0.28s ease,
            background 0.28s ease;

        cursor: pointer;

        position: relative;

        overflow: hidden;
    }


    /* ============================================================
       HOVER
       ============================================================ */

    [data-testid="stMetric"]:hover {

        transform:
            translateY(-8px)
            scale(1.025);

        background:
            rgba(255, 255, 255, 0.11);

        border-color:
            rgba(196, 181, 253, 0.55);

        box-shadow:

            0 18px 45px
            rgba(2, 6, 23, 0.45),

            0 0 25px
            rgba(124, 58, 237, 0.28),

            inset 0 1px 0
            rgba(255, 255, 255, 0.20);
    }


    /* ============================================================
       TOP GLOW LINE
       ============================================================ */

    [data-testid="stMetric"]::before {

        content: "";

        position: absolute;

        top: 0;
        left: 0;

        width: 100%;
        height: 3px;

        background:
            linear-gradient(
                90deg,
                #7C3AED,
                #A78BFA,
                #C4B5FD,
                #7C3AED
            );

        background-size:
            200% 100%;

        opacity: 0.75;

        transition:
            opacity 0.25s ease,
            height 0.25s ease;
    }


    [data-testid="stMetric"]:hover::before {

        opacity: 1;

        height: 4px;

        animation:
            metricGlow 2s linear infinite;
    }


    /* ============================================================
       SUBTLE SHINE EFFECT
       ============================================================ */

    [data-testid="stMetric"]::after {

        content: "";

        position: absolute;

        top: -100%;
        left: -120%;

        width: 70%;
        height: 300%;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(255,255,255,0.12),
                transparent
            );

        transform:
            rotate(20deg);

        transition:
            left 0.65s ease;

        pointer-events: none;
    }


    [data-testid="stMetric"]:hover::after {

        left: 140%;
    }


    /* ============================================================
       VALUE ANIMATION
       ============================================================ */

    [data-testid="stMetricValue"] {

        transition:
            transform 0.25s ease,
            text-shadow 0.25s ease;

    }


    [data-testid="stMetric"]:hover
    [data-testid="stMetricValue"] {

        transform:
            scale(1.08);

        text-shadow:

            0 0 12px
            rgba(196, 181, 253, 0.55),

            0 0 30px
            rgba(124, 58, 237, 0.30);
    }


    /* ============================================================
       LABEL
       ============================================================ */

    [data-testid="stMetricLabel"] {

        transition:
            transform 0.25s ease;

    }


    [data-testid="stMetric"]:hover
    [data-testid="stMetricLabel"] {

        transform:
            translateY(-2px);
    }


    /* ============================================================
       GLOW ANIMATION
       ============================================================ */

    @keyframes metricGlow {

        0% {
            background-position: 0% 50%;
        }

        100% {
            background-position: 200% 50%;
        }

    }


    /* ============================================================
       REDUCE MOTION FOR ACCESSIBILITY
       ============================================================ */

    @media (prefers-reduced-motion: reduce) {

        [data-testid="stMetric"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {

            transition: none !important;

            animation: none !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# METRICS
# ============================================================================

c1, c2, c3, c4 = st.columns(
    4,
    gap="medium"
)


c1.metric(
    "Registered Stakeholders",
    stats["stakeholders"]
)


c2.metric(
    "Total Visits Logged",
    stats["visits"]
)


c3.metric(
    "Visits Today",
    stats["visits_today"]
)


c4.metric(
    "Unknown Persons",
    stats["unknowns"]
)


# ============================================================================
# TABS
# ============================================================================

(
    tab_live,
    tab_visits,
    tab_unknown,
    tab_people,
    tab_reports
) = st.tabs(
    [
        "📡 Live Monitor",
        "🧾 Visit Logs",
        "🚨 Unknown Persons",
        "👩‍💻🧑‍💻 Stakeholders",
        "📊 Reports",
    ]
)


# ============================================================================
# LIVE MONITOR
# ============================================================================

with tab_live:

    st.markdown(
        '<div class="live-monitor">',
        unsafe_allow_html=True
    )

    render_live_monitor()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# VISIT LOGS
# ============================================================================

with tab_visits:

    st.subheader(
        "Stakeholder Visit History"
    )

    col_filter, col_limit = st.columns(
        [2, 1]
    )

    with col_filter:

        name_filter = st.text_input(
            "Search by name",
            ""
        )

    with col_limit:

        limit = st.number_input(
            "Rows",
            min_value=10,
            max_value=5000,
            value=500,
            step=50
        )

    df = visits_dataframe(
        int(limit),
        name_filter or None
    )

    if df.empty:

        st.info(
            "No visit records found."
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(

            "⬇️ Export CSV",

            df.to_csv(
                index=False
            ).encode("utf-8"),

            "visit_logs.csv",

            "text/csv"

        )


# ============================================================================
# UNKNOWN PERSONS
# ============================================================================

with tab_unknown:

    st.subheader(
        "Unknown Person Registrations"
    )

    only_pending = st.checkbox(
        "Show only unverified",
        value=False
    )

    udf = unknowns_dataframe(
        200,
        only_pending
    )

    if udf.empty:

        st.success(
            "No unknown persons recorded. ✅"
        )

    else:

        st.dataframe(
            udf.drop(
                columns=["Image"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "#### Captured Faces"
        )

        cols = st.columns(4)

        for i, row in udf.head(24).iterrows():

            with cols[i % 4]:

                img = row["Image"]

                if (
                    img
                    and os.path.isfile(img)
                ):

                    st.image(
                        img,
                        use_container_width=True
                    )

                else:

                    st.write(
                        "*(image missing)*"
                    )

                st.caption(
                    f"#{row['ID']} • "
                    f"{row['Timestamp']} • "
                    f"{row['Camera Location']}"
                )

                if not row["Verified"]:

                    if st.button(
                        "Mark verified",
                        key=f"verify_{row['ID']}"
                    ):

                        db_manager.mark_unknown_verified(
                            int(row["ID"])
                        )

                        st.rerun()

                else:

                    st.caption(
                        "✔ verified"
                    )


# ============================================================================
# STAKEHOLDERS
# ============================================================================

with tab_people:

    st.subheader(
        "Registered Stakeholders"
    )

    sdf = stakeholders_dataframe()

    if sdf.empty:

        st.info(
            "No stakeholders registered yet."
        )

    else:

        st.dataframe(
            sdf.drop(
                columns=["Image"]
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================================
# REPORTS
# ============================================================================

with tab_reports:

    st.subheader(
        "Activity Overview"
    )

    df = visits_dataframe(
        5000
    )

    if df.empty:

        st.info(
            "No visit data yet."
        )

    else:

        df["Date"] = pd.to_datetime(
            df["Timestamp"]
        ).dt.date

        # ----------------------------------------------------
        # Aggregations
        # ----------------------------------------------------

        daily_visits = (
            df.groupby("Date")
            .size()
            .reset_index(
                name="Count"
            )
        )

        cam_visits = (
            df.groupby("Camera Location")
            .size()
            .reset_index(
                name="Count"
            )
        )

        role_visits = (
            df.groupby("Role")
            .size()
            .reset_index(
                name="Count"
            )
        )

        # ----------------------------------------------------
        # Common chart configuration
        # ----------------------------------------------------

        layout_config = dict(

            xaxis=dict(
                fixedrange=True
            ),

            yaxis=dict(
                fixedrange=True,
                rangemode="tozero",
                tickformat="d"
            ),

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            font=dict(
                color="#E5E7EB"
            ),

        )

        ui_config = {
            "displayModeBar": False
        }

        # ====================================================
        # ROW 1
        # ====================================================

        left, right = st.columns(2)

        with left:

            fig_day = px.bar(

                daily_visits,

                x="Date",

                y="Count",

                color="Count",

                color_continuous_scale="Sunsetdark",

                title="<b>Visits per Day</b>"

            )

            fig_day.update_traces(
                width=0.4
            )

            fig_day.update_layout(
                **layout_config,
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig_day,
                use_container_width=True,
                config=ui_config
            )

        with right:

            fig_cam = px.bar(

                cam_visits,

                x="Camera Location",

                y="Count",

                color="Camera Location",

                color_discrete_sequence=
                    px.colors.qualitative.Vivid,

                title="<b>Visits per Camera</b>"

            )

            fig_cam.update_traces(
                width=0.4
            )

            fig_cam.update_layout(
                **layout_config,
                showlegend=False
            )

            st.plotly_chart(
                fig_cam,
                use_container_width=True,
                config=ui_config
            )

        # ====================================================
        # ROLE CHART
        # ====================================================

        fig_role = px.bar(

            role_visits,

            x="Role",

            y="Count",

            color="Role",

            color_discrete_sequence=
                px.colors.qualitative.Alphabet,

            title="<b>Visits per Role</b>"

        )

        fig_role.update_traces(
            width=0.2
        )

        fig_role.update_layout(
            **layout_config,
            showlegend=False
        )

        st.plotly_chart(
            fig_role,
            use_container_width=True,
            config=ui_config
        )

        # ====================================================
        # TREND
        # ====================================================

        st.markdown(
            "<br><hr><br>",
            unsafe_allow_html=True
        )

        fig_trend = px.line(

            daily_visits,

            x="Date",

            y="Count",

            title=
                "<b>Visitor Trend Over Time</b>",

            markers=True,

            line_shape="spline"

        )

        fig_trend.update_traces(
            line=dict(
                width=4
            ),

            marker=dict(
                size=9
            )
        )

        fig_trend.update_layout(
            **layout_config
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True,
            config=ui_config
        )


# ============================================================================
# FOOTER
# ============================================================================

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