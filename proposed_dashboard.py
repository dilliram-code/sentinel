import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from PIL import Image
import time


# PAGE CONFIGURATION
##############################################################

st.set_page_config(
    page_title="University Stakeholders Detection System",
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

##############################################################
# SIDEBAR
##############################################################

st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("Campus AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📹 Live Monitoring",

        "👥 Stakeholders",

        "🚨 Unknown Persons",

        "📋 Visit History",

        "📊 Analytics",

        "⚙ Settings"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Current Version

Version 1.0

Development Phase
"""
)

##############################################################
# DUMMY DATA
##############################################################

total_people = 48

known_people = 41

unknown_people = 7

camera_online = 4

##############################################################
# HOME DASHBOARD
##############################################################

if page == "🏠 Dashboard":

    st.markdown(
        "<div class='title'>🎓 University Stakeholder Detection System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Real-Time Stakeholder Identification & Unknown Person Registration</div>",
        unsafe_allow_html=True
    )

    ##########################################################

    c1,c2,c3,c4 = st.columns(4)

    ##########################################################

    with c1:
        st.metric(
            "👥 Total Persons",
            total_people,
            "+8 Today"
        )

    ##########################################################

    with c2:
        st.metric(
            "✅ Known Persons",
            known_people,
            "+4"
        )

    ##########################################################

    with c3:
        st.metric(
            "🚨 Unknown Persons",
            unknown_people,
            "-2"
        )

    ##########################################################

    with c4:

        st.metric(

            "📹 Cameras Online",

            camera_online,

            "100%"

        )

    st.divider()

    ##########################################################

    left,right = st.columns([2,1])

    ##########################################################

    with left:

        st.subheader("📹 Live Camera Preview")

        st.info(
            """
            Live CCTV feed will appear here.

            During development this section will later be connected to:

            OpenCV ➜ Webcam

            Production:

            RTSP CCTV Stream
            """
        )

        st.image(

            "https://placehold.co/1000x500/png?text=Live+Camera+Feed",

            use_container_width=True

        )

    ##########################################################

    with right:

        st.subheader("📡 Camera Status")

        status = pd.DataFrame(

            {

                "Camera":[

                    "Gate",

                    "Library",

                    "Parking",

                    "Admin"

                ],

                "Status":[

                    "🟢 Online",

                    "🟢 Online",

                    "🟢 Online",

                    "🟢 Online"

                ]

            }

        )

        st.dataframe(

            status,

            use_container_width=True,

            hide_index=True

        )

        st.success("All Cameras Operational")

    st.divider()

    ##########################################################

    c1,c2 = st.columns(2)

    ##########################################################

    with c1:

        st.subheader("🕒 Recent Detections")

        recent = pd.DataFrame(

            {

                "Name":[

                    "Dilli Ram",

                    "Piyush",

                    "Unknown",

                    "Faculty 01",

                    "Student 02"

                ],

                "Role":[

                    "Faculty",

                    "Student",

                    "-",

                    "Faculty",

                    "Student"

                ],

                "Time":[

                    "09:05",

                    "09:07",

                    "09:10",

                    "09:11",

                    "09:13"

                ]

            }

        )

        st.dataframe(

            recent,

            use_container_width=True,

            hide_index=True

        )

    ##########################################################

    with c2:

        st.subheader("🚨 Recent Unknown Persons")

        cols = st.columns(2)

        for col in cols:

            with col:

                st.image(

                    "https://placehold.co/250x250/png?text=Unknown",

                    use_container_width=True

                )

                st.caption("Detected: Today")

    ##########################################################

    st.divider()

    st.subheader("📈 Daily Detection Trend")

    chart_data = pd.DataFrame(

        {

            "Day":[

                "Mon",

                "Tue",

                "Wed",

                "Thu",

                "Fri",

                "Sat",

                "Sun"

            ],

            "Persons":[

                80,

                92,

                110,

                101,

                130,

                75,

                50

            ]

        }

    )

    fig = px.line(

        chart_data,

        x="Day",

        y="Persons",

        markers=True,

        template="plotly_white"

    )

    fig.update_layout(

        height=400,

        margin=dict(

            l=20,

            r=20,

            t=20,

            b=20

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

##############################################################
# PLACEHOLDER PAGES
##############################################################

elif page == "📹 Live Monitoring":
    st.title("📹 Live Monitoring")
    st.info("Coming in Part 2")
elif page == "👥 Stakeholders":
    st.title("👥 Stakeholder Management")
    st.info("Coming in Part 3")
elif page == "🚨 Unknown Persons":
    st.title("🚨 Unknown Persons")
    st.info("Coming in Part 3")
elif page == "📋 Visit History":
    st.title("📋 Visit History")
    st.info("Coming in Part 4")
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    st.info("Coming in Part 4")

elif page == "⚙ Settings":

    st.title("⚙ Settings")

    st.info("Coming in Part 4")

##############################################################
# FOOTER
##############################################################

st.markdown("---")

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