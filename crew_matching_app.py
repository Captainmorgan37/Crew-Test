import streamlit as st
from docx import Document
import re
import pandas as pd
import matplotlib.pyplot as plt

# Parse roster from Word document
def parse_roster_docx(file):
    doc = Document(file)
    data = []

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line:
            continue

        # Match: Full Name (ABC) followed by duties
        m = re.match(r"^(.*?)\s+\(([A-Z]{3})\)\s+(.*)$", line)
        if not m:
            continue

        full_name = m.group(1).strip()
        pilot_code = m.group(2).strip()
        duties = m.group(3).split()

        data.append({
            "Full Name": full_name,
            "Pilot": pilot_code,
            "Duties": duties
        })

    return pd.DataFrame(data)


# Streamlit app
st.title("Crew A-Day Tracker (DOCX Version)")

uploaded_file = st.file_uploader("Upload a roster (.docx)", type=["docx"])

if uploaded_file:
    df = parse_roster_docx(uploaded_file)

    if not df.empty:
        st.success("Roster parsed successfully!")

        # Figure out how many days are in schedule (based on longest duties list)
        max_days = df["Duties"].map(len).max()
        selected_day = st.number_input("Select a day", min_value=1, max_value=max_days, value=1)

        # Find A-day pilots
        a_day_pilots = df[df["Duties"].map(lambda d: d[selected_day-1] if selected_day-1 < len(d) else None) == "A"]

        st.subheader(f"Pilots on A day {selected_day}")
        if not a_day_pilots.empty:
            st.dataframe(a_day_pilots[["Full Name", "Pilot"]])

            # Simple count chart
            fig, ax = plt.subplots()
            ax.bar(a_day_pilots["Pilot"], [1]*len(a_day_pilots))
            ax.set_title(f"A-day Pilots on Day {selected_day}")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info(f"No pilots found on A day {selected_day}.")
    else:
        st.error("No valid roster data found in the document.")
