import io
import re
import pandas as pd
import networkx as nx
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import streamlit as st

st.set_page_config(page_title="Crew Matching Tool", page_icon="✈️", layout="wide")
st.title("✈️ Crew Matching Tool (PDF → availability → pairings via OCR)")

# -----------------------------
# Helpers
# -----------------------------
def parse_pdf_ocr(file_like, available_code="A", valid_pilots=None, debug=False):
    """
    OCR-based PDF parser.
    Converts each PDF page to an image, extracts text, and finds pilot codes + availability.
    Returns:
        days: list of days as strings
        availability: dict {day -> set of pilot codes}
    """
    availability = {}
    all_days = set()
    valid_pilots = set([p.upper() for p in valid_pilots]) if valid_pilots else None

    images = convert_from_bytes(file_like.read())
    for page_num, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img)
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Find all 3-letter pilot codes in parentheses
            pilot_codes = re.findall(r"\(([A-Z]{3})\)", line)
            if not pilot_codes:
                continue
            if valid_pilots:
                pilot_codes = [p for p in pilot_codes if p in valid_pilots]
            if not pilot_codes:
                continue

            # Find all day columns with the availability code
            # Assuming format like: "1 A 2 OFF 3 A ..." or "1A 2OFF 3A"
            day_matches = re.findall(r"(\d{1,2})\s*(" + re.escape(available_code) + r")", line, re.IGNORECASE)
            for day_str, code in day_matches:
                all_days.add(day_str)
                for pilot_code in pilot_codes:
                    availability.setdefault(day_str, set()).add(pilot_code)
                if debug:
                    st.write(f"Found {available_code} for {pilot_codes} on day {day_str}")

    return sorted(list(all_days), key=int), availability

def build_allowed_pairs(available_codes, role_map, restrictions_set):
    PICs = [p for p in available_codes if role_map.get(p, "").upper() == "PIC"]
    SICs = [p for p in available_codes if role_map.get(p, "").upper() == "SIC"]
    allowed = [(pic, sic) for pic in PICs for sic in SICs if (pic, sic) not in restrictions_set]
    return PICs, SICs, allowed

def max_bipartite_pairings(PICs, SICs, edges):
    G = nx.Graph()
    G.add_nodes_from(PICs, bipartite=0)
    G.add_nodes_from(SICs, bipartite=1)
    G.add_edges_from(edges)
    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=PICs)
    return [(pic, partner) for pic, partner in matching.items() if pic in PICs]

def load_dataframe(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    pdf_file = st.file_uploader("Roster PDF", type=["pdf"])
    roles_file = st.file_uploader("Pilot roles (CSV or Excel)", type=["csv", "xlsx"])
    restr_file = st.file_uploader("Restrictions (CSV or Excel)", type=["csv", "xlsx"])
    avail_code = st.text_input("Availability code to look for", "A")
    debug_mode = st.checkbox("Debug OCR output", value=True)

if not pdf_file or not roles_file or not restr_file:
    st.warning("Upload PDF, Roles, and Restrictions files to compute pairings.")
    st.stop()

# -----------------------------
# Load roles and restrictions
# -----------------------------
roles_df = load_dataframe(roles_file)
valid_pilots = roles_df["Pilot"].astype(str).str.upper().tolist()
role_map = dict(zip(roles_df["Pilot"].astype(str).str.upper(), roles_df["Role"].astype(str).str.upper()))

restr_df = load_dataframe(restr_file)
restrictions = set((str(r.PIC).upper(), str(r.SIC).upper()) for _, r in restr_df.iterrows())

# -----------------------------
# Parse PDF with OCR
# -----------------------------
with st.spinner("Performing OCR on PDF and detecting availability…"):
    pdf_bytes = io.BytesIO(pdf_file.read())
    days, availability = parse_pdf_ocr(
        pdf_bytes,
        available_code=avail_code,
        valid_pilots=valid_pilots,
        debug=debug_mode
    )

if not days:
    st.error("No day columns detected. OCR may have failed. Check the PDF.")
    st.stop()

# -----------------------------
# Select day and show availability
# -----------------------------
chosen_day = st.selectbox("Select day of month", days)
avail_today = sorted(list(availability.get(chosen_day, set())))

st.subheader(f"Availability (code = '{avail_code}')")
st.write(f"Pilots with '{avail_code}' on day {chosen_day}: **{len(avail_today)}**")
st.dataframe(pd.DataFrame(avail_today, columns=["Pilot code"]))

# -----------------------------
# Compute pairings
# -----------------------------
PICs, SICs, allowed_edges = build_allowed_pairs([p.upper() for p in avail_today], role_map, restrictions)
pairings = max_bipartite_pairings(PICs, SICs, allowed_edges)

st.markdown("---")
st.subheader(f"Pairing results for day {chosen_day}")
m1, m2, m3 = st.columns(3)
m1.metric("PICs available", len(PICs))
m2.metric("SICs available", len(SICs))
m3.metric("Max crewed planes", len(pairings))

pairs_df = pd.DataFrame(pairings, columns=["PIC", "SIC"])
st.dataframe(pairs_df, use_container_width=True)

pairs_csv = pairs_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download pairings CSV",
    data=pairs_csv,
    file_name=f"pairings_day_{chosen_day}.csv",
    mime="text/csv"
)
