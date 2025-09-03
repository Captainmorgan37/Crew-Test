import io
import re
import json
import pdfplumber
import pandas as pd
import networkx as nx
import streamlit as st

st.set_page_config(page_title="Crew Matching Tool", page_icon="✈️", layout="wide")
st.title("✈️ Crew Matching Tool (PDF → availability → pairings)")

# -----------------------------
# Helpers
# -----------------------------
def cluster_rows(words, tol=3.0):
    """Group words by rough row using their vertical center."""
    rows = {}
    for w in words:
        yc = (w["top"] + w["bottom"]) / 2
        key = round(yc / tol)  # bucket
        rows.setdefault(key, []).append(w)
    # sort by vertical position
    return [sorted(v, key=lambda x: x["x0"]) for _, v in sorted(rows.items(), key=lambda kv: kv[0])]

def detect_day_columns(page_words):
    """Return dict: day(str '1'..'31') -> x_center from header line with most day numbers."""
    day_words = [w for w in page_words if w["text"].isdigit() and 1 <= int(w["text"]) <= 31]
    if not day_words:
        return {}
    rows = cluster_rows(day_words, tol=3.0)
    best_row = max(rows, key=lambda r: len({w["text"] for w in r}))
    colmap = {}
    for w in best_row:
        d = str(int(w["text"]))
        xcenter = (w["x0"] + w["x1"]) / 2
        colmap[d] = xcenter
    return colmap

def nearest_day(x, day_xcenters):
    if not day_xcenters:
        return None
    return min(day_xcenters.keys(), key=lambda d: abs(day_xcenters[d] - x))

def parse_pdf_availability(file_like, available_code="A", debug=False):
    """
    Parses a roster PDF and returns:
      - sorted list of all days detected
      - availability dict: day -> set of pilot codes
    """
    availability = {}
    all_days = set()

    with pdfplumber.open(file_like) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            if not words:
                if debug: st.write(f"Page {page_num}: no words detected")
                continue

            # Normalize day words: numbers 1-31
            day_words = [w for w in words if w["text"].lstrip("0").isdigit() and 1 <= int(w["text"].lstrip("0")) <= 31]
            if not day_words:
                if debug: st.write(f"Page {page_num}: no day numbers found")
                continue

            # Cluster day words into header row
            day_rows = cluster_rows(day_words, tol=5.0)
            best_row = max(day_rows, key=lambda r: len({w["text"] for w in r}))
            day_cols = {str(int(w["text"].lstrip("0"))): (w["x0"] + w["x1"]) / 2 for w in best_row}
            all_days.update(day_cols.keys())

            if debug: st.write(f"Page {page_num} day columns:", day_cols)

            # Cluster all words into rows
            rows = cluster_rows(words, tol=5.0)
            for row in rows:
                text_line = " ".join(w["text"] for w in row)
                # Pilot code: match either (KVB) or just KVB
                m = re.search(r"\(?([A-Z]{1,3})\)?", text_line)
                if not m:
                    continue
                pilot_code = m.group(1).upper()

                for w in row:
                    if w["text"].strip().upper() == available_code.upper():
                        xcenter = (w["x0"] + w["x1"]) / 2
                        # Find nearest day
                        if not day_cols:
                            continue
                        nearest = min(day_cols.keys(), key=lambda d: abs(day_cols[d] - xcenter))
                        availability.setdefault(nearest, set()).add(pilot_code)
                        if debug:
                            st.write(f"Found {available_code} for pilot {pilot_code} on day {nearest}")

    return sorted(all_days, key=lambda x: int(x)), availability


def build_allowed_pairs(available_codes, role_map, restrictions_set):
    PICs = [p for p in available_codes if role_map.get(p, "").upper() == "PIC"]
    SICs = [p for p in available_codes if role_map.get(p, "").upper() == "SIC"]
    allowed = []
    for pic in PICs:
        for sic in SICs:
            if (pic, sic) not in restrictions_set:
                allowed.append((pic, sic))
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
# UI
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    pdf_file = st.file_uploader("Roster PDF", type=["pdf"])
    roles_file = st.file_uploader("Pilot roles (CSV or Excel)", type=["csv", "xlsx"])
    restr_file = st.file_uploader("Restrictions (CSV or Excel)", type=["csv", "xlsx"])
    avail_code = st.text_input("Availability code to look for", "A")

    st.caption("Roles file example:\nPilot,Role\nKVB,PIC\nHEB,SIC\n...")
    st.caption("Restrictions file example:\nPIC,SIC\nKVB,HEB\nYAD,TJF\n...")

if not pdf_file:
    st.info("Upload your roster PDF to begin.")
    st.stop()

# Parse PDF
with st.spinner("Reading PDF and detecting availability…"):
    data = pdf_file.read()
    pdf_bytes = io.BytesIO(data)
    days, availability = parse_pdf_availability(pdf_bytes, available_code=avail_code)

if not days:
    st.error("No day columns detected. Share a sample PDF if parsing fails.")
    st.stop()

chosen_day = st.selectbox("Select day of month", days)

st.subheader(f"Availability (code = '{avail_code}')")
avail_today = sorted(list(availability.get(chosen_day, set())))
st.write(f"Pilots with '{avail_code}' on day {chosen_day}: **{len(avail_today)}**")
st.dataframe(pd.DataFrame(avail_today, columns=["Pilot code"]))

# Need roles + restrictions
if not roles_file or not restr_file:
    st.warning("Upload the Roles and Restrictions files to compute pairings.")
    st.stop()

roles_df = load_dataframe(roles_file)
role_map = dict(zip(roles_df["Pilot"].astype(str).str.upper(), roles_df["Role"].astype(str).str.upper()))

restr_df = load_dataframe(restr_file)
restrictions = set((str(r.PIC).upper(), str(r.SIC).upper()) for _, r in restr_df.iterrows())

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
st.download_button("Download pairings CSV", data=pairs_csv,
                   file_name=f"pairings_day_{chosen_day}.csv", mime="text/csv")

