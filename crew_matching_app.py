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
# Helpers: PDF parsing with pdfplumber (no system deps)
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

    # cluster by row, pick the row containing the most distinct day numbers
    rows = cluster_rows(day_words, tol=3.0)
    best_row = max(rows, key=lambda r: len({w["text"] for w in r}))
    colmap = {}
    for w in best_row:
        d = str(int(w["text"]))  # normalize like '3'
        xcenter = (w["x0"] + w["x1"]) / 2
        colmap[d] = xcenter
    return colmap

def nearest_day(x, day_xcenters):
    """Map an x position to the nearest day column key."""
    if not day_xcenters:
        return None
    return min(day_xcenters.keys(), key=lambda d: abs(day_xcenters[d] - x))

def parse_pdf_availability(file_like, available_code="A"):
    """
    Parse the roster PDF and return:
      - days: sorted list of day strings ({'1','2',...})
      - availability: dict day -> set of pilot_codes available that day
      We identify pilot codes like '(KVB)' and mark days where 'A' appears on their row.
    """
    availability = {}
    all_days = set()
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            if not words:
                continue

            day_cols = detect_day_columns(words)
            all_days.update(day_cols.keys())

            # Build rows from all words, then within each row find pilot code and 'A' tokens
            rows = cluster_rows(words, tol=3.0)
            for row in rows:
                text_line = " ".join(w["text"] for w in row)
                m = re.search(r"\(([A-Z]{3})\)", text_line)  # pilot code in parentheses
                if not m:
                    continue
                pilot_code = m.group(1)

                # Find each 'A' word in the row and map its x to the nearest day
                for w in row:
                    if w["text"].strip() == available_code:
                        xcenter = (w["x0"] + w["x1"]) / 2
                        d = nearest_day(xcenter, day_cols)
                        if d is not None:
                            availability.setdefault(d, set()).add(pilot_code)

    return sorted(all_days, key=lambda x: int(x)), availability

# -----------------------------
# Matching
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    pdf_file = st.file_uploader("Roster PDF", type=["pdf"])
    roles_csv = st.file_uploader("Pilot roles CSV (Pilot,Role)", type=["csv"])
    restr_csv = st.file_uploader("Restrictions CSV (PIC,SIC)", type=["csv"])
    avail_code = st.text_input("Availability code to look for", "A")

    st.caption("Roles CSV example:\n\nPilot,Role\nKVB,PIC\nHEB,SIC\n...")
    st.caption("Restrictions CSV example:\n\nPIC,SIC\nKVB,HEB\nYAD,TJF\n...")

if not pdf_file:
    st.info("Upload your roster PDF to begin.")
    st.stop()

# Parse PDF once
with st.spinner("Reading PDF and detecting availability…"):
    # streamlit uploads are SpooledTemporaryFile; read into bytes so pdfplumber can reopen multiple times if needed
    data = pdf_file.read()
    pdf_bytes = io.BytesIO(data)
    days, availability = parse_pdf_availability(pdf_bytes, available_code=avail_code)

if not days:
    st.error("I couldn't detect the day columns. If your PDF layout differs, share a sample and I can adjust the parser.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    chosen_day = st.selectbox("Select day of month", days)
with col2:
    st.write("")

st.subheader(f"Availability (code = '{avail_code}')")
st.write(f"Detected days: {', '.join(days)}")

# Show raw availability for the chosen day
avail_today = sorted(list(availability.get(chosen_day, set())))
st.write(f"Pilots with '{avail_code}' on day {chosen_day}: **{len(avail_today)}**")
st.dataframe(pd.DataFrame(avail_today, columns=["Pilot code"]))

# Need roles + restrictions to compute pairings
if not roles_csv or not restr_csv:
    st.warning("Upload the Roles and Restrictions CSVs (left sidebar) to compute pairings.")
    st.stop()

roles_df = pd.read_csv(roles_csv)
role_map = dict(zip(roles_df["Pilot"].astype(str).str.upper(), roles_df["Role"].astype(str).str.upper()))

restr_df = pd.read_csv(restr_csv)
restrictions = set((str(r.PIC).upper(), str(r.SIC).upper()) for _, r in restr_df.iterrows())

# Build and solve matching
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

# Download
pairs_csv = pairs_df.to_csv(index=False).encode("utf-8")
st.download_button("Download pairings CSV", data=pairs_csv, file_name=f"pairings_day_{chosen_day}.csv", mime="text/csv")

# Debug/advanced
with st.expander("Debug info"):
    st.write("Allowed edges:", len(allowed_edges))
    st.text(json.dumps({"PICs": PICs, "SICs": SICs}, indent=2))
