import io
import re
import pandas as pd
import networkx as nx
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Crew Matching Tool", page_icon="✈️", layout="wide")
st.title("✈️ Crew Matching Tool (PDF → availability → pairings)")

# -----------------------------
# Helpers
# -----------------------------
def cluster_rows(words, tol=6.0):
    """Group words by rough row using their vertical center."""
    rows = {}
    for w in words:
        yc = (w["top"] + w["bottom"]) / 2
        key = round(yc / tol)
        rows.setdefault(key, []).append(w)
    return [sorted(v, key=lambda x: x["x0"]) for _, v in sorted(rows.items(), key=lambda kv: kv[0])]

def parse_pdf_availability(file_like, available_code="A", valid_pilots=None, debug=False):
    """Parse roster PDF where pilot codes are in parentheses after full names."""
    if valid_pilots is not None:
        valid_pilots = set([p.upper() for p in valid_pilots])

    availability = {}
    all_days = set()

    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            if not words:
                continue

            # Detect day columns (1-31)
            day_words = [w for w in words if w["text"].lstrip("0").isdigit() and 1 <= int(w["text"].lstrip("0")) <= 31]
            if not day_words:
                continue

            day_rows = cluster_rows(day_words)
            best_row = max(day_rows, key=lambda r: len({w["text"] for w in r}))
            day_cols = {str(int(w["text"].lstrip("0"))): (w["x0"] + w["x1"]) / 2 for w in best_row}
            all_days.update(day_cols.keys())

            rows = cluster_rows(words)
            for row in rows:
                text_line = " ".join(w["text"] for w in row)
                # Robustly capture 3-letter pilot codes in parentheses, even if split
                matches = re.findall(r"\(?([A-Z]{3})\)?", text_line)
                if not matches:
                    continue

                pilot_codes = [m for m in matches if not valid_pilots or m in valid_pilots]
                if not pilot_codes:
                    continue

                for w in row:
                    if w["text"].strip().upper() == available_code.upper():
                        xcenter = (w["x0"] + w["x1"]) / 2
                        if not day_cols:
                            continue
                        nearest_day = min(day_cols.keys(), key=lambda d: abs(day_cols[d] - xcenter))
                        nearest_day_str = str(nearest_day)
                        for pilot_code in pilot_codes:
                            availability.setdefault(nearest_day_str, set()).add(pilot_code)
                        if debug:
                            st.write(f"Found {available_code} for pilots {pilot_codes} on day {nearest_day_str}")

    return sorted([str(d) for d in all_days], key=int), availability

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

if not pdf_file:
    st.info("Upload your roster PDF to begin.")
    st.stop()

if not roles_file or not restr_file:
    st.warning("Upload the Roles and Restrictions files to compute pairings.")
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
# Parse PDF
# -----------------------------
with st.spinner("Reading PDF and detecting availability…"):
    pdf_file.seek(0)  # Reset file pointer
    pdf_bytes = io.BytesIO(pdf_file.read())
    days, availability = parse_pdf_availability(
        pdf_bytes,
        available_code=avail_code,
        valid_pilots=valid_pilots,
        debug=True  # Set False in production
    )

if not days:
    st.error("No day columns detected. Share a sample PDF if parsing fails.")
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

