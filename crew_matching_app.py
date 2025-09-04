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
def parse_pdf_table(file_like, available_code="A", valid_pilots=None, debug=False):
    """
    Extract pilot availability from a PDF roster table.
    Assumes table has:
    - First column: "Full Name (XXX)"
    - Subsequent columns: day numbers (1-31) with availability code ("A")
    Returns:
      - list of days (str)
      - dict: day -> set of pilot codes
    """
    availability = {}
    all_days = set()

    with pdfplumber.open(file_like) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            table = page.extract_table()
            if not table:
                continue

            # First row assumed to be header (days)
            header = table[0]
            day_cols = {}
            for idx, h in enumerate(header):
                if h is None:
                    continue
                h_clean = str(h).strip()
                if h_clean.isdigit() and 1 <= int(h_clean) <= 31:
                    day_cols[idx] = h_clean
                    all_days.add(h_clean)

            # Process data rows
            for row in table[1:]:
                if not row or len(row) < 2:
                    continue

                # Extract pilot codes from first column
                first_col = row[0] or ""
                matches = re.findall(r"\(([A-Z]{3})\)", first_col)
                if not matches:
                    continue

                pilot_codes = [m for m in matches if not valid_pilots or m in valid_pilots]
                if not pilot_codes:
                    continue

                # Check each day column
                for idx, cell in enumerate(row):
                    if idx not in day_cols:
                        continue
                    if cell and str(cell).strip().upper() == available_code.upper():
                        day = day_cols[idx]
                        for code in pilot_codes:
                            availability.setdefault(day, set()).add(code)
                        if debug:
                            st.write(f"Found {available_code} for {pilot_codes} on day {day}")

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
# Parse PDF
# -----------------------------
with st.spinner("Reading PDF and detecting availability…"):
    pdf_bytes = io.BytesIO(pdf_file.read())
    days, availability = parse_pdf_table(
        pdf_bytes,
        available_code=avail_code,
        valid_pilots=valid_pilots,
        debug=True  # turn off once working
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
