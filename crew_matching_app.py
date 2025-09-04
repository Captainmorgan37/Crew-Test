import pandas as pd
import networkx as nx
import streamlit as st

st.set_page_config(page_title="Crew Matching Tool", page_icon="✈️", layout="wide")
st.title("✈️ Crew Matching Tool (CSV → availability → pairings)")

# -----------------------------
# Helpers
# -----------------------------
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
    avail_file = st.file_uploader("Availability CSV", type=["csv", "xlsx"])
    roles_file = st.file_uploader("Pilot roles CSV/Excel", type=["csv", "xlsx"])
    restr_file = st.file_uploader("Restrictions CSV/Excel", type=["csv", "xlsx"])

if not avail_file or not roles_file or not restr_file:
    st.warning("Upload Availability, Roles, and Restrictions files.")
    st.stop()

# -----------------------------
# Load roles and restrictions
# -----------------------------
roles_df = load_dataframe(roles_file)
role_map = dict(zip(roles_df["Pilot"].astype(str).str.upper(), roles_df["Role"].astype(str).str.upper()))

restr_df = load_dataframe(restr_file)
restrictions = set((str(r.PIC).upper(), str(r.SIC).upper()) for _, r in restr_df.iterrows())

# -----------------------------
# Full name → code mapping
# -----------------------------
# You can expand this dict to include all pilots
name_to_code = {
    "Kendall Beckles": "KVB",
    "Hendrik Britz": "HEB",
    "Yann Delavault": "YAD",
    "Tyler Ferris": "TJF",
    "Thomas Klingler": "TJK",
    "Andrew MacSween": "AMA",
    "Mohammad Mansoor": "MAM",
    "Erik Marcille": "EDM",
    "Robert McEwen": "RJM",
    "Nicolas Pare": "NXP",
    "Matthew Rolleman": "MNR",
    "Maksym Sokol": "MSS",
    "Francis Therrien": "FXT",
    "Michael Van Der Mark": "MJV",
    "Carel Wentzel": "CDW",
    "Trevor Wright": "TDW",
    "Francisco Andrade": "FXA",
    "Nadim Bendidane": "NBB",
    "Vivien Frelat": "VJF",
    "Gary Goertzen": "GDG",
}

}

# -----------------------------
# Parse availability CSV
# -----------------------------
avail_df = load_dataframe(avail_file)
full_names = avail_df.iloc[:, 0].astype(str).tolist()  # first column only
pilot_codes = [name_to_code.get(name) for name in full_names if name_to_code.get(name)]

if not pilot_codes:
    st.error("No valid pilot codes found. Check your mapping.")
    st.stop()

# For simplicity, assume all pilots in this CSV are available for the **same day**
# If you have multiple days, you could have a column per day and expand accordingly
days = ["All Days"]
availability = {"All Days": set(pilot_codes)}

chosen_day = st.selectbox("Select day", days)
avail_today = sorted(list(availability.get(chosen_day, set())))

st.subheader("Available Pilots")
st.write(f"Pilots available: **{len(avail_today)}**")
st.dataframe(pd.DataFrame(avail_today, columns=["Pilot code"]))

# -----------------------------
# Compute pairings
# -----------------------------
PICs, SICs, allowed_edges = build_allowed_pairs([p.upper() for p in avail_today], role_map, restrictions)
pairings = max_bipartite_pairings(PICs, SICs, allowed_edges)

st.markdown("---")
st.subheader("Pairing Results")
m1, m2, m3 = st.columns(3)
m1.metric("PICs available", len(PICs))
m2.metric("SICs available", len(SICs))
m3.metric("Max crewed planes", len(pairings))

pairs_df = pd.DataFrame(pairings, columns=["PIC", "SIC"])
st.dataframe(pairs_df, use_container_width=True)

pairs_csv = pairs_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Pairings CSV",
    data=pairs_csv,
    file_name=f"pairings.csv",
    mime="text/csv"
)
