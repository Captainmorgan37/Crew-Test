import streamlit as st
import camelot
import pandas as pd
import networkx as nx

# -----------------------------
# Helpers
# -----------------------------
def load_roster(pdf_file):
    tables = camelot.read_pdf(pdf_file, pages="all", flavor="stream")
    df = tables[0].df
    df.columns = df.iloc[0]   # first row as header
    df = df.drop(0).reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: "Pilot"})
    return df

def load_roles(csv_file):
    roles = pd.read_csv(csv_file)
    return dict(zip(roles["Pilot"], roles["Role"]))

def load_restrictions(csv_file):
    restr = pd.read_csv(csv_file)
    return set((row.PIC, row.SIC) for _, row in restr.iterrows())

def get_available_pilots(df, day_col):
    available = df[df[day_col] == "A"]["Pilot"].tolist()
    return available

def build_allowed_pairs(available, roles, restrictions):
    PICs = [p for p in available if roles.get(p) == "PIC"]
    SICs = [p for p in available if roles.get(p) == "SIC"]

    allowed_pairs = []
    for pic in PICs:
        for sic in SICs:
            if (pic, sic) not in restrictions:
                allowed_pairs.append((pic, sic))
    return PICs, SICs, allowed_pairs

def find_max_matching(PICs, SICs, allowed_pairs):
    G = nx.Graph()
    G.add_nodes_from(PICs, bipartite=0)
    G.add_nodes_from(SICs, bipartite=1)
    G.add_edges_from(allowed_pairs)

    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=PICs)
    pairings = [(pic, sic) for pic, sic in matching.items() if pic in PICs]
    return pairings

# -----------------------------
# Streamlit App
# -----------------------------
st.title("✈️ Crew Matching Tool")

pdf_file = st.file_uploader("Upload Crew Roster PDF", type=["pdf"])
roles_file = st.file_uploader("Upload Pilot Roles CSV", type=["csv"])
restrictions_file = st.file_uploader("Upload Restrictions CSV", type=["csv"])

day = st.text_input("Enter day of month (e.g. '3')", "")

if pdf_file and roles_file and restrictions_file and day:
    with st.spinner("Processing..."):
        df = load_roster(pdf_file)
        roles = load_roles(roles_file)
        restrictions = load_restrictions(restrictions_file)

        available = get_available_pilots(df, day)
        PICs, SICs, allowed_pairs = build_allowed_pairs(available, roles, restrictions)
        pairings = find_max_matching(PICs, SICs, allowed_pairs)

    st.subheader(f"Results for Day {day}")
    st.write(f"✅ Available pilots: {len(available)}")
    st.write(f"👨‍✈️ PICs: {len(PICs)}, 👩‍✈️ SICs: {len(SICs)}")
    st.write(f"🛩️ Max crewed planes: {len(pairings)}")

    st.subheader("Pairings")
    st.table(pd.DataFrame(pairings, columns=["PIC", "SIC"]))
