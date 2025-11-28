import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import numpy as np
import os

# نصب خودکار py3dmol (در Streamlit Cloud کار می‌کنه)
try:
    import py3Dmol
except ImportError:
    os.system("pip install py3Dmol")
    import py3Dmol

# ====================== DATA LOADING ======================
@st.cache_data
def load_data():
    df_atomic = pd.read_csv("ptable2.csv")
    df_atomic['symbol'] = df_atomic['symbol'].str.strip()

    with open("vaspkit_output.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for material, data in raw.items():
        if not data: continue
        row = {"material": material}
        # (همون کد قبلی برای Cij, Sij, Hill, Stability ...)
        if "Elastic_Tensor_Voigt" in data:
            for k, v in data["Elastic_Tensor_Voigt"].items():
                try: row[k] = float(v)
                except: pass
        if "Compliance_Tensor" in data:
            for k, v in data["Compliance_Tensor"].items():
                try: row[f"S_{k}"] = float(v)
                except: pass
        if "Anisotropic_Mechanical_Properties" in data:
            for prop, vals in data["Anisotropic_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_").replace("_(GPa)", "")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        if stat in vals:
                            try: row[f"{clean}_{stat}"] = float(vals[stat])
                            except: pass
        if "Average_Mechanical_Properties" in data:
            for prop, vals in data["Average_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_").replace("_(GPa)", "")
                if isinstance(vals, dict) and "Hill" in vals:
                    try: row[f"{clean}_Hill"] = float(vals["Hill"])
                    except: pass
        if "Additional_Properties" in data:
            for k, v in data["Additional_Properties"].items():
                if k in ["Mechanical_Stability", "Brittleness_Indicator"]:
                    row[k] = v
                else:
                    try: row[k] = float(v)
                    except: row[k] = v
        records.append(row)

    mech_df = pd.DataFrame(records)

    # میانگین اتمی (همون قبلی)
    def parse_formula(f):
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', f)
        return [(e, int(c) if c else 1) for e, c in matches]

    feature_cols = [c for c in df_atomic.columns if c != 'symbol']
    atomic_rows = []
    for _, row in mech_df.iterrows():
        material = row['material']
        try:
            elements = parse_formula(material)
        except:
            continue
        weighted = {col: 0.0 for col in feature_cols}
        total = 0
        for elem, count in elements:
            erow = df_atomic[df_atomic['symbol'] == elem]
            if erow.empty: break
            vals = erow.iloc[0]
            for col in feature_cols:
                val = pd.to_numeric(vals[col], errors='coerce')
                if not pd.isna(val):
                    weighted[col] += val * count
            total += count
        else:
            if total > 0:
                avg = {col: weighted[col] / total for col in feature_cols}
                avg['material'] = material
                atomic_rows.append(avg)

    atomic_df = pd.DataFrame(atomic_rows)
    df = pd.merge(atomic_df, mech_df, on='material', how='inner')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = sorted([c for c in numeric_cols if c != "material"])

    return df, all_features

df, all_features = load_data()
if df.empty:
    st.error("No data loaded!")
    st.stop()

st.set_page_config(page_title="MAX Phase 3D Explorer", layout="wide")
st.title("MAX Phase & Elastic Properties Explorer Pro")
st.markdown("**Point color: Green = Stable | Red = Unstable | Click → View 3D Structure**")

# ====================== NEON SLIDERS ======================
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #00ccff11, #00f2ff33) !important;
        height: 6px !important;
    }
    .stSlider > div > div > div[role="slider"] {
        background: #00ccff !important;
        border: 1.5px solid #00f2ff !important;
        box-shadow: 0 0 0 10px #00f2ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR + 3D VIEWER ======================
with st.sidebar:
    st.header("Material Details")

    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        row = df[df['material'] == mat].iloc[0]

        # Stability
        stability = row.get("Mechanical_Stability", "Unknown")
        if stability == "Stable":
            st.success("Mechanically Stable")
        else:
            st.error("Mechanically Unstable")

        brittleness = row.get("Brittleness_Indicator", "Unknown")
        if brittleness == "Brittle":
            st.warning("Brittle")
        elif brittleness == "Ductile":
            st.info("Ductile")

        st.markdown(f"### **{mat}**")

        # دکمه 3D ویو
        if st.button("View 3D Crystal Structure", type="primary"):
            st.session_state.show_3d = True

        # نمایش خواص
        details = row.drop("material")
        for col in df.columns:
            if col == "material": continue
            v = details.get(col)
            if pd.isna(v): continue
            if isinstance(v, (int, float, np.number)):
                st.write(f"**{col.replace('_', ' ')}**: {v:.4f}")
            else:
                st.write(f"**{col.replace('_', ' ')}**: {v}")

        if st.button("Clear Selection"):
            st.session_state.selected_material = None
            st.session_state.show_3d = False
            st.rerun()
    else:
        st.info("Click on a point to view details + 3D structure")

# ====================== 3D MODAL ======================
if st.session_state.get("show_3d", False) and st.session_state.get("selected_material"):
    mat = st.session_state.selected_material
    poscar_path = f"poscars/{mat}"

    if os.path.exists(poscar_path):
        with open(poscar_path, "r") as f:
            poscar_content = f.read()

        st.markdown("### 3D Crystal Structure")
        view = py3Dmol.view(width=600, height=500)
        view.addModel(poscar_content, "poscar")
        view.setStyle = {'stick': {'radius': 0.15, 'color': 'spectrum'}, 'sphere': {'scale': 0.3}}
        view.addStyle(viewStyle)
        view.setBackgroundColor('0x000000')
        view.zoomTo()
        view.spin(True)  # انیمیشن چرخش
        view.show()
        st.py3Dmol(view)
    else:
        st.error(f"POSCAR file not found for {mat}")

# ====================== PLOT + AXIS + SLIDERS ======================
# (همون کد قبلی برای نمودار، رنگ، رنج و کلیک)

col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X-axis", all_features, index=all_features.index("atomic_number") if "atomic_number" in all_features else 0)
with col2:
    y_axis = st.selectbox("Y-axis", all_features, index=0)

col_a, col_b = st.columns(2)
with col_a:
    x_min, x_max = st.slider(f"X Range ({x_axis})", float(df[x_axis].min()), float(df[x_axis].max()), (float(df[x_axis].min()), float(df[x_axis].max())))
with col_b:
    y_min, y_max = st.slider(f"Y Range ({y_axis})", float(df[y_axis].min()), float(df[y_axis].max()), (float(df[y_axis].min()), float(df[y_axis].max())))

filtered_df = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) & (df[y_axis] >= y_min) & (df[y_axis] <= y_max)].copy()
filtered_df = filtered_df[['material', x_axis, y_axis, 'Mechanical_Stability']].dropna()

if filtered_df.empty:
    st.warning("No data in range.")
else:
    filtered_df['color'] = filtered_df['Mechanical_Stability'].map({"Stable": "#00cc96", "Unstable": "#ff4444"}).fillna("#888888")
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='color', color_discrete_map="identity", hover_data=['material'], custom_data=['material'])

    try:
        slope, intercept, r, _, _ = linregress(filtered_df[x_axis], filtered_df[y_axis])
        line_x = [x_min, x_max]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash'), name=f'R² = {r**2:.3f}'))
    except: pass

    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="plot")
    if clicked and clicked.get("selection", {}).get("points"):
        st.session_state.selected_material = clicked["selection"]["points"][0]["customdata"][0]
        st.session_state.show_3d = False  # reset

st.caption("MAX Phase 3D Explorer — Interactive 3D POSCAR Viewer • Neon Sliders • Full English • 2025")
