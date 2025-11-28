import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import numpy as np

# نصب خودکار py3Dmol
try:
    import py3Dmol
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py3Dmol"])
    import py3Dmol

# ====================== بارگذاری داده ======================
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
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        if stat in vals:
                            try: row[f"{clean}_{stat}"] = float(vals[stat])
                            except: pass
        if "Average_Mechanical_Properties" in data:
            for prop, vals in data["Average_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
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

    # میانگین اتمی
    def parse_formula(f):
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', f)
        return [(e, int(c) if c else 1) for e, c in matches]

    feature_cols = [c for c in df_atomic.columns if c != 'symbol']
    atomic_rows = []
    for _, row in mech_df.iterrows():
        material = row['material']
        try:
            elements = parse_formula(material)
        except: continue
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

# ====================== صفحه و استایل ======================
st.set_page_config(page_title="MAX Phase 3D Explorer", layout="wide")
st.title("MAX Phase & Elastic Properties Explorer Pro")
st.markdown("**Green = Stable | Red = Unstable | Click → View 3D Structure**")

# اسلایدر نئونی آبی کم‌رنگ
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #00ccff11, #00f2ff33) !important;
        height: 6px !important;
    }
    .stSlider > div > div > div[role="slider"] {
        background: #00ccff !important;
        border: 1.5px solid #00f2ff !important;
        box-shadow: 0 0 10px #00f2ff !important;
    }
    .stSlider label { color: #e0f8ff !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== سایدبار + 3D ویوور ======================
with st.sidebar:
    st.header("Material Details")

    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        row = df[df['material'] == mat].iloc[0]

        # وضعیت پایداری
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

        # دکمه 3D
        if st.button("View 3D Crystal Structure", type="primary", use_container_width=True):
            st.session_state.show_3d = True

        # نمایش همه خواص
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
            for key in ["selected_material", "show_3d"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.info("Click on a point to view details + 3D structure")

# ====================== ویوور 3D از poscars.txt ======================
if st.session_state.get("show_3d", False) and st.session_state.get("selected_material"):
    mat = st.session_state.selected_material
    try:
        with open("poscars.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # پیدا کردن بلوک مربوط به این ماده
        pattern = rf">>> {re.escape(mat)}\n(.*?)(?:\n>>> |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            poscar_data = match.group(1).strip()
            st.markdown(f"### 3D Structure of **{mat}**")

            view = py3Dmol.view(width=800, height=600)
            view.addModel(poscar_data, "poscar")
            view.setStyle({'stick': {'radius': 0.15, 'color': 'spectrum'}, 
                           'sphere': {'scale': 0.35}})
            view.setBackgroundColor('#111111')
            view.zoomTo()
            view.spin(True)
            view.show()
            st.components.v1.html(view._make_html(), width=800, height=600, scrolling=False)
        else:
            st.warning(f"Structure not found for {mat}")
    except FileNotFoundError:
        st.error("poscars.txt file not found! Please upload it.")

# ====================== نمودار اصلی ======================
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X-axis", all_features,
                          index=all_features.index("atomic_number") if "atomic_number" in all_features else 0)
with col2:
    y_axis = st.selectbox("Y-axis", all_features, index=0)

# اسلایدرهای رنج
col_a, col_b = st.columns(2)
with col_a:
    x_min, x_max = st.slider(f"X Range ({x_axis})", 
                             float(df[x_axis].min()), float(df[x_axis].max()),
                             (float(df[x_axis].min()), float(df[x_axis].max())))
with col_b:
    y_min, y_max = st.slider(f"Y Range ({y_axis})",
                             float(df[y_axis].min()), float(df[y_axis].max()),
                             (float(df[y_axis].min()), float(df[y_axis].max())))

# فیلتر داده
plot_df = df[
    (df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
    (df[y_axis] >= y_min) & (df[y_axis] <= y_max)
][['material', x_axis, y_axis, 'Mechanical_Stability']].dropna()

if plot_df.empty:
    st.info("No data in selected range.")
else:
    plot_df['color'] = plot_df['Mechanical_Stability'].map({
        "Stable": "#00cc96", "Unstable": "#ff4444"
    }).fillna("#888888")

    fig = px.scatter(plot_df, x=x_axis, y=y_axis,
                     color='color', color_discrete_map="identity",
                     hover_data=['material'], custom_data=['material'])

    # خط رگرسیون
    try:
        slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
        line_x = [x_min, x_max]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 line=dict(color='red', dash='dash', width=3),
                                 name=f'R² = {r**2:.3f}'))
    except: pass

    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="main_plot")
    if clicked and clicked.get("selection", {}).get("points"):
        st.session_state.selected_material = clicked["selection"]["points"][0]["customdata"][0]

st.caption("MAX Phase Explorer Pro — Single poscars.txt • 3D Viewer • Neon Sliders • Full English • 2025")
