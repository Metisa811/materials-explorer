import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import numpy as np

@st.cache_data
def load_data():
    df_atomic = pd.read_csv("ptable2.csv")
    df_atomic['symbol'] = df_atomic['symbol'].str.strip()

    with open("vaspkit_output.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []

    for material, data in raw.items():
        if not data:
            continue

        row = {"material": material}

        # --- Cij و Sij ---
        if "Elastic_Tensor_Voigt" in data:
            for k, v in data["Elastic_Tensor_Voigt"].items():
                try:
                    row[k] = float(v)
                except:
                    row[k] = None
        if "Compliance_Tensor" in data:
            for k, v in data["Compliance_Tensor"].items():
                try:
                    row[f"S_{k}"] = float(v)
                except:
                    row[f"S_{k}"] = None

        # --- آنیزوتروپیک ---
        if "Anisotropic_Mechanical_Properties" in data:
            for prop, vals in data["Anisotropic_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_").replace("_(GPa)", "")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        if stat in vals:
                            try:
                                row[f"{clean}_{stat}"] = float(vals[stat])
                            except:
                                pass

        # --- میانگین (Hill) ---
        if "Average_Mechanical_Properties" in data:
            for prop, vals in data["Average_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_").replace("_(GPa)", "")
                if isinstance(vals, dict) and "Hill" in vals:
                    try:
                        row[f"{clean}_Hill"] = float(vals["Hill"])
                    except:
                        pass

        # --- خواص اضافی ---
        if "Additional_Properties" in data:
            for k, v in data["Additional_Properties"].items():
                if k in ["Mechanical_Stability", "Brittleness_Indicator"]:
                    row[k] = v
                else:
                    try:
                        row[k] = float(v)
                    except:
                        row[k] = v

        records.append(row)

    mech_df = pd.DataFrame(records)

    # --- میانگین وزنی اتمی ---
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

    # همه ستون‌های عددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = [c for c in df.columns if c in numeric_cols or c in ["Mechanical_Stability", "Brittleness_Indicator"]]

    return df, sorted(all_features)

# === اجرا ===
df, all_features = load_data()
if df.empty:
    st.error("No data loaded!")
    st.stop()

st.set_page_config(page_title="MAX Phase Explorer Pro", layout="wide")
st.title("MAX Phase & Elastic Properties Explorer Pro")
st.markdown("**X-axis و Y-axis: همه ویژگی‌های اتمی + الاستیک (Cij, Sij, Min/Max, Hill, ...)**")

# --- سایدبار: نمایش همه مشخصات ---
with st.sidebar:
    st.header("Material Details")

    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        row = df[df['material'] == mat].iloc[0]

        # وضعیت پایداری و شکنندگی
        stability = row.get("Mechanical_Stability", "Unknown")
        if stability == "Stable":
            st.success("Mechanically Stable")
        else:
            st.error("Mechanically Unstable")

        brittleness = row.get("Brittleness_Indicator", "Unknown")
        if brittleness == "Brittle":
            st.warning("Brittle Material")
        elif brittleness == "Ductile":
            st.info("Ductile Material")

        st.markdown(f"### **{mat}**")

        # نمایش همه مشخصات — مرتب و کامل
        details = row.drop("material")

        # 1. ویژگی‌های اتمی
        atomic_keys = [c for c in details.index if c in ["atomic_number", "density", "melting_point", "atomic_radius", "electronegativity"]]
        if atomic_keys:
            st.subheader("Atomic Properties")
            for k in sorted(atomic_keys):
                if pd.notna(details[k]):
                    st.write(f"**{k.replace('_', ' ')}**: {float(details[k]):.4f}")

        # 2. ضرایب الاستیک Cij
        cij_keys = [c for c in details.index if c.startswith("C") and not c.endswith(("Min", "Max", "Anisotropy", "Hill"))]
        if cij_keys:
            st.subheader("Elastic Constants (GPa)")
            for k in sorted(cij_keys):
                if pd.notna(details[k]):
                    st.write(f"**{k}**: {float(details[k]):.3f}")

        # 3. Compliance Tensor
        sij_keys = [c for c in details.index if c.startswith("S_")]
        if sij_keys:
            st.subheader("Compliance Tensor (GPa⁻¹)")
            for k in sorted(sij_keys):
                if pd.notna(details[k]):
                    st.write(f"**{k.replace('S_', '')}**: {float(details[k]):.6f}")

        # 4. آنیزوتروپیک
        aniso_keys = [c for c in details.index if any(x in c for x in ["_Min", "_Max", "_Anisotropy"])]
        if aniso_keys:
            st.subheader("Anisotropic Properties")
            for k in sorted(aniso_keys):
                if pd.notna(details[k]):
                    st.write(f"**{k.replace('_', ' ')}**: {float(details[k]):.4f}")

        # 5. میانگین (Hill)
        hill_keys = [c for c in details.index if "_Hill" in c]
        if hill_keys:
            st.subheader("Average Properties (Hill Approximation)")
            for k in sorted(hill_keys):
                if pd.notna(details[k]):
                    st.write(f"**{k.replace('_Hill', '').replace('_', ' ')}**: {float(details[k]):.4f}")

        # 6. خواص اضافی
        extra_keys = ["Pughs_Ratio_B_div_G", "Cauchy_Pressure_Pc_GPa", "Debye_Temperature_K", "Universal_Elastic_Anisotropy"]
        if any(k in details.index for k in extra_keys):
            st.subheader("Additional Properties")
            for k in extra_keys:
                if k in details and pd.notna(details[k]):
                    st.write(f"**{k.replace('_', ' ')}**: {float(details[k]):.4f}")

        if st.button("Clear Selection"):
            st.session_state.selected_material = None
            st.rerun()

    else:
        st.info("Click on a point to see **all properties**")

# --- انتخاب محورها ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X-axis → Any Property", options=all_features,
                          index=all_features.index("atomic_number") if "atomic_number" in all_features else 0)
with col2:
    y_axis = st.selectbox("Y-axis → Any Property", options=all_features,
                          index=all_features.index("C44") if "C44" in all_features else 0)

# --- نمودار ---
plot_df = df[['material', x_axis, y_axis]].dropna()

if len(plot_df) < 2:
    st.warning("Not enough data.")
else:
    fig = px.scatter(plot_df, x=x_axis, y=y_axis, hover_data=['material'], custom_data=['material'],
                     color_discrete_sequence=['#00cc96'], opacity=0.85)

    try:
        slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
        line_x = [plot_df[x_axis].min(), plot_df[x_axis].max()]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 line=dict(color='red', dash='dash'), name=f'R² = {r**2:.3f}'))
    except:
        pass

    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="plot")

    if clicked and clicked.get("selection", {}).get("points"):
        mat_name = clicked["selection"]["points"][0]["customdata"][0]
        st.session_state.selected_material = mat_name

st.caption("MAX Phase Explorer Pro — Full Details on Click • All Elastic & Atomic Properties in X/Y • Professional & Complete")
