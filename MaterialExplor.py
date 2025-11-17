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
                row[k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace(".", "").replace("-", "").replace("E", "").isdigit() else None
        if "Compliance_Tensor" in data:
            for k, v in data["Compliance_Tensor"].items():
                row[f"S_{k}"] = float(v) if isinstance(v, (int, float, str)) and str(v).replace(".", "").replace("-", "").replace("E", "").isdigit() else None

        # --- آنیزوتروپیک ---
        if "Anisotropic_Mechanical_Properties" in data:
            for prop, vals in data["Anisotropic_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        if stat in vals:
                            row[f"{clean}_{stat}"] = float(vals[stat])

        # --- میانگین (Hill اولویت دارد) ---
        if "Average_Mechanical_Properties" in data:
            for prop, vals in data["Average_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict) and "Hill" in vals:
                    row[f"{clean}_Hill"] = float(vals["Hill"])

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

    # --- همه ویژگی‌های عددی برای X و Y ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = ["material"]
    all_features += [c for c in numeric_cols if c != "material"]

    return df, all_features

# === اجرا ===
df, all_features = load_data()
if df.empty:
    st.error("No data loaded!")
    st.stop()

st.set_page_config(page_title="MAX Phase Explorer Pro", layout="wide")
st.title("MAX Phase & Elastic Properties Explorer Pro")
st.markdown("**X-axis و Y-axis هر دو شامل ویژگی‌های اتمی و الاستیک هستند**")

# --- سایدبار ---
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
            st.warning("Brittle Material")
        elif brittleness == "Ductile":
            st.info("Ductile Material")

        st.markdown(f"### **{mat}**")

        # نمایش مهم‌ترین خواص
        important = ["atomic_number", "C11", "C44", "Bulk_Modulus_B_(GPa)_Min", "Poissons_Ratio_v_Min", "Youngs_Modulus_E_(GPa)_Hill"]
        for key in important:
            if key in row and pd.notna(row[key]):
                st.write(f"**{key.replace('_', ' ')}**: {float(row[key]):.4f}")

        if st.button("Clear Selection"):
            st.session_state.selected_material = None
            st.rerun()
    else:
        st.info("Click on any point to see full details")

# --- انتخاب محورها (هر دو از همه ویژگی‌های عددی) ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox(
        "X-axis → Atomic or Elastic Property",
        options=all_features,
        index=all_features.index("atomic_number") if "atomic_number" in all_features else 0
    )
with col2:
    y_axis = st.selectbox(
        "Y-axis → Atomic or Elastic Property",
        options=all_features,
        index=all_features.index("C44") if "C44" in all_features else 0
    )

# --- نمودار ---
plot_df = df[['material', x_axis, y_axis]].dropna()

if plot_df.empty or len(plot_df) < 2:
    st.warning("Not enough data for selected axes.")
else:
    fig = px.scatter(
        plot_df, x=x_axis, y=y_axis,
        hover_data=['material'], custom_data=['material'],
        color_discrete_sequence=['#00cc96'],
        opacity=0.8
    )

    # خط رگرسیون
    try:
        slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
        line_x = [plot_df[x_axis].min(), plot_df[x_axis].max()]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 line=dict(color='red', dash='dash', width=2),
                                 name=f'R² = {r**2:.3f}'))
    except:
        pass

    # کلیک پایدار
    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="plot")

    if clicked and clicked.get("selection", {}).get("points"):
        point = clicked["selection"]["points"][0]
        selected_mat = point["customdata"][0]
        st.session_state.selected_material = selected_mat
    # اگر خارج از نقطه کلیک شد → انتخاب پاک نشود

st.caption("MAX Phase Explorer Pro — Full Freedom: X & Y from Atomic + All Elastic Properties (Cij, Sij, Min/Max, Hill, etc.)")
