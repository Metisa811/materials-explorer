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
    # --- جدول تناوبی ---
    df_atomic = pd.read_csv("ptable2.csv")
    df_atomic['symbol'] = df_atomic['symbol'].str.strip()

    # --- داده‌های مکانیکی (ساختار جدید) ---
    with open("vaspkit_output.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []

    for material, data in raw.items():
        if not data:
            continue

        row = {"material": material}

        # --- ویژگی‌های اتمی ---
        # (بعداً با میانگین وزنی اضافه می‌شه)

        # --- Cij و Sij ---
        if "Elastic_Tensor_Voigt" in data:
            for k, v in data["Elastic_Tensor_Voigt"].items():
                row[k] = float(v) if v is not None else None
        if "Compliance_Tensor" in data:
            for k, v in data["Compliance_Tensor"].items():
                row[f"Compliance_{k}"] = float(v) if v is not None else None

        # --- آنیزوتروپیک (Min/Max/Anisotropy) ---
        if "Anisotropic_Mechanical_Properties" in data:
            for prop, vals in data["Anisotropic_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        if stat in vals:
                            row[f"{clean}_{stat}"] = float(vals[stat])

        # --- میانگین (Voigt/Reuss/Hill) ---
        if "Average_Mechanical_Properties" in data:
            for prop, vals in data["Average_Mechanical_Properties"].items():
                clean = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict):
                    for method in ["Voigt", "Reuss", "Hill"]:
                        if method in vals:
                            row[f"{clean}_{method}"] = float(vals[method])

        # --- خواص اضافی ---
        if "Additional_Properties" in data:
            for k, v in data["Additional_Properties"].items():
                if k == "Mechanical_Stability":
                    row["Mechanical_Stability"] = v
                elif k == "Brittleness_Indicator":
                    row["Brittleness"] = v
                else:
                    row[k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace(".", "").replace("-", "").isdigit() else v

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

    atomic_features = sorted([c for c in atomic_df.columns if c != 'material'])
    mechanical_properties = sorted([c for c in mech_df.columns if c != 'material' and c not in atomic_features])

    return df, atomic_features, mechanical_properties

# === اجرا ===
df, atomic_features, mechanical_properties = load_data()
if df.empty:
    st.error("No data loaded!")
    st.stop()

st.set_page_config(page_title="MAX Phase Explorer", layout="wide")
st.title("Interactive MAX Phase & Elastic Properties Explorer")

# --- سایدبار ---
with st.sidebar:
    st.header("Material Details")

    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        data = df[df['material'] == mat].iloc[0]
        stability = data.get("Mechanical_Stability", "Unknown")
        brittleness = data.get("Brittleness", "Unknown")

        # نمایش وضعیت پایداری با رنگ
        if stability == "Stable":
            st.success("Mechanically Stable")
        else:
            st.error("Mechanically Unstable")

        if brittleness == "Brittle":
            st.warning("Brittle")
        elif brittleness == "Ductile":
            st.info("Ductile")

        st.markdown(f"### **{mat}**")

        details = data.drop('material')

        # گروه‌بندی زیبا
        st.subheader("Atomic Properties")
        for k in sorted(atomic_features):
            if k in details:
                v = details[k]
                st.write(f"**{k}**: {float(v):.4f}" if pd.notna(v) else f"**{k}**: N/A")

        st.subheader("Elastic Constants (Cij)")
        cij_keys = [k for k in details.index if k.startswith("C")]
        for k in sorted(cij_keys):
            if pd.notna(details[k]):
                st.write(f"**{k}**: {float(details[k]):.3f} GPa")

        st.subheader("Compliance Tensor (Sij)")
        sij_keys = [k for k in details.index if k.startswith("Compliance_S")]
        for k in sorted(sij_keys):
            if pd.notna(details[k]):
                st.write(f"**{k.replace('Compliance_', '')}**: {float(details[k]):.6f} GPa⁻¹")

        st.subheader("Anisotropic Properties")
        aniso_keys = [k for k in details.index if "_Min" in k or "_Max" in k or "_Anisotropy" in k]
        for k in sorted(aniso_keys):
            if pd.notna(details[k]):
                st.write(f"**{k}**: {float(details[k]):.4f}")

        st.subheader("Average Properties (Hill)")
        avg_keys = [k for k in details.index if "_Hill" in k]
        for k in sorted(avg_keys):
            if pd.notna(details[k]):
                st.write(f"**{k.replace('_Hill', '')} (Hill)**: {float(details[k]):.4f}")

        if st.button("Clear Selection"):
            st.session_state.selected_material = None
            st.rerun()

    else:
        st.info("Click on a point to view full details")

# --- نمودار ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X-axis (Atomic)", atomic_features,
                          index=atomic_features.index('atomic_number') if 'atomic_number' in atomic_features else 0)
with col2:
    y_axis = st.selectbox("Y-axis (Elastic Property)", mechanical_properties, index=0)

plot_df = df[['material', x_axis, y_axis]].dropna()

if plot_df.empty:
    st.warning("No data for selected axes.")
else:
    fig = px.scatter(plot_df, x=x_axis, y=y_axis,
                     hover_data=['material'], custom_data=['material'],
                     color_discrete_sequence=['#00cc96'])

    if len(plot_df) > 5:
        try:
            slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
            line_x = [plot_df[x_axis].min(), plot_df[x_axis].max()]
            line_y = [slope * x + intercept for x in line_x]
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                     line=dict(color='red', dash='dash'),
                                     name=f'R² = {r**2:.3f}'))
        except:
            pass

    # کلیک پایدار و بدون پرش
    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="main_plot")

    if clicked and clicked.get("selection", {}).get("points"):
        selected_point = clicked["selection"]["points"][0]
        material_name = selected_point["customdata"][0]
        st.session_state.selected_material = material_name
        # بدون rerun اضافی — فقط نمایش بده
    # اگر کلیک خارج از نقطه بود، انتخاب پاک نشه مگر دکمه بزنن

st.caption("Professional MAX Phase Explorer — Full Cij/Sij • Anisotropic • Stability Indicator • Stable Selection")
