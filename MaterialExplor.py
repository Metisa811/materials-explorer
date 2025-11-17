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
    cij_keys = [f"C{i}{j}" for i in range(1,7) for j in range(i,7)]

    for material, data in raw.items():
        if not data:
            continue

        row = {"material": material}

        # --- Cij Tensor ---
        if "Elastic_Tensor_Voigt" in data:
            for k, v in data["Elastic_Tensor_Voigt"].items():
                row[k] = v

        # --- Anisotropic Properties (Min/Max/Anisotropy) ---
        if "Anisotropic_Mechanical_Properties" in data:
            for prop, vals in data["Anisotropic_Mechanical_Properties"].items():
                clean_prop = prop.replace("’", "").replace("'", "").replace(" ", "_")
                if isinstance(vals, dict):
                    for stat in ["Min", "Max", "Anisotropy"]:
                        key = f"{clean_prop}_{stat}"
                        if stat in vals:
                            row[key] = vals[stat]

        # --- Additional Properties (بدون تکرار) ---
        if "Additional_Properties" in data:
            for k, v in data["Additional_Properties"].items():
                # فقط اگر قبلاً اضافه نشده باشه
                if k not in row:
                    row[k] = v

        records.append(row)

    mech_df = pd.DataFrame(records)

    # --- Weighted Atomic Average ---
    def parse_formula(f):
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', f)
        return [(e, int(c) if c else 1) for e, c in matches]

    feature_cols = [c for c in df_atomic.columns if c != 'symbol']
    atomic_data = []

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
            if erow.empty:
                break
            vals = erow.iloc[0]
            for col in feature_cols:
                val = pd.to_numeric(vals[col], errors='coerce')
                if not pd.isna(val):
                    weighted[col] += val * count
            total += count
        else:
            if total > 0:
                avg_row = {col: weighted[col] / total for col in feature_cols}
                avg_row['material'] = material
                atomic_data.append(avg_row)

    atomic_df = pd.DataFrame(atomic_data)
    df = pd.merge(atomic_df, mech_df, on='material', how='inner')

    # --- لیست نهایی بدون تکرار ---
    atomic_features = sorted([c for c in atomic_df.columns if c != 'material'])
    
    # حذف تکراری‌ها و مرتب‌سازی
    mech_cols = list(dict.fromkeys([c for c in mech_df.columns if c != 'material']))  # حذف تکرار
    mechanical_properties = sorted([c for c in mech_cols if c not in atomic_features])

    return df, atomic_features, mechanical_properties

# === Main App ===
df, atomic_features, mechanical_properties = load_data()
if df.empty:
    st.error("No data loaded!")
    st.stop()

st.set_page_config(page_title="Materials Elastic Explorer", layout="wide")
st.title("Interactive Materials Elastic & Atomic Property Explorer")
st.markdown("**Click on a point → Full details including negative Poisson's ratio and Cij tensor**")

# --- Sidebar ---
with st.sidebar:
    st.header("Material Details")
    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        data = df[df['material'] == mat].iloc[0]
        st.success(f"**{mat}**")

        details = data.drop('material')

        # گروه‌بندی و نمایش ایمن
        st.subheader("Atomic Properties")
        for k in sorted([c for c in atomic_features if c in details.index]):
            v = details[k]
            if pd.isna(v):
                st.write(f"**{k}**: N/A")
            else:
                st.write(f"**{k}**: {float(v):.4f}")

        st.subheader("Elastic & Mechanical Properties")
        elastic_keys = sorted([c for c in mechanical_properties if c in details.index])
        for k in elastic_keys:
            v = details[k]
            if pd.isna(v):
                st.write(f"**{k}**: N/A")
            elif isinstance(v, (int, float, np.number)):
                st.write(f"**{k}**: {float(v):.4f}")
            else:
                st.write(f"**{k}**: {v}")

        if st.button("Clear Selection"):
            st.session_state.pop("selected_material", None)
            st.rerun()
    else:
        st.info("Click on a point to view details")

# --- Plot ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox(
        "X-axis (Atomic Feature)",
        options=atomic_features,
        index=atomic_features.index('atomic_number') if 'atomic_number' in atomic_features else 0
    )
with col2:
    # فقط یک بار Poisson's Ratio v Min
    y_options = [c for c in mechanical_properties if "Poissons_Ratio_v_Min" in c or c != "Poissons_Ratio_v_Min"]
    y_default = next((c for c in y_options if "Poissons_Ratio_v_Min" in c), y_options[0] if y_options else None)
    
    y_axis = st.selectbox(
        "Y-axis (Elastic Property)",
        options=y_options,
        index=y_options.index(y_default) if y_default in y_options else 0
    )

plot_df = df[['material', x_axis, y_axis]].dropna()

if plot_df.empty:
    st.warning("No data available for selected axes.")
else:
    fig = px.scatter(
        plot_df, x=x_axis, y=y_axis,
        hover_data=['material'], custom_data=['material'],
        color_discrete_sequence=['#00cc96']
    )

    if len(plot_df) > 3:
        try:
            slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
            x_line = np.array([plot_df[x_axis].min(), plot_df[x_axis].max()])
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                   line=dict(color='red', dash='dash'),
                                   name=f'R² = {r**2:.3f}'))
        except:
            pass

    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="plot")

    if clicked and clicked["selection"]["points"]:
        mat_name = clicked["selection"]["points"][0]["customdata"][0]
        st.session_state.selected_material = mat_name
    elif "selected_material" in st.session_state:
        st.session_state.pop("selected_material", None)
        st.rerun()

# Footer
st.markdown("---")
st.caption("Professional Materials Explorer — Fixed: No duplicates • Safe formatting • Negative Poisson fully supported")
