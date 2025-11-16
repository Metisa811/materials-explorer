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
    try:
        # --- خواندن جدول تناوبی ---
        ptable_df = pd.read_csv("ptable2.csv")
        ptable_df['symbol'] = ptable_df['symbol'].str.strip()

        # --- خواندن داده‌های مکانیکی ---
        with open("vaspkit_output.json", "r", encoding="utf-8") as f:
            mech_data_nested = json.load(f)

        mechanical_properties_list = []
        anisotropic_keys = ["Linear_compressibility", "Poisson's_Ratio_v", "Shear_Modulus_G_(GPa)",
                            "Young's_Modulus_E_(GPa)", "Bulk_Modulus_B_(GPa)"]
        additional_keys = ["Pughs_Ratio_B_div_G", "Cauchy_Pressure_Pc_GPa", "Kleinmans_Parameter",
                           "Universal_Elastic_Anisotropy", "Chung_Buessem_Anisotropy",
                           "Isotropic_Poissons_Ratio", "Wave_Velocity_Longitudinal",
                           "Wave_Velocity_Transverse", "Wave_Velocity_Average", "Debye_Temperature_K",
                           "Brittleness_Indicator", "Mechanical_Stability","Bulk_Modulus_B_(GPa)"]

        for material, data in mech_data_nested.items():
            if not data: continue
            props = {"material": material}
            if "Anisotropic_Mechanical_Properties" in data:
                for key in anisotropic_keys:
                    clean_key = key.lstrip('|__')
                    found_key = key if key in data["Anisotropic_Mechanical_Properties"] else clean_key
                    if found_key in data["Anisotropic_Mechanical_Properties"]:
                        d = data["Anisotropic_Mechanical_Properties"][found_key]
                        if isinstance(d, dict):
                            for stat in ["Min", "Max", "Anisotropy"]:
                                props[f"{clean_key}_{stat}"] = d.get(stat)
            if "Additional_Properties" in data:
                for key in additional_keys:
                    props[key] = data["Additional_Properties"].get(key)
            mechanical_properties_list.append(props)

        mech_df = pd.DataFrame(mechanical_properties_list)
        if mech_df.empty:
            st.error("Mechanical data is empty.")
            return pd.DataFrame(), [], []

        # --- تجزیه فرمول با عدد ---
        def parse_formula(formula):
            pattern = r'([A-Z][a-z]?)(\d*)'
            matches = re.findall(pattern, formula)
            elements = []
            for elem, count in matches:
                count = int(count) if count else 1
                elements.append((elem, count))
            return elements

        # --- محاسبه میانگین وزنی ---
        feature_cols = [c for c in ptable_df.columns if c not in ['symbol']]
        materials_data = []

        for _, row in mech_df.iterrows():
            material = row['material']
            try:
                parsed = parse_formula(material)
            except:
                continue

            weighted_values = {col: 0.0 for col in feature_cols}
            total_atoms = 0

            for elem, count in parsed:
                elem_row = ptable_df[ptable_df['symbol'] == elem]
                if elem_row.empty:
                    break
                row_values = elem_row.iloc[0]
                for col in feature_cols:
                    val = pd.to_numeric(row_values[col], errors='coerce')
                    if not pd.isna(val):
                        weighted_values[col] += val * count
                total_atoms += count
            else:
                if total_atoms > 0:
                    averaged = {col: weighted_values[col] / total_atoms for col in feature_cols}
                    averaged['material'] = material
                    materials_data.append(averaged)

        features_avg_df = pd.DataFrame(materials_data)
        if features_avg_df.empty:
            st.error("Atomic features are empty.")
            return pd.DataFrame(), [], []

        merged_df = pd.merge(features_avg_df, mech_df, on='material', how='inner')
        atomic_features = sorted([c for c in features_avg_df.columns if c not in ['material']])
        mechanical_properties = sorted([c for c in mech_df.columns if c not in ['material', 'Brittleness_Indicator', 'Mechanical_Stability']])

        return merged_df, atomic_features, mechanical_properties

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], []

# --- اجرا ---
df, atomic_features, mechanical_properties = load_data()
if df.empty:
    st.stop()

st.set_page_config(page_title="Materials Explorer", layout="wide")
st.title("Interactive Materials Property Explorer")

# --- سایدبار ---
with st.sidebar:
    st.header("Material Details")
    if st.session_state.get("selected_material"):
        material = st.session_state.selected_material
        data = df[df['material'] == material].iloc[0]
        st.success(f"**{material}**")
        for key, value in data.drop('material').to_dict().items():
            if pd.isna(value):
                value = "N/A"
            elif isinstance(value, (int, float, np.number)):
                value = f"{value:.4f}"
            st.write(f"**{key}**: {value}")
    else:
        st.info("Click on a point to see details")

# --- نمودار ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("Atomic Feature (X)", atomic_features, index=atomic_features.index('atomic_number') if 'atomic_number' in atomic_features else 0)
with col2:
    y_axis = st.selectbox("Mechanical Property (Y)", mechanical_properties, index=0)

plot_df = df[['material', x_axis, y_axis]].dropna()
if plot_df.empty:
    st.warning("No valid data.")
else:
    fig = px.scatter(plot_df, x=x_axis, y=y_axis, hover_data=['material'], custom_data=['material'])
    if plot_df[x_axis].nunique() > 1:
        slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
        line_x = [plot_df[x_axis].min(), plot_df[x_axis].max()]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash'), name=f'R² = {r**2:.3f}'))

    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="scatter")
    if clicked and clicked["selection"]["points"]:
        material_name = clicked["selection"]["points"][0]["customdata"][0]
        st.session_state.selected_material = material_name
    elif "selected_material" in st.session_state:
        del st.session_state.selected_material

