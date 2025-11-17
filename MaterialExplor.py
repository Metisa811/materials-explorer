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
        ptable_df = pd.read_csv("ptable2.csv")
        ptable_df['symbol'] = ptable_df['symbol'].str.strip()

        with open("vaspkit_output.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # --- استخراج همه خواص مکانیکی + Cij ---
        records = []
        cij_keys = [f"C{i}{j}" for i in range(1,7) for j in range(i,7)]  # C11, C12, ..., C44, C66

        for material, data in raw_data.items():
            if not data:
                continue

            row = {"material": material}

            # --- خواص آنیزوتروپیک (Min, Max, Anisotropy) ---
            if "Anisotropic_Mechanical_Properties" in data:
                amp = data["Anisotropic_Mechanical_Properties"]
                for key, val in amp.items():
                    clean_key = key.replace("|__", "").strip()
                    if isinstance(val, dict):
                        for stat in ["Min", "Max", "Anisotropy"]:
                            if stat in val:
                                row[f"{clean_key}_{stat}"] = val[stat]
                    else:
                        row[clean_key] = val

            # --- خواص اضافی ---
            if "Additional_Properties" in data:
                for k, v in data["Additional_Properties"].items():
                    row[k] = v

            # --- ماتریس الاستیک Cij (اگر وجود داشته باشه) ---
            if "Elastic_Tensor_Voigt" in data:
                tensor = data["Elastic_Tensor_Voigt"]
                for key in cij_keys:
                    if key in tensor:
                        row[key] = tensor[key]

            records.append(row)

        mech_df = pd.DataFrame(records)
        if mech_df.empty:
            st.error("هیچ داده مکانیکی پیدا نشد!")
            return pd.DataFrame(), [], []

        # --- تجزیه فرمول با عدد (Cr2PbC → Cr:2, Pb:1, C:1) ---
        def parse_formula(formula):
            matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
            return [(elem, int(count) if count else 1) for elem, count in matches]

        # --- میانگین وزنی ویژگی‌های اتمی ---
        feature_cols = [c for c in ptable_df.columns if c not in ['symbol']]
        atomic_avg_data = []

        for _, row in mech_df.iterrows():
            material = row['material']
            try:
                elements = parse_formula(material)
            except:
                continue

            weighted = {col: 0.0 for col in feature_cols}
            total_atoms = 0

            for elem, count in elements:
                elem_row = ptable_df[ptable_df['symbol'] == elem]
                if elem_row.empty:
                    break
                vals = elem_row.iloc[0]
                for col in feature_cols:
                    val = pd.to_numeric(vals[col], errors='coerce')
                    if not pd.isna(val):
                        weighted[col] += val * count
                total_atoms += count
            else:
                if total_atoms > 0:
                    avg_row = {col: weighted[col] / total_atoms for col in feature_cols}
                    avg_row['material'] = material
                    atomic_avg_data.append(avg_row)

        atomic_df = pd.DataFrame(atomic_avg_data)
        if atomic_df.empty:
            st.error("هیچ ویژگی اتمی محاسبه نشد!")
            return pd.DataFrame(), [], []

        # --- ادغام ---
        df = pd.merge(atomic_df, mech_df, on='material', how='inner')

        # --- لیست ویژگی‌های اتمی (X) ---
        atomic_features = sorted([c for c in atomic_df.columns if c != 'material'])

        # --- لیست همه خواص مکانیکی (Y) — شامل Cij و Bulk و Poisson منفی ---
        mech_cols = [c for c in mech_df.columns if c not in ['material']]
        mechanical_properties = sorted(mech_cols)

        return df, atomic_features, mechanical_properties

    except Exception as e:
        st.error(f"خطا در بارگذاری داده: {e}")
        return pd.DataFrame(), [], []

# === اجرای برنامه ===
df, atomic_features, mechanical_properties = load_data()
if df.empty:
    st.stop()

st.set_page_config(page_title="Materials Elastic Explorer", layout="wide")
st.title("Interactive Materials Elastic & Atomic Property Explorer")
st.markdown("**کلیک روی نقطه → نمایش تمام خواص شامل Cij و Poisson منفی**")

# --- سایدبار ---
with st.sidebar:
    st.header("جزئیات ماده")
    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        data = df[df['material'] == mat].iloc[0]
        st.success(f"**{mat}**")

        # مرتب‌سازی: اول خواص اتمی، بعد مکانیکی، بعد Cij
        details = data.drop('material')
        atomic_keys = [k for k in atomic_features if k in details.index]
        mech_keys = [k for k in mechanical_properties if k in details.index]

        st.subheader("ویژگی‌های اتمی")
        for k in atomic_keys:
            v = details[k]
            if pd.isna(v): v = "N/A"
            elif isinstance(v, (int, float)): v = f"{v:.4f}"
            st.write(f"**{k}**: {v}")

        st.subheader("خواص الاستیک")
        for k in mech_keys:
            v = details[k]
            if pd.isna(v): v = "N/A"
            elif isinstance(v, (int, float)): v = f"{v:.4f}"
            st.write(f"**{k}**: {v}")

        if st.button("پاک کردن انتخاب"):
            if "selected_material" in st.session_state:
                del st.session_state.selected_material
            st.rerun()
    else:
        st.info("روی یک نقطه کلیک کنید")

# --- انتخاب محورها ---
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox(
        "ویژگی اتمی (X)",
        options=atomic_features,
        index=atomic_features.index('atomic_number') if 'atomic_number' in atomic_features else 0
    )
with col2:
    y_axis = st.selectbox(
        "خواص الاستیک (Y) — شامل Cij و Poisson منفی",
        options=mechanical_properties,
        index=0
    )

# --- نمودار ---
plot_df = df[['material', x_axis, y_axis]].dropna()

if plot_df.empty:
    st.warning("داده کافی برای رسم وجود ندارد.")
else:
    fig = px.scatter(
        plot_df, x=x_axis, y=y_axis,
        hover_data=['material'],
        custom_data=['material'],
        color_discrete_sequence=['#00CC96']
    )

    # خط رگرسیون
    if len(plot_df) > 2 and plot_df[x_axis].nunique() > 1:
        slope, intercept, r, _, _ = linregress(plot_df[x_axis], plot_df[y_axis])
        line_x = np.array([plot_df[x_axis].min(), plot_df[x_axis].max()])
        line_y = slope * line_x + intercept
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 line=dict(color='red', dash='dash'),
                                 name=f'R² = {r**2:.3f}'))

    # کلیک روی نقطه
    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="main_plot")

    if clicked and clicked["selection"]["points"]:
        material = clicked["selection"]["points"][0]["customdata"][0]
        st.session_state.selected_material = material
    elif "selected_material" in st.session_state:
        # اگر کلیک روی فضای خالی بود
        del st.session_state.selected_material
        st.rerun()

# --- فوتر ---
st.markdown("---")
st.caption("نسخه حرفه‌ای — Poisson منفی، Cij، Bulk Modulus، میانگین وزنی دقیق")
