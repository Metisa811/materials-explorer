import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import numpy as np

# ====================== DATA LOADING ======================
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

        # Cij, Sij, Anisotropic, Average, Additional
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

    # میانگین وزنی اتمی
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

    # همه ویژگی‌های عددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = sorted([c for c in numeric_cols if c != "material"])

    return df, all_features

df, all_features = load_data()
if df.empty:
    st.error("داده‌ای بارگذاری نشد!")
    st.stop()

# ====================== صفحه اصلی ======================
st.set_page_config(page_title="MAX Phase Explorer", layout="wide")
st.title("MAX Phase & Elastic Properties Explorer Pro")
st.markdown("**رنگ نقاط: سبز = پایدار | قرمز = ناپایدار**")

# ====================== استایل نئون آبی کم‌رنگ برای اسلایدرها ======================
st.markdown("""
<style>
    /* خط اسلایدر آبی نئونی کم‌رنگ و شفاف */
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #00ccff22, #00f2ff44) !important;
        height: 6px !important;
        border-radius: 5px;
    }
    /* دستگیره اسلایدر با گلو ملایم */
    .stSlider > div > div > div[role="slider"] {
        background: #00ccff !important;
        border: 2px solid #00f2ff !important;
        box-shadow: 0 0 12px #00f2ff !important;
        width: 20px !important;
        height: 20px !important;
    }
    .stSlider label {
        color: #e0f8ff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ====================== سایدبار ======================
with st.sidebar:
    st.header("جزئیات ماده")

    if st.session_state.get("selected_material"):
        mat = st.session_state.selected_material
        row = df[df['material'] = mat].iloc[0]

        # وضعیت پایداری و شکنندگی با رنگ
        stability = row.get("Mechanical_Stability", "Unknown")
        if stability == "Stable":
            st.success("Mechanically Stable")
        else:
            st.error("Mechanically Unstable")

        brittleness = row.get("Brittleness_Indicator", "Unknown")
        if brittleness == "Brittleness_Indicator":
            st.warning("Brittle")
        elif brittleness == "Ductile":
            st.info("Ductile")

        st.markdown(f"### **{mat}**")

        details = row.drop("material")
        for col in df.columns:
            if col == "material": continue
            v = details.get(col)
            if pd.isna(v): continue
            if isinstance(v, (int, float, np.number):
                st.write(f"**{col.replace('_', ' '}: {v:.4f}")
            else:
                st.write(f"**{col}: {v}")

        if st.button("پاک کردن انتخاب"):
            st.session_state.selected_material = None
            st.rerun()
    else:
        st.info("برای دیدن همه خواص روی یک نقطه کلیک کنید")

# ====================== انتخاب محورها ======================
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X-axis → هر ویژگی اتمی یا الاستیک", all_features,
                          index=all_features.index("atomic_number") if "atomic_number" in all_features else 0)
with col2:
    y_axis = st.selectbox("Y-axis → هر ویژگی اتمی یا الاستیک", all_features, index=0)

# ====================== اسلایدرهای رنج با نئون آبی ======================
col_a, col_b = st.columns(2)
with col_a:
    x_min, x_max = st.slider(
        f"رنج X ({x_axis.replace('_', ' ')}",
        min_value=float(df[x_axis].min(),
        max_value=float(df[x_axis].max(),
        value=(float(df[x_axis].min(), float(df[x_axis].max()),
        step=(df[x_axis].max() / 100,
        key="x_slider"
    )

with col_b:
    y_min, y_min, y_max = st.slider(
        f"رنج Y ({y_axis.replace('_', ')}",
        min_value=float(df[y_axis].min(),
        max_value=float(df[y_axis].max(),
        value=(float(df[y_axis].min(), float(df[y_axis].max()),
        step=(df[y_axis].max() / 100,
        key="y_slider"
    )

# ====================== فیلتر داده بر اساس رنج ===
filtered_df = df[
    (df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
    (df[y_axis] >= y_min) & (df[y_axis] <= y_max)
].copy()

filtered_df = filtered_df[['material', x_axis, y_axis, 'Mechanical_Stability']].dropna()

if filtered_df.empty:
    st.warning("هیچ داده‌ای در رنج انتخابی وجود ندارد.")
else:
    # رنگ بر اساس پایداری
    filtered_df['color'] = filtered_df['Mechanical_Stability'].map({
        "Stable": "#00cc96",  # سبز نئونی
        "Unstable": "#ff4444"  # قرمز
    }).fillna("#888888")

    fig = px.scatter(
        filtered_df, x=x_axis, y=y_axis,
        color='color',
        color_discrete_map="identity",
        hover_data=['material'],
        custom_data=['material'],
        labels={x_axis: x_axis.replace('_', ' '), y_axis: y_axis.replace('_', ' ')},
        color_disrete_map="identity"
    )

    # خط رگرسیون فقط روی داده‌های فیلتر شده
    try:
        slope, intercept, r, _, _ = linregress(filtered_df[x_axis], filtered['y_axis'])
        line_x = [x_min, x_max]
        line_y = [slope * x + intercept + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                         line=dict(color='red', dash='dash', width=3),
                         name=f'R² = {r**2:.3f}'))
    except:
        pass

    pass

    # کلیک پایدار
    clicked = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="main_plot")

    if clicked and clicked["selection"]["points"]:
        point = clicked["selection"]["points"][0]
        mat_name = point["customdata"][0]["customdata"][0]
        st.session_state.selected_material = mat_name

# ====================== فوتر ======================
st.markdown("---")
st.caption("MAX Phase Explorer Pro — اسلایدر نئونی آبی کم‌رنگ • رنگ بر اساس پایداری • نمایش همه خواص • 2025")
