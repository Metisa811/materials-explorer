import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc  # <-- 1. Import Bootstrap
import numpy as np
import os
import sys

# --- Helper Function for .exe file paths ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # _MEIPASS not set, so running in normal Python environment
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 1. Data Loading and Processing ---
def load_data():
    """
    Loads atomic and mechanical data, flattens the mechanical data,
    calculates atomic averages, and merges everything into a single DataFrame.
    """
    try:
        # --- Load Atomic Features ---
        ptable_path = resource_path("ptable2.csv")
        ptable_df = pd.read_csv(ptable_path)
        ptable_df.rename(columns={"symbol": "element"}, inplace=True)
        ptable_df['element'] = ptable_df['element'].str.strip()

        # --- Load and Parse VASPkit Mechanical Data ---
        json_path = resource_path("vaspkit_output.json")
        with open(json_path, "r", encoding="utf-8") as f: # Added encoding for safety
            mech_data_nested = json.load(f)

        mechanical_properties_list = []
        
        # Define the properties to extract
        anisotropic_keys = [
            "Linear_compressibility",
            "Poisson's_Ratio_v",
            "Shear_Modulus_G_(GPa)",
            "Young's_Modulus_E_(GPa)",
            "Bulk_Modulus_B_(GPa)"
        ]
        
        additional_keys = [
            "Pughs_Ratio_B_div_G", "Cauchy_Pressure_Pc_GPa", "Kleinmans_Parameter",
            "Universal_Elastic_Anisotropy", "Chung_Buessem_Anisotropy",
            "Isotropic_Poissons_Ratio", "Wave_Velocity_Longitudinal",
            "Wave_Velocity_Transverse", "Wave_Velocity_Average", "Debye_Temperature_K",
            "Brittleness_Indicator", "Mechanical_Stability"
        ]

        for material, data in mech_data_nested.items():
            if not data:  # Skip empty entries
                continue
                
            props = {"material": material}

            # 1. Flatten Anisotropic Properties
            if "Anisotropic_Mechanical_Properties" in data:
                for key in anisotropic_keys:
                    # Handle keys that might have a |__ prefix
                    clean_key = key.lstrip('|__')
                    found_key = None
                    if key in data["Anisotropic_Mechanical_Properties"]:
                        found_key = key
                    elif clean_key in data["Anisotropic_Mechanical_Properties"]:
                        found_key = clean_key
                    
                    if found_key and isinstance(data["Anisotropic_Mechanical_Properties"].get(found_key), dict):
                        for stat in ["Min", "Max", "Anisotropy"]:
                            props[f"{clean_key}_{stat}"] = data["Anisotropic_Mechanical_Properties"][found_key].get(stat)

            # 2. Flatten Additional Properties
            if "Additional_Properties" in data:
                for key in additional_keys:
                    props[key] = data["Additional_Properties"].get(key)
            
            mechanical_properties_list.append(props)

        mech_df = pd.DataFrame(mechanical_properties_list)
        if mech_df.empty:
            print("Warning: Mechanical data DataFrame is empty after parsing.")
            return pd.DataFrame(), [], []

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        print("Please make sure 'ptable2.csv' and 'vaspkit_output.json' are in the same directory as app.py.")
        return pd.DataFrame(), [], []
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame(), [], []

    # --- 3. Element Extraction and Averaging (Same as before) ---
    def extract_elements(formula):
        return re.findall(r'[A-Z][a-z]?', formula)

    feature_cols = [c for c in ptable_df.columns if c not in ['element']]
    materials_data = []

    for _, row in mech_df.iterrows():
        material = row['material']
        elements = extract_elements(material)
        sub_df = ptable_df[ptable_df['element'].isin(elements)]
        
        if len(sub_df) == 0 or len(sub_df) != len(elements):
            # Skip if any element is not found in ptable or if no elements are found
            continue
            
        averaged = sub_df[feature_cols].mean(numeric_only=True)
        averaged['material'] = material
        materials_data.append(averaged)

    features_avg_df = pd.DataFrame(materials_data)
    if features_avg_df.empty:
        print("Warning: Atomic features DataFrame is empty. Check ptable2.csv and material formulas.")
        return pd.DataFrame(), [], []

    # --- 4. Merging ---
    merged_df = pd.merge(features_avg_df, mech_df, on='material', how='inner')
    
    # --- 5. Define feature lists for dropdowns ---
    # Atomic features (X-axis)
    atomic_features = sorted([c for c in features_avg_df.columns if c not in ['material']])
    
    # Mechanical properties (Y-axis)
    mechanical_properties = sorted([c for c in mech_df.columns if c not in ['material', 'Brittleness_Indicator', 'Mechanical_Stability']])
    
    return merged_df, atomic_features, mechanical_properties

# --- Load data globally on app start ---
global_df, atomic_features, mechanical_properties = load_data()

if not global_df.empty:
    print(f"✅ Data loaded successfully. Final material count: {len(global_df)}")
else:
    print("❌ Data loading failed. Check console for errors.")

# --- 2. Initialize Dash App ---
# <-- 2. Add external_stylesheets=...
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Materials Property Explorer"

# --- 3. App Layout ---
app.layout = dbc.Container([  # <-- 3. Use dbc.Container for better layout
    
    # --- Header ---
    html.H1("Interactive Materials Property Explorer", style={'textAlign': 'center', 'color': '#333', 'paddingTop': '20px'}),
    html.Hr(),

    # --- Control Panel ---
    dbc.Card(  # <-- 4. Use dbc.Card for a nice container
        dbc.CardBody([
            dbc.Row([
                # X-Axis Dropdown
                dbc.Col([
                    html.Label("Select Atomic Feature (X-Axis):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': f, 'value': f} for f in atomic_features],
                        value=atomic_features[0] if atomic_features else None,
                        clearable=False
                    )
                ], width=6),

                # Y-Axis Dropdown
                dbc.Col([
                    html.Label("Select Mechanical Property (Y-Axis):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': p, 'value': p} for p in mechanical_properties],
                        value=mechanical_properties[0] if mechanical_properties else None,
                        clearable=False
                    )
                ], width=6),
            ])
        ]),
        style={'marginBottom': '20px'}
    ),

    # --- Graph ---
    dcc.Graph(id='scatter-plot', style={'height': '600px'}),

    # --- Click Data Modal ---
    # <-- 5. Replaced dcc.Modal with dbc.Modal
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Material Details"), close_button=True),
            dbc.ModalBody(
                dcc.Loading(
                    type="default",
                    children=[html.Div(id='modal-content')]
                )
            ),
        ],
        id="material-modal",
        is_open=False,
        size="lg",  # Large modal
        centered=True, # Center it vertically
    ),
], fluid=True, style={'backgroundColor': '#f9f9f9', 'minHeight': '100vh'}) # <-- 3. (continued)

# --- 4. Callbacks ---

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_graph(x_axis_name, y_axis_name):
    if not x_axis_name or not y_axis_name or global_df.empty:
        return go.Figure()

    # Create dataframe for plotting, drop NaNs
    plot_df = global_df[['material', x_axis_name, y_axis_name]].copy()
    plot_df[x_axis_name] = pd.to_numeric(plot_df[x_axis_name], errors='coerce')
    plot_df[y_axis_name] = pd.to_numeric(plot_df[y_axis_name], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_axis_name, y_axis_name])

    if plot_df.empty:
        return go.Figure().update_layout(title_text=f"No valid data for {x_axis_name} vs {y_axis_name}")

    # Create the scatter plot
    fig = px.scatter(
        plot_df,
        x=x_axis_name,
        y=y_axis_name,
        hover_data=['material'],
        custom_data=['material'] # Pass material name to clickData
    )

    # Add regression line
    r_val = None
    try:
        x = plot_df[x_axis_name]
        y = plot_df[y_axis_name]
        if x.nunique() > 1: # Check for more than one unique x value
            slope, intercept, r, p, stderr = linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            
            fig.add_trace(go.Scatter(
                x=line_x, y=line_y, mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'Regression (R² = {r**2:.3f})'
            ))
            r_val = r
    except Exception as e:
        print(f"Could not compute regression: {e}")

    fig.update_layout(
        title=f'{y_axis_name} vs. {x_axis_name}',
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='#f9f9f9',
        font_color='#333',
        transition_duration=500
    )
    
    fig.update_xaxes(gridcolor='#eee', zerolinecolor='#ddd')
    fig.update_yaxes(gridcolor='#eee', zerolinecolor='#ddd')

    return fig


@app.callback(
    [Output('material-modal', 'is_open'),
     Output('modal-content', 'children')],
    [Input('scatter-plot', 'clickData')],
    [State('material-modal', 'is_open')]
)
def toggle_material_modal(clickData, is_open):
    # This callback is simplified. It only opens.
    # The 'close_button=True' in dbc.ModalHeader handles closing.
    if clickData:
        material_name = clickData['points'][0]['customdata'][0]
        
        # Get all data for this material
        material_data = global_df[global_df['material'] == material_name].iloc[0]
        
        # Create a table
        table_header = [html.Thead(html.Tr([html.Th("Property"), html.Th("Value")]))]
        table_body = []
        
        # Sort keys for a cleaner display
        sorted_keys = sorted([key for key in material_data.keys() if key != 'material'])
        
        for key in sorted_keys:
            value = material_data[key]
            # Round numeric values for cleaner display
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value):
                    value_str = "N/A"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            table_body.append(html.Tr([html.Td(key, style={'fontWeight': 'bold'}), html.Td(value_str)]))

        table = dbc.Table(table_header + [html.Tbody(table_body)], 
                          bordered=True, striped=True, hover=True, 
                          responsive=True, style={'marginTop': '15px'})
        
        content = [
            html.H3(material_name, style={'color': '#007bff'}),
            table
        ]
        
        return True, content

    return is_open, dash.no_update


# --- 5. Run the App ---
if __name__ == '__main__':
    if global_df.empty:
        print("="*50)
        print("Error: Data loading failed. The application will not run.")
        print("Please check that 'ptable2.csv' and 'vaspkit_output.json' are present")
        print("in the same folder as this script and are not empty.")
        print("="*50)
    else:
        print("Data loaded. Starting Dash server...")
        print("Access the app at: http://127.0.0.1:8050/")
        app.run(debug=True, port=8050)
