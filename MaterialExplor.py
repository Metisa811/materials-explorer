import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
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
    try:
        # --- Load Atomic Features ---
        ptable_path = resource_path("ptable2.csv")
        ptable_df = pd.read_csv(ptable_path)
        ptable_df.rename(columns={"symbol": "element"}, inplace=True)
        ptable_df['element'] = ptable_df['element'].str.strip()

        # --- Load and Parse VASPkit Mechanical Data ---
        json_path = resource_path("vaspkit_output.json")
        with open(json_path, "r", encoding="utf-8") as f:
            mech_data_nested = json.load(f)

        mechanical_properties_list = []
        
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
            if not data:
                continue
            props = {"material": material}

            if "Anisotropic_Mechanical_Properties" in data:
                for key in anisotropic_keys:
                    clean_key = key.lstrip('|__')
                    found_key = None
                    if key in data["Anisotropic_Mechanical_Properties"]:
                        found_key = key
                    elif clean_key in data["Anisotropic_Mechanical_Properties"]:
                        found_key = clean_key
                    
                    if found_key and isinstance(data["Anisotropic_Mechanical_Properties"].get(found_key), dict):
                        for stat in ["Min", "Max", "Anisotropy"]:
                            props[f"{clean_key}_{stat}"] = data["Anisotropic_Mechanical_Properties"][found_key].get(stat)

            if "Additional_Properties" in data:
                for key in additional_keys:
                    props[key] = data["Additional_Properties"].get(key)
            
            mechanical_properties_list.append(props)

        mech_df = pd.DataFrame(mechanical_properties_list)
        if mech_df.empty:
            return pd.DataFrame(), [], []

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        return pd.DataFrame(), [], []
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame(), [], []

    def extract_elements(formula):
        return re.findall(r'[A-Z][a-z]?', formula)

    feature_cols = [c for c in ptable_df.columns if c not in ['element']]
    materials_data = []

    for _, row in mech_df.iterrows():
        material = row['material']
        elements = extract_elements(material)
        sub_df = ptable_df[ptable_df['element'].isin(elements)]
        
        if len(sub_df) == 0 or len(sub_df) != len(elements):
            continue
            
        averaged = sub_df[feature_cols].mean(numeric_only=True)
        averaged['material'] = material
        materials_data.append(averaged)

    features_avg_df = pd.DataFrame(materials_data)
    if features_avg_df.empty:
        return pd.DataFrame(), [], []

    merged_df = pd.merge(features_avg_df, mech_df, on='material', how='inner')
    
    # Clean up stability column for consistent coloring
    if 'Mechanical_Stability' in merged_df.columns:
        merged_df['Mechanical_Stability'] = merged_df['Mechanical_Stability'].str.strip()

    atomic_features = sorted([c for c in features_avg_df.columns if c not in ['material']])
    mechanical_properties = sorted([c for c in mech_df.columns if c not in ['material', 'Brittleness_Indicator', 'Mechanical_Stability']])
    
    return merged_df, atomic_features, mechanical_properties

global_df, atomic_features, mechanical_properties = load_data()

# --- 2. Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Materials Property Explorer v2.0"

# --- 3. App Layout ---
app.layout = dbc.Container([
    
    html.H1("Interactive Materials Property Explorer v2.0", style={'textAlign': 'center', 'color': '#333', 'paddingTop': '20px'}),
    html.Hr(),

    # --- Main Controls ---
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Atomic Feature (X-Axis):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': f, 'value': f} for f in atomic_features],
                        value=atomic_features[0] if atomic_features else None,
                        clearable=False
                    )
                ], width=6),

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

    # --- Axis Range Controls ---
    dbc.Card(
        dbc.CardBody([
            html.Label("Manual Axis Range Control (Optional)", style={'fontWeight': 'bold', 'color': '#555'}),
            dbc.Row([
                # X Axis Controls
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("X Min"),
                        dbc.Input(id="x-min-input", type="number", placeholder="Auto"),
                    ], size="sm"),
                ], width=3),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("X Max"),
                        dbc.Input(id="x-max-input", type="number", placeholder="Auto"),
                    ], size="sm"),
                ], width=3),

                # Y Axis Controls
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("Y Min"),
                        dbc.Input(id="y-min-input", type="number", placeholder="Auto"),
                    ], size="sm"),
                ], width=3),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("Y Max"),
                        dbc.Input(id="y-max-input", type="number", placeholder="Auto"),
                    ], size="sm"),
                ], width=3),
            ], className="g-2", style={'marginBottom': '10px'}),
            
            dbc.Row([
                dbc.Col(
                    dbc.Button("Reset Ranges", id="reset-ranges-btn", color="secondary", size="sm", outline=True, style={'width': '100%'}),
                    width={"size": 2, "offset": 10}
                )
            ])
        ]),
        style={'marginBottom': '20px', 'backgroundColor': '#f0f0f0'}
    ),

    # --- Graph ---
    dcc.Loading(
        id="loading-1",
        type="default",
        children=dcc.Graph(id='scatter-plot', style={'height': '600px'})
    ),

    # --- Modal ---
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
        size="lg",
        centered=True,
    ),
], fluid=True, style={'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})

# --- 4. Callbacks ---

@app.callback(
    [Output("x-min-input", "value"),
     Output("x-max-input", "value"),
     Output("y-min-input", "value"),
     Output("y-max-input", "value")],
    [Input("reset-ranges-btn", "n_clicks"),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def reset_inputs(n_clicks, x_change, y_change):
    return None, None, None, None

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('x-min-input', 'value'),
     Input('x-max-input', 'value'),
     Input('y-min-input', 'value'),
     Input('y-max-input', 'value')]
)
def update_graph(x_axis_name, y_axis_name, x_min, x_max, y_min, y_max):
    if not x_axis_name or not y_axis_name or global_df.empty:
        return go.Figure()

    # Ensure Mechanical_Stability exists, otherwise fill with 'Unknown'
    cols_to_copy = ['material', x_axis_name, y_axis_name]
    if 'Mechanical_Stability' in global_df.columns:
        cols_to_copy.append('Mechanical_Stability')
    
    plot_df = global_df[cols_to_copy].copy()
    
    if 'Mechanical_Stability' not in plot_df.columns:
        plot_df['Mechanical_Stability'] = 'Unknown'

    plot_df[x_axis_name] = pd.to_numeric(plot_df[x_axis_name], errors='coerce')
    plot_df[y_axis_name] = pd.to_numeric(plot_df[y_axis_name], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_axis_name, y_axis_name])

    if plot_df.empty:
        return go.Figure().update_layout(title_text=f"No valid data for {x_axis_name} vs {y_axis_name}")

    # Create scatter plot
    # Logic: Map 'Stable' to Green, 'Unstable' to Red.
    fig = px.scatter(
        plot_df,
        x=x_axis_name,
        y=y_axis_name,
        color='Mechanical_Stability',
        color_discrete_map={
            'Stable': 'green',
            'Unstable': 'red',
            'Unknown': 'gray'
        },
        hover_data=['material'],
        custom_data=['material']
    )

    # Regression logic
    try:
        reg_x = plot_df[x_axis_name]
        reg_y = plot_df[y_axis_name]
        
        if reg_x.nunique() > 1:
            slope, intercept, r, p, stderr = linregress(reg_x, reg_y)
            
            # Calculate line limits based on auto range OR manual range
            x_start = reg_x.min()
            x_end = reg_x.max()
            
            # If manual ranges are wider, extend the line visual
            if x_min is not None: x_start = min(x_start, x_min)
            if x_max is not None: x_end = max(x_end, x_max)
            
            line_x_range = np.array([x_start, x_end])
            line_y_vals = slope * line_x_range + intercept
            
            fig.add_trace(go.Scatter(
                x=line_x_range, y=line_y_vals, mode='lines',
                line=dict(color='blue', dash='dash', width=2), # Blue regression line
                name=f'Regression (R² = {r**2:.3f})'
            ))
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

    if x_min is not None: fig.update_xaxes(range=[x_min, None if x_max is None else x_max])
    if x_max is not None: fig.update_xaxes(range=[None if x_min is None else x_min, x_max])
    if y_min is not None: fig.update_yaxes(range=[y_min, None if y_max is None else y_max])
    if y_max is not None: fig.update_yaxes(range=[None if y_min is None else y_min, y_max])
    
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
    if clickData:
        material_name = clickData['points'][0]['customdata'][0]
        material_data = global_df[global_df['material'] == material_name].iloc[0]
        
        table_header = [html.Thead(html.Tr([html.Th("Property"), html.Th("Value")]))]
        table_body = []
        
        sorted_keys = sorted([key for key in material_data.keys() if key != 'material'])
        
        for key in sorted_keys:
            value = material_data[key]
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value):
                    value_str = "N/A"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            # Highlight Stability row
            style = {'fontWeight': 'bold'}
            row_style = {}
            if key == 'Mechanical_Stability':
                if str(value).strip() == 'Stable':
                    row_style = {'backgroundColor': '#d4edda', 'color': '#155724'} # Light green
                elif str(value).strip() == 'Unstable':
                    row_style = {'backgroundColor': '#f8d7da', 'color': '#721c24'} # Light red

            table_body.append(html.Tr([html.Td(key, style=style), html.Td(value_str)], style=row_style))

        table = dbc.Table(table_header + [html.Tbody(table_body)], 
                          bordered=True, striped=True, hover=True, 
                          responsive=True, style={'marginTop': '15px'})
        
        content = [html.H3(material_name, style={'color': '#007bff'}), table]
        return True, content

    return is_open, dash.no_update

if __name__ == '__main__':
    if global_df.empty:
        print("="*50)
        print("Error: Data loading failed.")
        print("="*50)
    else:
        print("Data loaded. Starting Dash server...")
        # IMPORTANT: debug=False ensures stability and removes signal errors
        app.run(debug=False, port=8050)
