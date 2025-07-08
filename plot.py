#!/usr/bin/env python3
"""
Teensy flight-log dashboard
===========================

Visualises CSV logs that look like
    t_us,X,Y,H,VX,VY,u_x,u_y
produced by your SdFs logger.

• File selector for multiple logs
• Three linked plots: position, velocity, controller outputs
• Uses Dash 2.x  (Plotly under the hood)

Author: Elliot’s ChatGPT helper
Date: 2025-07-03
"""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go

LOG_DIR = Path("./logs")         # change if your logs live elsewhere
LOG_GLOB = "log_*.csv"

# ---------- helper -------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6  # µs ➜ s, t=0 start
    return df


def make_line(fig, x, y, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))


# ---------- Dash app ----------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server   # for gunicorn/Render deployments

log_options = [
    {"label": p.name, "value": str(p)} for p in sorted(LOG_DIR.glob(LOG_GLOB))
]
if not log_options:
    log_options = [{"label": "⚠  No logs found – place CSVs in ./logs", "value": ""}]

app.layout = dbc.Container(
    [
        html.H3("Teensy Flight-Log Dashboard", className="my-3"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="file-dropdown",
                        options=log_options,
                        value=log_options[0]["value"],
                        clearable=False,
                        persistence=True,
                    ),
                    width=6,
                ),
                dbc.Col(
                    html.Div(id="file-info", className="pt-2 text-muted"),
                    width="auto",
                ),
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="pos-graph"), width=12),
                dbc.Col(dcc.Graph(id="vel-graph"), width=12),
                dbc.Col(dcc.Graph(id="u-graph"),   width=12),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    [
        Output("file-info", "children"),
        Output("pos-graph", "figure"),
        Output("vel-graph", "figure"),
        Output("u-graph", "figure"),
    ],
    Input("file-dropdown", "value"),
    prevent_initial_call=False,
)
def update_dashboard(file_path):
    if not file_path:
        return dash.no_update

    path = Path(file_path)
    df = load_csv(path)

    # --- Position figure ---
    pos_fig = go.Figure()
    make_line(pos_fig, df["t"], df["X"], "X (m)")
    make_line(pos_fig, df["t"], df["Y"], "Y (m)")
    make_line(pos_fig, df["t"], df["H"], "H (m)")
    pos_fig.update_layout(
        title="Position vs time",
        xaxis_title="t (s)",
        yaxis_title="Position (m)",
        legend_title=None,
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )

    # --- Velocity figure ---
    vel_fig = go.Figure()
    make_line(vel_fig, df["t"], df["VX"], "VX (m/s)")
    make_line(vel_fig, df["t"], df["VY"], "VY (m/s)")
    vel_fig.update_layout(
        title="Velocity vs time",
        xaxis_title="t (s)",
        yaxis_title="Velocity (m/s)",
        legend_title=None,
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )

    # --- Control outputs figure ---
    u_fig = go.Figure()
    make_line(u_fig, df["t"], df["u_x"], "u_x")
    make_line(u_fig, df["t"], df["u_y"], "u_y")
    u_fig.update_layout(
        title="Controller outputs",
        xaxis_title="t (s)",
        yaxis_title="u (-1 … +1)",
        legend_title=None,
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )

    info = f"Loaded {path.name}  –  {len(df):,d} rows, " \
           f"{df['t'].iloc[-1]:.1f} s duration"

    return info, pos_fig, vel_fig, u_fig


if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, port=8050, host="127.0.0.1")