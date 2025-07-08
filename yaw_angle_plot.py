#!/usr/bin/env python3
"""
Full yaw PI dashboard
=====================
Handles CSV header:
  t_us,yaw_er, yaw_er_i, Omega.z, rate_er, rate_er_i, u_yaw

• Cleans spaces + dot, lower-cases names
• Converts rad and rad/s → deg and deg/s
• Three same-width Plotly graphs
"""

from pathlib import Path
import math
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# -------- paths / constants ---------------------------------------
LOG_DIR  = Path("./logs")
LOG_GLOB = "log_*.csv"
RAD2DEG  = 180.0 / math.pi

# -------- helper ---------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or "t_us" not in df.columns:
        return pd.DataFrame()

    # normalise header names
    df.columns = [c.strip().replace('.', '_').lower() for c in df.columns]

    # time baseline
    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6

    # convert to degrees / deg s-¹
    for col in ("yaw_er", "yaw_er_i",
                "omega_z", "rate_er", "rate_er_i"):
        if col in df.columns:
            df[col] = df[col] * RAD2DEG
    return df

def add_line(fig, x, y, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

# -------- Dash app -------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

opts = [{"label": p.name, "value": str(p)}
        for p in sorted(LOG_DIR.glob(LOG_GLOB))]
if not opts:
    opts = [{"label": "⚠  No logs found", "value": ""}]

app.layout = dbc.Container(
    [
        html.H4("Yaw-Controller Dashboard", className="my-3"),
        dcc.Dropdown(id="file-dd", options=opts, value=opts[0]["value"],
                     clearable=False, persistence=True,
                     style={"maxWidth": "480px"}),
        html.Small(id="file-info", className="d-block my-2 text-muted"),
        dcc.Graph(id="angle-fig", style={"height": 300}),
        dcc.Graph(id="rate-fig",  style={"height": 300, "marginTop": "10px"}),
        dcc.Graph(id="u-fig",     style={"height": 250, "marginTop": "10px"}),
    ], fluid=True)

# -------- callback -------------------------------------------------
@app.callback(
    [Output("file-info", "children"),
     Output("angle-fig", "figure"),
     Output("rate-fig",  "figure"),
     Output("u-fig",     "figure")],
    Input("file-dd", "value"),
    prevent_initial_call=False,
)
def update(file_path):
    if not file_path:
        raise dash.exceptions.PreventUpdate

    df = load_csv(Path(file_path))
    if df.empty:
        blank = go.Figure().update_layout(
            title="No data", xaxis_visible=False, yaxis_visible=False)
        return "Empty or malformed log.", blank, blank, blank

    # --- angle errors ---------------------------------------------
    ang_fig = go.Figure()
    add_line(ang_fig, df["t"], df["yaw_er"],   "yaw_err (deg)")
    add_line(ang_fig, df["t"], df["yaw_er_i"], "yaw_err_i (deg)")
    ang_fig.update_layout(
        title="Angle-loop errors",
        xaxis_title="time (s)",
        yaxis_title="deg",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0)
    )

    # --- rate loop -------------------------------------------------
    rate_fig = go.Figure()
    add_line(rate_fig, df["t"], df["omega_z"],     "omega_z (°/s)")
    add_line(rate_fig, df["t"], df["rate_er"],     "rate_err (°/s)")
    add_line(rate_fig, df["t"], df["rate_er_i"],   "rate_err_i (°/s)")
    rate_fig.update_layout(
        title="Rate-loop signals",
        xaxis_title="time (s)",
        yaxis_title="deg/s",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0)
    )

    # --- controller output ----------------------------------------
    u_fig = go.Figure()
    add_line(u_fig, df["t"], df["u_yaw"], "u_yaw")
    u_fig.update_layout(
        title="Controller output",
        xaxis_title="time (s)",
        yaxis_title="-1 … +1",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0)
    )

    info = (f"{Path(file_path).name} — {len(df):,d} rows, "
            f"{df['t'].iloc[-1]:.1f} s duration")
    return info, ang_fig, rate_fig, u_fig

# -------- run ------------------------------------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, host="0.0.0.0", port=8050)
