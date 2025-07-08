#!/usr/bin/env python3
"""
Yaw-rate dashboard  (same-width plots)
=====================================

Works with CSV header variants such as:
    t_us,rate_req, Omega.z, rate_er, u_yaw
(rate columns are rad/s; converted to deg/s here)

• Dropdown for any log_*.csv in ./logs
• Top graph: rate_req, omega_z, rate_er      [deg/s]
• Bottom graph: u_yaw                        [-1 … +1]
"""

from pathlib import Path
import math

import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# ---------- paths / constants -------------------------------------
LOG_DIR  = Path("./logs")
LOG_GLOB = "log_*.csv"
RAD2DEG  = 180.0 / math.pi

# ---------- helpers ------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV, normalise headers, add t (s), convert rad/s → deg/s."""
    df = pd.read_csv(path)
    if df.empty or "t_us" not in df.columns:
        return pd.DataFrame()

    # normalise header names
    df.columns = [c.strip().replace('.', '_').lower() for c in df.columns]

    # time base in seconds
    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6

    # rates to deg/s
    for col in ("rate_req", "omega_z", "rate_er"):
        if col in df.columns:
            df[col] = df[col] * RAD2DEG
    return df


def add_line(fig, x, y, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

# ---------- dash app ----------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

log_opts = [{"label": p.name, "value": str(p)}
            for p in sorted(LOG_DIR.glob(LOG_GLOB))]
if not log_opts:
    log_opts = [{"label": "⚠  No logs found", "value": ""}]

app.layout = dbc.Container(
    [
        html.H4("Yaw-Rate Log Viewer", className="my-3"),
        dcc.Dropdown(id="file-dd", options=log_opts,
                     value=log_opts[0]["value"], clearable=False,
                     persistence=True, style={"maxWidth": "480px"}),
        html.Small(id="file-info", className="d-block my-2 text-muted"),
        dcc.Graph(id="rate-fig", style={"height": 350}),
        dcc.Graph(id="u-fig",    style={"height": 250, "marginTop": "10px"}),
    ], fluid=True)

# ---------- callback ----------------------------------------------
@app.callback(
    [Output("file-info", "children"),
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
        msg = "Empty or malformed log."
        blank = go.Figure().update_layout(
            title=msg, xaxis_visible=False, yaxis_visible=False)
        return msg, blank, blank

    # -------- top graph: rate signals ------------------------------
    rate_fig = go.Figure()
    add_line(rate_fig, df["t"], df["rate_req"], "rate_req (°/s)")
    add_line(rate_fig, df["t"], df["omega_z"],  "omega_z  (°/s)")
    add_line(rate_fig, df["t"], df["rate_er"],  "rate_er  (°/s)")
    rate_fig.update_layout(
        title="Yaw-rate signals",
        xaxis_title="time (s)",
        yaxis_title="deg/s",
        margin=dict(l=50, r=50, t=70, b=40),
        legend=dict(orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="left",   x=0)
    )

    # -------- bottom graph: controller output ----------------------
    u_fig = go.Figure()
    add_line(u_fig, df["t"], df["u_yaw"], "u_yaw")
    u_fig.update_layout(
        title="Controller output (u_yaw)",
        xaxis_title="time (s)",
        yaxis_title="-1 … +1",
        margin=dict(l=50, r=50, t=70, b=40),
        legend=dict(orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="left",   x=0)
    )

    info = (f"{Path(file_path).name} — {len(df):,d} rows, "
            f"{df['t'].iloc[-1]:.1f} s duration")
    return info, rate_fig, u_fig

# ---------- run ----------------------------------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, host="0.0.0.0", port=8050)
