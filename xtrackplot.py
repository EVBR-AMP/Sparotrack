#!/usr/bin/env python3
"""
X-axis controller dashboard
===========================

Expects CSV header:
    t_us, x_req, Velocity.x, i_x, u_x
Creates two graphs:
    • x_req vs velocity_x
    • integrator (i_x) and actuator (u_x)
"""

from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go

# ---------- paths --------------------------------------------------
LOG_DIR  = Path("./logs")
LOG_GLOB = "log_*.csv"

# ---------- helper -------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")  # skip malformed
    if df.empty or "t_us" not in df.columns:
        return pd.DataFrame()

    # trim / lower-case / replace '.' with '_'
    df.columns = [c.strip().replace('.', '_').lower() for c in df.columns]

    # keep only first 5 columns if extra appear
    df = df.iloc[:, :5]

    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6   # μs → s
    return df

def add_line(fig, x, y, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

# ---------- Dash app ----------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

opts = [{"label": p.name, "value": str(p)}
        for p in sorted(LOG_DIR.glob(LOG_GLOB))]
if not opts:
    opts = [{"label": "⚠  No logs found", "value": ""}]

app.layout = dbc.Container(
    [
        html.H4("X-Controller Log Viewer", className="my-3"),
        dcc.Dropdown(id="file-dd", options=opts,
                     value=opts[0]["value"], clearable=False,
                     persistence=True, style={"maxWidth": "480px"}),
        html.Small(id="file-info", className="d-block my-2 text-muted"),
        dcc.Graph(id="track-fig", style={"height": 300}),
        dcc.Graph(id="ctrl-fig",  style={"height": 300, "marginTop": "10px"}),
    ], fluid=True)

# ---------- callback ----------------------------------------------
@app.callback(
    [Output("file-info", "children"),
     Output("track-fig", "figure"),
     Output("ctrl-fig",  "figure")],
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
        return "Empty or malformed log.", blank, blank

    # ---------- tracking plot --------------------------------------
    trk_fig = go.Figure()
    add_line(trk_fig, df["t"], df["x_req"],       "x_req")
    add_line(trk_fig, df["t"], df["velocity_x"],  "velocity_x")
    trk_fig.update_layout(
        title="Set-point vs Velocity",
        xaxis_title="time (s)",
        yaxis_title="meters  /  m·s⁻¹",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0)
    )

    # ---------- controller plot ------------------------------------
    ctrl_fig = go.Figure()
    add_line(ctrl_fig, df["t"], df["i_x"], "i_x")
    add_line(ctrl_fig, df["t"], df["u_x"], "u_x")
    ctrl_fig.update_layout(
        title="Integrator & u_x",
        xaxis_title="time (s)",
        yaxis_title="value",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0)
    )

    info = (f"{Path(file_path).name} — {len(df):,d} rows, "
            f"{df['t'].iloc[-1]:.1f} s duration")
    return info, trk_fig, ctrl_fig

# ---------- run ----------------------------------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, host="0.0.0.0", port=8050)
