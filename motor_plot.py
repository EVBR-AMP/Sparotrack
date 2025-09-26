#!/usr/bin/env python3
"""
Full-state log viewer  (rev G) — for header:
t_us, Position.x, Position.y, Height, Heading, Velocity.x, Velocity.y, OF_raw_x, OF_raw_y, Omega.z, u_x, u_y, u_yaw, m1, m2, m3, m4
"""

from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# ------------- configuration ----------------------------------------
LOG_DIR  = Path("./logs")
LOG_GLOB = "log_*.csv"

# ------------- helpers ----------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame()

    # normalize names: dots/spaces -> underscores, lower-case
    # e.g., "Position.x" -> "position_x", "OF_raw" -> "of_raw"
    df.columns = [c.strip().replace(".", "_").replace(" ", "_").lower() for c in df.columns]

    expected = [
        "t_us",
        "position_x", "position_y", "height", "heading",
        "velocity_x", "velocity_y", "of_raw_x", "of_raw_y", "omega_z",
        "u_x", "u_y", "u_yaw",
        "m1", "m2", "m3", "m4",
    ]

    if "t_us" not in df.columns:
        return pd.DataFrame()

    # Add missing columns as NaN to avoid KeyError; keep fixed order
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected]

    # numeric coercion
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows without timestamp
    df = df.dropna(subset=["t_us"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # μs → s, relative to first sample
    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6
    return df


def add_line(fig, x, y, name, **kw):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, **kw))

# ------------- Dash app ---------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

file_opts = [{"label": p.name, "value": str(p)} for p in sorted(LOG_DIR.glob(LOG_GLOB))]
if not file_opts:
    file_opts = [{"label": "⚠  No logs found", "value": ""}]

app.layout = dbc.Container(
    [
        html.H4("Full-State Dashboard (rev G)", className="my-3"),
        dcc.Dropdown(
            id="file-dd",
            options=file_opts,
            value=file_opts[0]["value"],
            clearable=False,
            persistence=True,
            style={"maxWidth": "520px"},
        ),
        html.Small(id="file-info", className="d-block my-2 text-muted"),
        dcc.Graph(id="pos-fig",    style={"height": 320}),
        dcc.Graph(id="rate-fig",   style={"height": 300, "marginTop": "10px"}),
        dcc.Graph(id="ctrl-fig",   style={"height": 300, "marginTop": "10px"}),
        dcc.Graph(id="motor-fig",  style={"height": 280, "marginTop": "10px"}),
    ],
    fluid=True,
)

# ------------- callback ---------------------------------------------
@app.callback(
    [
        Output("file-info",  "children"),
        Output("pos-fig",    "figure"),
        Output("rate-fig",   "figure"),
        Output("ctrl-fig",   "figure"),
        Output("motor-fig",  "figure"),
    ],
    Input("file-dd", "value"),
    prevent_initial_call=False,
)
def update_dash(file_path):
    if not file_path:
        raise dash.exceptions.PreventUpdate

    df = load_csv(Path(file_path))
    if df.empty:
        blank = go.Figure().update_layout(title="No data", xaxis_visible=False, yaxis_visible=False)
        return "Empty or malformed log.", blank, blank, blank, blank

    # Track any fully-missing series (all-NaN) to warn user
    expected_non_time = [
        "position_x", "position_y", "height", "heading",
        "velocity_x", "velocity_y", "of_raw_x", "of_raw_y", "omega_z",
        "u_x", "u_y", "u_yaw",
        "m1", "m2", "m3", "m4",
    ]
    missing = [c for c in expected_non_time if df[c].isna().all()]

    # ---------- Fig 1 : position + heading --------------------------
    pos = go.Figure()
    add_line(pos, df["t"], df["position_x"], "X (m)")
    add_line(pos, df["t"], df["position_y"], "Y (m)")
    add_line(pos, df["t"], df["height"],      "Height (m)")
    add_line(pos, df["t"], df["heading"],     "Heading", yaxis="y2")
    pos.update_layout(
        title="Position & Heading",
        xaxis_title="time (s)",
        yaxis_title="meters",
        yaxis2=dict(title="heading", overlaying="y", side="right"),
        margin=dict(l=50, r=60, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    # ---------- Fig 2 : velocity, optical flow, yaw rate ------------
    rate = go.Figure()
    add_line(rate, df["t"], df["velocity_x"], "Vx (m/s)")
    add_line(rate, df["t"], df["velocity_y"], "Vy (m/s)")
    add_line(rate, df["t"], df["of_raw_x"],     "OF_raw_x")
    add_line(rate, df["t"], df["of_raw_y"],     "OF_raw_y")
    add_line(rate, df["t"], df["omega_z"],    "Ωx (rad/s)")
    rate.update_layout(
        title="Velocity, optical flow & yaw rate",
        xaxis_title="time (s)",
        yaxis_title="value",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    # ---------- Fig 3 : control outputs -----------------------------
    ctrl = go.Figure()
    add_line(ctrl, df["t"], df["u_x"],   "u_x")
    add_line(ctrl, df["t"], df["u_y"],   "u_y")
    add_line(ctrl, df["t"], df["u_yaw"], "u_yaw")
    ctrl.update_layout(
        title="Control outputs",
        xaxis_title="time (s)",
        yaxis_title="value",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    # ---------- Fig 4 : motor commands ------------------------------
    motors = go.Figure()
    add_line(motors, df["t"], df["m1"], "m1")
    add_line(motors, df["t"], df["m2"], "m2")
    add_line(motors, df["t"], df["m3"], "m3")
    add_line(motors, df["t"], df["m4"], "m4")
    motors.update_layout(
        title="Motor commands",
        xaxis_title="time (s)",
        yaxis_title="PWM",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    warn = f" — missing: {', '.join(missing)}" if missing else ""
    info = f"{Path(file_path).name} — {len(df):,d} rows, {df['t'].iloc[-1]:.1f} s duration{warn}"
    return info, pos, rate, ctrl, motors

# ------------- run --------------------------------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, host="0.0.0.0", port=8050)
