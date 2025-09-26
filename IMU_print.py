#!/usr/bin/env python3
"""
3-panel log viewer — velocity/optical flow, omega, acc

Expected (case/spacing/dots normalized automatically):
t_us,
velocity_x, velocity_y,
of_raw_x, of_raw_y,
omega_x, omega_y, omega_z,      # omega_* optional; any subset ok
acc_x, acc_y, acc_z             # acc_* optional; any subset ok
"""

from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# ---------------- configuration ------------------------------------
LOG_DIR  = Path("./logs")
LOG_GLOB = "log_*.csv"

TITLE = "Velocity / OF, Omega, Acc — Viewer"

# ---------------- helpers ------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace(".", "_").replace(" ", "_").lower() for c in df.columns]
    return df

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame()
    df = normalize_columns(df)

    if "t_us" not in df.columns:
        return pd.DataFrame()

    # columns we care about (added as NaN if missing -> consistent plotting)
    expected = [
        "velocity_x", "velocity_y",
        "of_raw_x", "of_raw_y",
        "omega_x", "omega_y", "omega_z",
        "acc_x", "acc_y", "acc_z",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # numeric coercion & clean
    keep = ["t_us"] + expected
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t_us"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # μs → s, relative to first sample
    df["t"] = (df["t_us"] - df["t_us"].iloc[0]) * 1e-6
    return df

def add_line(fig, x, y, name, **kw):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, **kw))

# ---------------- Dash app -----------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

file_opts = [{"label": p.name, "value": str(p)} for p in sorted(LOG_DIR.glob(LOG_GLOB))]
if not file_opts:
    file_opts = [{"label": "⚠  No logs found", "value": ""}]

app.layout = dbc.Container(
    [
        html.H4(TITLE, className="my-3"),
        dcc.Dropdown(
            id="file-dd",
            options=file_opts,
            value=file_opts[0]["value"],
            clearable=False,
            persistence=True,
            style={"maxWidth": "520px"},
        ),
        html.Small(id="file-info", className="d-block my-2 text-muted"),
        dcc.Graph(id="velof-fig",  style={"height": 350}),
        dcc.Graph(id="omega-fig",  style={"height": 320, "marginTop": "10px"}),
        dcc.Graph(id="acc-fig",    style={"height": 320, "marginTop": "10px"}),
    ],
    fluid=True,
)

# ---------------- callback -----------------------------------------
@app.callback(
    [
        Output("file-info", "children"),
        Output("velof-fig", "figure"),
        Output("omega-fig", "figure"),
        Output("acc-fig",   "figure"),
    ],
    Input("file-dd", "value"),
    prevent_initial_call=False,
)
def update_dash(file_path):
    if not file_path:
        blank = go.Figure().update_layout(title="No data", xaxis_visible=False, yaxis_visible=False)
        return "Select a log file.", blank, blank, blank

    df = load_csv(Path(file_path))
    if df.empty:
        blank = go.Figure().update_layout(title="Empty or malformed log", xaxis_visible=False, yaxis_visible=False)
        return "Empty or malformed log.", blank, blank, blank

    # Detect fully missing to inform user
    expected_non_time = [
        "velocity_x", "velocity_y", "of_raw_x", "of_raw_y",
        "omega_x", "omega_y", "omega_z",
        "acc_x", "acc_y", "acc_z",
    ]
    missing = [c for c in expected_non_time if df[c].isna().all()]

    # ---- Fig 1: velocity + optical flow ----------------------------
    velof = go.Figure()
    if not df["velocity_x"].isna().all(): add_line(velof, df["t"], df["velocity_x"], "Vx (m/s)")
    if not df["velocity_y"].isna().all(): add_line(velof, df["t"], df["velocity_y"], "Vy (m/s)")
    if not df["of_raw_x"].isna().all():   add_line(velof, df["t"], df["of_raw_x"],   "OF_x")
    if not df["of_raw_y"].isna().all():   add_line(velof, df["t"], df["of_raw_y"],   "OF_y")
    if len(velof.data) == 0:              add_line(velof, df["t"], np.nan*df["t"],   "No vel/OF data")
    velof.update_layout(
        title="Velocity & Optical Flow",
        xaxis_title="time (s)",
        yaxis_title="value",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    # ---- Fig 2: omega ----------------------------------------------
    omega = go.Figure()
    if not df["omega_x"].isna().all(): add_line(omega, df["t"], df["omega_x"], "ωx (rad/s)")
    if not df["omega_y"].isna().all(): add_line(omega, df["t"], df["omega_y"], "ωy (rad/s)")
    if not df["omega_z"].isna().all(): add_line(omega, df["t"], df["omega_z"], "ωz (rad/s)")
    if len(omega.data) == 0:           add_line(omega, df["t"], np.nan*df["t"], "No omega data")
    omega.update_layout(
        title="Angular rate (ω)",
        xaxis_title="time (s)",
        yaxis_title="rad/s",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    # ---- Fig 3: accelerations --------------------------------------
    acc = go.Figure()
    if not df["acc_x"].isna().all(): add_line(acc, df["t"], df["acc_x"], "ax (m/s²)")
    if not df["acc_y"].isna().all(): add_line(acc, df["t"], df["acc_y"], "ay (m/s²)")
    if not df["acc_z"].isna().all(): add_line(acc, df["t"], df["acc_z"], "az (m/s²)")
    if len(acc.data) == 0:           add_line(acc, df["t"], np.nan*df["t"], "No acc data")
    acc.update_layout(
        title="Acceleration",
        xaxis_title="time (s)",
        yaxis_title="m/s²",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
    )

    warn = f" — missing: {', '.join(missing)}" if missing else ""
    info = f"{Path(file_path).name} — {len(df):,d} rows, {df['t'].iloc[-1]:.1f} s duration{warn}"
    return info, velof, omega, acc

# ---------------- run -----------------------------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050/ in a browser.")
    app.run(debug=True, host="0.0.0.0", port=8050)
