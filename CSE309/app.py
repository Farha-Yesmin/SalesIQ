# ================================================================
#  SalesIQ — app.py
#  Flask backend  +  scikit-learn Linear Regression ML
# ================================================================
#
#  Folder structure expected:
#
#  CSE309/
#  ├── app.py
#  ├── requirements.txt
#  ├── salesiq.db          ← auto-created on first run
#  └── templates/
#      ├── style.css
#      ├── login.html
#      ├── signup.html
#      ├── upload.html
#      ├── dashboard.html
#      ├── results.html
#      └── report.html
#
#  Run:
#      pip install -r requirements.txt
#      python app.py
#  Then open:  http://127.0.0.1:5000
# ================================================================

# ── Standard library ─────────────────────────────────────────
import csv as pycsv
import datetime
import hashlib
import re
import io
import json
import os
import secrets
import sqlite3
from typing import Any, Optional, Union

# ── Third-party ──────────────────────────────────────────────
import numpy as np
import pandas as pd
from flask import (Flask, Response, jsonify, redirect,
                   render_template, request, send_file,
                   send_from_directory, session)
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

# ================================================================
#  App & config
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMPL_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "salesiq.db")

app = Flask(__name__, template_folder=TMPL_DIR,
            static_folder=TMPL_DIR, static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
# Allow credentialed API calls from Live Server / other local ports and file:// (Origin: null).
CORS(
    app,
    supports_credentials=True,
    origins=[
        re.compile(r"^https?://127\.0\.0\.1(?::\d+)?$"),
        re.compile(r"^https?://localhost(?::\d+)?$"),
        re.compile(r"^null$", re.IGNORECASE),
    ],
)

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ================================================================
#  Database
# ================================================================


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT    NOT NULL,
            last_name  TEXT    NOT NULL,
            email      TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL REFERENCES users(id),
            run_date          TEXT    NOT NULL,
            months_used       TEXT    NOT NULL,
            months_count      INTEGER NOT NULL,
            predicted_month   TEXT    NOT NULL,
            product_name      TEXT    NOT NULL,
            predicted_units   INTEGER NOT NULL,
            predicted_revenue REAL    NOT NULL,
            demand_level      TEXT    NOT NULL,
            trend             TEXT    NOT NULL,
            recommendation    TEXT    NOT NULL,
            price             REAL    NOT NULL,
            historical_qty_json TEXT
        );
    """)
    _migrate_prediction_results(conn)
    conn.commit()
    conn.close()


def _migrate_prediction_results(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(prediction_results)").fetchall()}
    if "historical_qty_json" not in cols:
        conn.execute(
            "ALTER TABLE prediction_results ADD COLUMN historical_qty_json TEXT"
        )


init_db()

# ================================================================
#  Auth helpers
# ================================================================


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def now_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def get_uid() -> Optional[int]:
    return session.get("user_id")


def login_required() -> Union[int, tuple[Response, int]]:
    uid = get_uid()
    if uid is None:
        return jsonify({"error": "Not logged in"}), 401
    return uid

# ================================================================
#  Serve HTML pages  (Flask renders from templates/ folder)
# ================================================================


@app.route("/")
def index() -> Response:
    return redirect("/login")


@app.route("/login")
def login_page() -> str:
    return render_template("login.html")


@app.route("/signup")
def signup_page() -> str:
    return render_template("signup.html")


@app.route("/upload")
def upload_page() -> str:
    if not get_uid():
        return redirect("/login")
    return render_template("upload.html")


@app.route("/dashboard")
def dashboard_page() -> str:
    if not get_uid():
        return redirect("/login")
    return render_template("dashboard.html")


@app.route("/results")
def results_page() -> str:
    if not get_uid():
        return redirect("/login")
    return render_template("results.html")


@app.route("/report")
def report_page() -> str:
    if not get_uid():
        return redirect("/login")
    return render_template("report.html")

# ================================================================
#  Auth API routes
# ================================================================


@app.route("/api/signup", methods=["POST"])
def signup() -> tuple[Response, int]:
    body: dict[str, Any] = request.get_json(force=True) or {}
    first = str(body.get("first_name", "")).strip()
    last = str(body.get("last_name",  "")).strip()
    email = str(body.get("email",      "")).strip().lower()
    pw = str(body.get("password",   ""))

    if not all([first, last, email, pw]):
        return jsonify({"error": "All fields are required"}), 400
    if len(pw) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (first_name,last_name,email,password,created_at) VALUES (?,?,?,?,?)",
            (first, last, email, hash_pw(pw), now_str()),
        )
        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email,)).fetchone()
        session["user_id"] = user["id"]
        session["user_name"] = f"{user['first_name']} {user['last_name']}"
        session["user_email"] = user["email"]
        return jsonify({"success": True, "name": session["user_name"]}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409
    finally:
        conn.close()


@app.route("/api/login", methods=["POST"])
def login() -> tuple[Response, int]:
    body: dict[str, Any] = request.get_json(force=True) or {}
    email = str(body.get("email",    "")).strip().lower()
    pw = str(body.get("password", ""))

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, hash_pw(pw)),
    ).fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    session["user_id"] = user["id"]
    session["user_name"] = f"{user['first_name']} {user['last_name']}"
    session["user_email"] = user["email"]
    return jsonify({"success": True, "name": session["user_name"]}), 200


@app.route("/api/logout", methods=["POST"])
def logout() -> tuple[Response, int]:
    session.clear()
    return jsonify({"success": True}), 200


@app.route("/api/me")
def me() -> tuple[Response, int]:
    uid = get_uid()
    if uid is None:
        return jsonify({"logged_in": False}), 200
    return jsonify({
        "logged_in": True,
        "name":      session.get("user_name"),
        "email":     session.get("user_email"),
    }), 200

# ================================================================
#  ML — Data helpers
# ================================================================


def parse_csv_file(data: bytes) -> pd.DataFrame:
    """
    Parse raw CSV bytes into a cleaned DataFrame.
    Keeps only valid rows with Product, Quantity Ordered, Price Each.
    """
    text = data.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text))

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows where Product is literally 'Product' (repeated headers)
    if "Product" in df.columns:
        df = df[df["Product"] != "Product"]

    # Drop rows missing key columns
    required = ["Product", "Quantity Ordered", "Price Each"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df.dropna(subset=required)
    df["Quantity Ordered"] = pd.to_numeric(
        df["Quantity Ordered"], errors="coerce")
    df["Price Each"] = pd.to_numeric(df["Price Each"],       errors="coerce")
    df = df.dropna(subset=["Quantity Ordered", "Price Each"])
    df = df[df["Quantity Ordered"] > 0]
    df = df[df["Price Each"] > 0]

    return df


def month_from_filename(filename: str) -> tuple[Optional[str], Optional[int]]:
    lower = filename.lower()
    for idx, name in enumerate(MONTH_NAMES):
        if name.lower() in lower:
            return name, idx
    return None, None


# ================================================================
#  ML — scikit-learn Linear Regression
# ================================================================


def sklearn_predict(monthly_values: list[int]) -> int:
    """
    Fit LinearRegression on (month_index, qty_sold) pairs.
    Predict the value at month index = n  (next month).

    X shape: (n_months, 1)  — month index 0,1,2,...
    y shape: (n_months,)    — units sold each month
    """
    n = len(monthly_values)

    if n == 1:
        # Only one month — assume 5 % growth, can't fit a line
        return max(0, round(monthly_values[0] * 1.05))

    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.array(monthly_values, dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    # Predict the NEXT time step
    predicted = model.predict(np.array([[float(n)]]))[0]
    return max(0, round(float(predicted)))


def detect_trend(monthly_values: list[int]) -> str:
    """
    Compare second-half average vs first-half average.
    > +5 %  → up
    < -5 %  → down
    else    → stable
    """
    if len(monthly_values) < 2:
        return "stable"

    mid = max(1, len(monthly_values) // 2)
    first_avg = float(np.mean(monthly_values[:mid]))
    second_avg = float(np.mean(monthly_values[mid:]))

    if first_avg == 0:
        return "up" if second_avg > 0 else "stable"

    change = (second_avg - first_avg) / first_avg * 100
    if change > 5:
        return "up"
    if change < -5:
        return "down"
    return "stable"


def classify_demand(predicted: int, all_predicted: list[int]) -> str:
    """
    Top third  → High
    Bottom third → Low
    Middle      → Medium
    """
    if not all_predicted:
        return "Medium"
    sorted_desc = sorted(all_predicted, reverse=True)
    n = len(sorted_desc)
    third = max(1, n // 3)
    if predicted >= sorted_desc[third - 1]:
        return "High"
    if predicted <= sorted_desc[n - third]:
        return "Low"
    return "Medium"


def get_recommendation(trend: str, demand: str, qty: int, max_qty: int) -> str:
    pct = round((qty / max_qty) * 30) if max_qty > 0 else 0
    if trend == "up" and demand == "High":
        return f"Increase stock by {max(15, pct)}%"
    if trend == "up" and demand == "Medium":
        return f"Increase stock by {max(8, pct // 2)}%"
    if trend == "up" and demand == "Low":
        return "Maintain current stock"
    if trend == "down" and demand in ("Low", "Medium"):
        return "Reduce reorder quantity"
    if trend == "down" and demand == "High":
        return "Maintain current stock"
    return "Maintain current stock"

# ================================================================
#  ML — Main pipeline
# ================================================================


def run_ml_pipeline(files_data: list[tuple[str, bytes]]) -> dict[str, Any]:
    """
    1. Parse each CSV with pandas.
    2. Aggregate total qty + price per product per month.
    3. For each product: fit sklearn LinearRegression → predict next month.
    4. Classify demand, detect trend, generate recommendation.
    5. Return a result dict the frontend reads from sessionStorage.
    """

    # ── Step 1: Parse & aggregate ────────────────────────
    month_data: list[dict[str, Any]] = []

    for filename, content in files_data:
        df = parse_csv_file(content)
        if df.empty:
            continue

        label, idx = month_from_filename(filename)
        if label is None or idx is None:
            label = f"Month {len(month_data) + 1}"
            idx = len(month_data)

        # Aggregate: group by product → sum qty, mean price
        grp = (
            df.groupby("Product")
            .agg(qty=("Quantity Ordered", "sum"), price=("Price Each", "mean"))
            .reset_index()
        )
        totals: dict[str, dict[str, float]] = {}
        for _, row in grp.iterrows():
            totals[str(row["Product"])] = {
                "qty":     float(row["qty"]),
                "price":   float(row["price"]),
                "revenue": float(row["qty"]) * float(row["price"]),
            }

        month_data.append({"label": label, "index": idx, "totals": totals})

    if not month_data:
        raise ValueError("No valid CSV data found in uploaded files.")

    # Sort chronologically
    month_data.sort(key=lambda m: m["index"])

    # ── Step 2: Collect all unique products ──────────────
    all_products: dict[str, float] = {}
    for month in month_data:
        for prod, info in month["totals"].items():
            if prod not in all_products:
                all_products[prod] = float(info["price"])

    # ── Step 3: Per-product sklearn LinearRegression ─────
    product_results: list[dict[str, Any]] = []

    for product, price in all_products.items():
        hist_qty: list[int] = [
            int(month["totals"].get(product, {}).get("qty", 0))
            for month in month_data
        ]

        pred_qty = sklearn_predict(hist_qty)
        pred_rev = round(pred_qty * price, 2)
        trend = detect_trend(hist_qty)

        product_results.append({
            "product":       product,
            "historicalQty": hist_qty,
            "predictedQty":  pred_qty,
            "predictedRev":  pred_rev,
            "price":         price,
            "trend":         trend,
        })

    # ── Step 4: Demand classification ────────────────────
    all_pred_qty = [r["predictedQty"] for r in product_results]
    for result in product_results:
        result["demand"] = classify_demand(
            result["predictedQty"], all_pred_qty)

    # Sort by predicted qty desc
    product_results.sort(key=lambda r: r["predictedQty"], reverse=True)

    # ── Step 5: Recommendations ──────────────────────────
    max_qty = max((r["predictedQty"] for r in product_results), default=1)
    for result in product_results:
        result["recommendation"] = get_recommendation(
            result["trend"], result["demand"],
            result["predictedQty"], max_qty,
        )

    # ── Step 6: Summary ──────────────────────────────────
    last_idx = month_data[-1]["index"]
    predicted_month = MONTH_NAMES[(last_idx + 1) % 12]
    total_predicted = sum(r["predictedQty"] for r in product_results)
    top = product_results[0] if product_results else {}
    rising_count = sum(1 for r in product_results if r["trend"] == "up")

    return {
        "months":         [m["label"] for m in month_data],
        "predictedMonth": predicted_month,
        "totalPredicted": total_predicted,
        "topProduct":     top.get("product", "—"),
        "topProductQty":  top.get("predictedQty", 0),
        "risingCount":    rising_count,
        "totalProducts":  len(product_results),
        "products":       product_results,
    }

# ================================================================
#  Predict API route
# ================================================================


@app.route("/api/predict", methods=["POST"])
def predict() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400
    if len(uploaded) > 12:
        return jsonify({"error": "Maximum 12 files allowed"}), 400

    files_data: list[tuple[str, bytes]] = [
        (f.filename or f"file_{i}.csv", f.read())
        for i, f in enumerate(uploaded)
    ]

    try:
        result = run_ml_pipeline(files_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"ML error: {str(e)}"}), 500

    # ── Save every product row to DB ─────────────────────
    run_date = now_str()
    months_json = json.dumps(result["months"])
    months_count = len(result["months"])
    conn = get_db()

    for p in result["products"]:
        conn.execute(
            """INSERT INTO prediction_results
               (user_id, run_date, months_used, months_count, predicted_month,
                product_name, predicted_units, predicted_revenue,
                demand_level, trend, recommendation, price, historical_qty_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid, run_date, months_json, months_count,
                result["predictedMonth"],
                p["product"],
                p["predictedQty"],
                p["predictedRev"],
                p["demand"],
                p["trend"],
                p["recommendation"],
                p["price"],
                json.dumps(p.get("historicalQty", [])),
            ),
        )

    conn.commit()
    conn.close()

    return jsonify({
        "success":     True,
        "result":      result,
        "run_date":    run_date,
        "user_email":  session.get("user_email", ""),
    }), 200

# ================================================================
#  Results API routes
# ================================================================


def _row_historical_qty(row: sqlite3.Row) -> list[int]:
    try:
        raw = row["historical_qty_json"]
    except (KeyError, IndexError):
        return []
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [int(x) for x in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def rows_to_products(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [
        {
            "product":        row["product_name"],

            "predictedQty":   row["predicted_units"],
            "predictedRev":   row["predicted_revenue"],
            "demand":         row["demand_level"],
            "trend":          row["trend"],
            "recommendation": row["recommendation"],
            "price":          row["price"],
            "historicalQty":  _row_historical_qty(row),
        }
        for row in rows
    ]


@app.route("/api/results/latest")
def results_latest() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    conn = get_db()
    meta = conn.execute(
        """SELECT run_date, months_used, months_count, predicted_month
           FROM prediction_results WHERE user_id=?
           ORDER BY id DESC LIMIT 1""",
        (uid,),
    ).fetchone()

    if meta is None:
        conn.close()
        return jsonify({"has_data": False}), 200

    rows = conn.execute(
        """SELECT product_name, predicted_units, predicted_revenue,
                  demand_level, trend, recommendation, price, historical_qty_json
           FROM prediction_results
           WHERE user_id=? AND run_date=?
           ORDER BY predicted_units DESC""",
        (uid, meta["run_date"]),
    ).fetchall()
    conn.close()

    months = json.loads(meta["months_used"])
    products = rows_to_products(rows)

    return jsonify({
        "has_data":        True,
        "run_date":        meta["run_date"],
        "predictedMonth":  meta["predicted_month"],
        "months":          months,
        "totalPredicted":  sum(p["predictedQty"] for p in products),
        "totalProducts":   len(products),
        "topProduct":      products[0]["product"] if products else "—",
        "topProductQty":   products[0]["predictedQty"] if products else 0,
        "risingCount":     sum(1 for p in products if p["trend"] == "up"),
        "products":        products,
    }), 200


@app.route("/api/results/history")
def results_history() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    conn = get_db()
    rows = conn.execute(
        """SELECT run_date, months_used, months_count, predicted_month
           FROM prediction_results WHERE user_id=?
           GROUP BY run_date
           ORDER BY MAX(id) DESC""",
        (uid,),
    ).fetchall()
    conn.close()

    total = len(rows)
    runs: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        months = json.loads(row["months_used"])
        runs.append({
            "run_number":      total - i,
            "run_date":        row["run_date"],
            "predicted_month": row["predicted_month"],
            "months_count":    row["months_count"],
            "months_used":     months,
            "dataset_str":     (
                months[0] if len(months) == 1
                else f"{months[0]} – {months[-1]} ({len(months)} months)"
            ),
        })
    return jsonify({"runs": runs}), 200


@app.route("/api/results/by-date")
def results_by_date() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    run_date = request.args.get("run_date", "")
    if not run_date:
        return jsonify({"error": "run_date param required"}), 400

    conn = get_db()
    meta = conn.execute(
        """SELECT run_date, months_used, predicted_month
           FROM prediction_results WHERE user_id=? AND run_date=? LIMIT 1""",
        (uid, run_date),
    ).fetchone()

    if meta is None:
        conn.close()
        return jsonify({"error": "Run not found"}), 404

    rows = conn.execute(
        """SELECT product_name, predicted_units, predicted_revenue,
                  demand_level, trend, recommendation, price, historical_qty_json
           FROM prediction_results
           WHERE user_id=? AND run_date=?
           ORDER BY predicted_units DESC""",
        (uid, run_date),
    ).fetchall()
    conn.close()

    return jsonify({
        "run_date":        meta["run_date"],
        "predictedMonth":  meta["predicted_month"],
        "months":          json.loads(meta["months_used"]),
        "products":        rows_to_products(rows),
    }), 200


def _delete_prediction_run(uid: int, run_date: str) -> tuple[Response, int]:
    if not run_date:
        return jsonify({"error": "run_date required"}), 400
    conn = get_db()
    cur = conn.execute(
        "DELETE FROM prediction_results WHERE user_id=? AND run_date=?",
        (uid, run_date),
    )
    n = cur.rowcount
    conn.commit()
    conn.close()
    if n == 0:
        return jsonify({"error": "Run not found"}), 404
    return jsonify({"success": True, "deleted_rows": n}), 200


@app.route("/api/results/run", methods=["DELETE"])
def results_delete_run() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    body: dict[str, Any] = request.get_json(silent=True) or {}
    run_date = str(body.get("run_date", "")).strip()
    return _delete_prediction_run(auth, run_date)


@app.route("/api/results/delete", methods=["POST"])
def results_delete_post() -> tuple[Response, int]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    body: dict[str, Any] = request.get_json(silent=True) or {}
    run_date = str(body.get("run_date", "")).strip()
    return _delete_prediction_run(auth, run_date)

# ================================================================
#  Report download routes
# ================================================================


def fetch_report_rows(uid: int, run_date: Optional[str]) -> list[sqlite3.Row]:
    conn = get_db()
    query = (
        "SELECT product_name, predicted_units, predicted_revenue,"
        " demand_level, trend, recommendation, predicted_month"
        " FROM prediction_results WHERE user_id=?"
    )
    params: list[Any] = [uid]
    if run_date:
        query += " AND run_date=?"
        params.append(run_date)
    query += " ORDER BY predicted_units DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


@app.route("/api/report/csv")
def report_csv() -> Union[Response, tuple[Response, int]]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    rows = fetch_report_rows(uid, request.args.get("run_date"))
    if not rows:
        return jsonify({"error": "No data found"}), 404

    buf = io.StringIO()
    writer = pycsv.writer(buf)
    writer.writerow([
        "Product Name", "Predicted Units",
        "Predicted Revenue", "Demand Level", "Trend", "Recommendation",
    ])
    for row in rows:
        writer.writerow([
            row["product_name"],
            row["predicted_units"], f"${row['predicted_revenue']:.2f}",
            row["demand_level"], row["trend"], row["recommendation"],
        ])

    buf.seek(0)
    month_slug = rows[0]["predicted_month"].lower().replace(" ", "_")
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"salesiq_report_{month_slug}.csv",
    )


@app.route("/api/report/pdf")
def report_pdf() -> Union[Response, tuple[Response, int]]:
    auth = login_required()
    if isinstance(auth, tuple):
        return auth
    uid: int = auth

    try:
        from reportlab.lib import colors                             # type: ignore
        from reportlab.lib.pagesizes import A4                      # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet        # type: ignore
        from reportlab.platypus import (                            # type: ignore
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        return jsonify({"error": "reportlab not installed"}), 500

    rows = fetch_report_rows(uid, request.args.get("run_date"))
    if not rows:
        return jsonify({"error": "No data found"}), 404

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    month = rows[0]["predicted_month"]
    elems.append(
        Paragraph(f"SalesIQ — Prediction Report ({month})", styles["Title"]))
    elems.append(Paragraph(f"Generated: {now_str()}", styles["Normal"]))
    elems.append(Spacer(1, 16))

    tdata = [["#", "Product", "Pred. Units",
              "Pred. Revenue", "Demand", "Trend", "Recommendation"]]
    for i, row in enumerate(rows, 1):
        tdata.append([
            str(i), row["product_name"],
            str(row["predicted_units"]), f"${row['predicted_revenue']:.2f}",
            row["demand_level"], row["trend"], row["recommendation"],
        ])

    tbl = Table(tdata, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4b4")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",  (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elems.append(tbl)
    doc.build(elems)
    buf.seek(0)

    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"salesiq_report_{month.lower()}.pdf")

# ================================================================
#  Run
# ================================================================


if __name__ == "__main__":
    print("\n  SalesIQ — open in your browser: http://127.0.0.1:5000/login\n")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
