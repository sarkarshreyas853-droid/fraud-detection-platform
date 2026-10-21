"""Day 12: fraud analyst dashboard.

  streamlit run dashboard/app.py

Reads what the live service scored (data/scored.jsonl) and lets an analyst work the alert queue.
Case decisions are stored in data/cases.db (SQLite) and can be exported as labels for retraining.
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

CASES_DB = ROOT / "data" / "cases.db"
STATUSES = ["New", "Investigating", "Confirmed fraud", "False alarm"]

st.set_page_config(page_title="Fraud Alerts", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load_scored(path: str, mtime: float) -> pd.DataFrame:      # mtime makes the cache refresh when the file changes
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    df = pd.DataFrame(rows)
    df["row"] = range(len(df))
    df["case_id"] = df["tx_id"].fillna(df["row"]).astype("int64").astype(str)
    df["hour"] = df["step"].astype(int)
    return df


def db():
    con = sqlite3.connect(CASES_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY, status TEXT NOT NULL, note TEXT, analyst TEXT, updated_at TEXT)""")
    return con


def load_cases() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql("SELECT * FROM cases", con)


def save_case(case_id, status, note, analyst):
    with db() as con:
        con.execute("""INSERT INTO cases VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(case_id) DO UPDATE SET status=excluded.status, note=excluded.note,
                       analyst=excluded.analyst, updated_at=excluded.updated_at""",
                    (case_id, status, note, analyst, datetime.now(timezone.utc).isoformat(timespec="seconds")))


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- page
st.title("🛡️ Fraud alert dashboard")

if not config.SCORED_LOG.exists():
    st.info("No scored transactions yet. Start the API and run the producer, then refresh this page.")
    st.stop()

df = load_scored(str(config.SCORED_LOG), config.SCORED_LOG.stat().st_mtime)
cases = load_cases()
alerts = df[df["is_alert"]].merge(cases, on="case_id", how="left")
alerts["status"] = alerts["status"].fillna("New")
has_labels = df["label"].notna().any()
card = read_json(config.MODEL_CARD_PATH) or {}

st.caption(f"Model **{card.get('run', '?')}** · alert threshold {card.get('threshold', float('nan')):.3f} · "
           f"hours {df.hour.min()}-{df.hour.max()} of simulated time · source `data/scored.jsonl`")

# KPIs
k = st.columns(5)
k[0].metric("Transactions scored", f"{len(df):,}")
k[1].metric("Alerts", f"{len(alerts):,}", f"{len(alerts) / max(len(df), 1):.2%} of traffic", delta_color="off", delta_arrow="off")
k[2].metric("Open cases", f"{int(alerts.status.isin(['New', 'Investigating']).sum()):,}")
reviewed = alerts[alerts.status.isin(["Confirmed fraud", "False alarm"])]
k[3].metric("Analyst-confirmed precision",
            f"{(reviewed.status == 'Confirmed fraud').mean():.0%}" if len(reviewed) else "–",
            f"{len(reviewed)} reviewed", delta_color="off", delta_arrow="off")
if has_labels:   # replayed history carries true labels; live traffic would not
    tp = int((df.is_alert & (df.label == 1)).sum())
    k[4].metric("Recall (replay labels)", f"{tp / max(int((df.label == 1).sum()), 1):.0%}",
                f"{tp} of {int((df.label == 1).sum())} fraud caught", delta_color="off", delta_arrow="off")

tab_queue, tab_trends, tab_model = st.tabs(["Alert queue", "Trends", "Model health"])

# ---------------------------------------------------------------- queue
with tab_queue:
    c1, c2, c3 = st.columns([2, 2, 1])
    show = c1.multiselect("Status", STATUSES, default=["New", "Investigating"])
    types = c2.multiselect("Type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
    min_amount = c3.number_input("Min amount", min_value=0.0, value=0.0, step=1000.0)
    q = alerts[alerts.status.isin(show) & alerts["type"].isin(types) & (alerts.amount >= min_amount)]
    q = q.sort_values("fraud_score", ascending=False)

    st.write(f"**{len(q):,} alerts** (highest risk first)")
    st.dataframe(q[["case_id", "hour", "type", "amount", "nameDest", "fraud_score", "status", "analyst"]]
                 .rename(columns={"nameDest": "receiver", "fraud_score": "score"}),
                 hide_index=True, width="stretch", height=320,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f"),
                                "amount": st.column_config.NumberColumn("amount", format="%.2f")})

    if len(q):
        st.subheader("Work a case")
        pick = st.selectbox("Case", q["case_id"].tolist(),
                            format_func=lambda c: f"{c} · score {q.set_index('case_id').loc[c, 'fraud_score']:.3f} · "
                                                  f"{q.set_index('case_id').loc[c, 'type']} {q.set_index('case_id').loc[c, 'amount']:,.2f}")
        r = q.set_index("case_id").loc[pick]
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Receiver** `{r['nameDest']}` · **hour** {int(r['hour'])} · **score** {r['fraud_score']:.3f}")
            feats = pd.DataFrame({"value": r["features"]})
            st.dataframe(feats, width="stretch")
            history = df[(df.nameDest == r["nameDest"]) & (df.row < r["row"])]
            st.caption(f"This receiver got {len(history)} earlier payment(s) in this log, "
                       f"{int(history.is_alert.sum())} of them alerted.")
        with right:
            with st.form(f"case-{pick}"):
                status = st.radio("Decision", STATUSES, index=STATUSES.index(r["status"]), horizontal=True)
                note = st.text_area("Note", value=r["note"] if isinstance(r.get("note"), str) else "")
                analyst = st.text_input("Analyst", value=r["analyst"] if isinstance(r.get("analyst"), str) else "")
                if st.form_submit_button("Save decision", type="primary"):
                    save_case(pick, status, note, analyst)
                    st.success("Saved.")
                    st.rerun()
            if has_labels and pd.notna(r["label"]):
                st.caption(f"Replay ground truth (hidden from a real analyst): {'fraud' if r['label'] == 1 else 'normal'}")

    st.divider()
    fb = alerts[alerts.status.isin(["Confirmed fraud", "False alarm"])].copy()
    fb["analyst_label"] = (fb.status == "Confirmed fraud").astype(int)
    st.download_button("Export analyst labels (CSV) for retraining",
                       fb[["case_id", "hour", "type", "amount", "nameDest", "fraud_score", "analyst_label", "note", "analyst", "updated_at"]]
                       .to_csv(index=False), file_name="analyst_labels.csv", disabled=fb.empty)

# ---------------------------------------------------------------- trends
with tab_trends:
    per_hour = df.groupby("hour").agg(transactions=("row", "size"), alerts=("is_alert", "sum"))
    if has_labels:
        per_hour["fraud (replay labels)"] = df.groupby("hour")["label"].sum()
    st.subheader("Alerts per hour")
    st.bar_chart(per_hour.drop(columns="transactions"))
    st.subheader("Traffic per hour")
    st.bar_chart(per_hour["transactions"])
    st.subheader("Score distribution")
    bins = pd.cut(df["fraud_score"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    st.bar_chart(df.groupby(bins, observed=False).size().rename("transactions").set_axis(
        [f"{b.left:.1f}-{b.right:.1f}" for b in bins.cat.categories]))
    st.caption("Most transactions sit near 0; alerts are those at or above the threshold.")

# ---------------------------------------------------------------- model health
with tab_model:
    drift = read_json(ROOT / "reports" / "drift_summary.json")
    if drift:
        st.subheader("Latest drift check")
        st.write(f"Checked {drift['checked_at']} · reference {drift['reference']} · current: {drift['current']}")
        (st.error if drift["retrain"] else st.success)(
            "Retraining recommended: " + "; ".join(drift["reasons"]) if drift["retrain"] else "No retraining needed.")
        psi = pd.DataFrame({"PSI": drift["psi"]}).sort_values("PSI", ascending=False)
        psi["status"] = psi.PSI.map(lambda v: "drift" if v >= drift["psi_limit"] else ("moderate" if v >= 0.1 else "stable"))
        st.dataframe(psi, width="stretch")
        if drift.get("performance"):
            p = drift["performance"]
            st.write(f"PR-AUC {p['pr_auc']:.3f} vs {p['reference_test_pr_auc']:.3f} at training time ({p['ratio']:.0%}).")
        if (ROOT / "reports" / "drift_report.html").exists():
            st.caption("Full visual report: reports/drift_report.html")
    else:
        st.info("No drift check yet. Run: python monitoring/drift_check.py")
    log = ROOT / "reports" / "retrain_log.jsonl"
    if log.exists():
        st.subheader("Retraining attempts")
        rows = [json.loads(x) for x in log.open(encoding="utf-8") if x.strip()]
        st.dataframe(pd.DataFrame([{"at": r["at"], "eval steps": f"{r['eval_steps'][0]}-{r['eval_steps'][1]}",
                                    "champion PR-AUC": round(r["results"]["champion"]["pr_auc"], 3),
                                    "challenger PR-AUC": round(r["results"]["challenger"]["pr_auc"], 3),
                                    "gain": round(r["gain"], 3), "promoted": r["promoted"]} for r in rows]),
                     hide_index=True, width="stretch")
