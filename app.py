"""Dynamic Solvency & Risk Simulator — Streamlit app.

Βασίζεται ΑΠΟΚΛΕΙΣΤΙΚΑ στο Dynamic Solvency_Risk Simulator.ipynb.
Κάθε γράφημα, κάθε νούμερο και κάθε συμπέρασμα προέρχεται από το notebook.

Τρέξε:  streamlit run app.py
"""

import logging
import time

import matplotlib
matplotlib.use("Agg")            # headless: ο server δεν έχει GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Dynamic Solvency & Risk Simulator",
                   page_icon="📊", layout="wide")

Z_THRESHOLD = 1.10        # Is_Distressed = Z_score <= 1.10  (cell 21)
RUIN_THRESHOLD = 0.35     # custom_threshold του cell 38/41


# =========================================================================
# ΦΑΣΗ 1 — Data Engineering (cells 3-6)
# =========================================================================

SEC_HEADERS = {
    "User-Agent": "eliaslaspatzakes@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

CONCEPTS = {
    "Total_Revenue": ["Revenue", "Revenues",
                      "RevenueFromContractWithCustomerExcludingAssessedTax",
                      "SalesRevenueNet"],
    "Cost_Of_Revenue": ["CostOfRevenue"],
    "Operating_Expenses": ["OperatingExpenses"],
    "EBIT": ["OperatingIncomeLoss"],
    "Total_Assets": ["Assets"],
    "Total_Liabilities": ["Liabilities"],
    "Current_Assets": ["AssetsCurrent"],
    "Current_Liabilities": ["LiabilitiesCurrent"],
    "Retained_Earnings": ["RetainedEarningsAccumulatedDeficit"],
    "Equity": ["StockholdersEquity"],
    "Gross_Profit": ["GrossProfit"],
}

PORTFOLIO = ["URBN", "SHAK", "GPRO", "FSLY", "TNC",
             "ROKU", "KSS", "PLAY", "TDOC", "TEX", "YETI"]

INDUSTRY_MAPPING = {
    "URBN": "Consumer Cyclical", "KSS": "Consumer Cyclical",
    "YETI": "Consumer Cyclical", "SHAK": "Consumer Cyclical",
    "PLAY": "Consumer Cyclical", "FSLY": "Technology",
    "ROKU": "Technology", "GPRO": "Technology",
    "TNC": "Industrials", "TEX": "Industrials",
    "OMI": "Healthcare", "TDOC": "Healthcare",
}


def get_sec_ticker_map() -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    raw = response.json()
    return {item["ticker"].upper(): str(item["cik_str"]).zfill(10)
            for item in raw.values()}


def extract_usd_facts(companyfacts: dict, candidates: list[str]) -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    for concept in candidates:
        concept_data = facts.get(concept)
        if not concept_data:
            continue
        rows = []
        for item in concept_data.get("units", {}).get("USD", []):
            if item.get("form") != "10-K" or item.get("fp") != "FY":
                continue
            rows.append({
                "Fiscal_Year_Num": item.get("fy"),
                "Fiscal_Year": pd.to_datetime(item.get("end"), errors="coerce"),
                "Filed": pd.to_datetime(item.get("filed"), errors="coerce"),
                "value": item.get("val"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df.dropna(subset=["Fiscal_Year", "Fiscal_Year_Num"])
        df = df.sort_values(["Fiscal_Year_Num", "Filed"])
        df = df.drop_duplicates(subset=["Fiscal_Year_Num"], keep="last")
        return df[["Fiscal_Year_Num", "Fiscal_Year", "value"]]
    return pd.DataFrame(columns=["Fiscal_Year_Num", "Fiscal_Year", "value"])


def fetch_3yr_financial_data_sec(tickers: list[str]) -> pd.DataFrame:
    ticker_map = get_sec_ticker_map()
    output_rows = []
    progress = st.progress(0.0, text="Άντληση SEC financials...")

    for i, ticker in enumerate(tickers):
        ticker = ticker.upper()
        progress.progress((i + 1) / len(tickers), text=f"SEC: {ticker}")
        cik = ticker_map.get(ticker)
        if cik is None:
            logger.warning("Δεν βρέθηκε CIK για %s", ticker)
            continue

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            response = requests.get(url, headers=SEC_HEADERS, timeout=30)
            response.raise_for_status()
            companyfacts = response.json()
        except requests.RequestException as exc:
            logger.warning("Αποτυχία SEC request για %s: %s", ticker, exc)
            continue

        company_df = None
        for output_name, candidates in CONCEPTS.items():
            metric_df = extract_usd_facts(companyfacts, candidates)
            metric_df = metric_df.rename(columns={"value": output_name})
            if metric_df.empty:
                continue
            if company_df is None:
                company_df = metric_df
            else:
                company_df = company_df.merge(
                    metric_df, on=["Fiscal_Year_Num", "Fiscal_Year"], how="outer")

        if company_df is None or company_df.empty:
            logger.warning("Δεν βρέθηκαν 10-K facts για %s", ticker)
            continue

        company_df["Ticker"] = ticker
        output_rows.append(company_df.sort_values("Fiscal_Year").tail(3))
        time.sleep(0.2)                       # SEC rate limit

    progress.empty()
    if not output_rows:
        raise ValueError("Δεν αντλήθηκαν SEC financials για κανένα ticker.")

    df = pd.concat(output_rows, ignore_index=True)
    df["Fiscal_Year"] = pd.to_datetime(df["Fiscal_Year"]).dt.normalize()
    return df.sort_values(["Ticker", "Fiscal_Year"]).reset_index(drop=True)


def fetch_fred_series(series_id: str, column_name: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, na_values=".")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={series_id: column_name})
    return df.set_index("observation_date")[[column_name]]


def fetch_macro_data_fred() -> pd.DataFrame:
    macro_df = pd.concat([
        fetch_fred_series("SP500", "SP500"),
        fetch_fred_series("DGS10", "Interest_Rate_10Yr"),
        fetch_fred_series("VIXCLS", "VIX"),
    ], axis=1)
    return macro_df.sort_index().ffill().dropna()


def clean_data(df_final: pd.DataFrame) -> pd.DataFrame:
    """Τα per-ticker fills του cell 6. Η σειρά μετράει — πιστή αντιγραφή."""
    df_final = df_final.drop(columns=["Fiscal_Year"])
    df_final = df_final.astype({"Fiscal_Year_Num": "string"})
    df_final["sector"] = df_final["Ticker"].map(INDUSTRY_MAPPING)

    mask_cons = df_final["sector"] == "Consumer Cyclical"
    mean_cons = round(df_final.loc[mask_cons, "Operating_Expenses"].mean(), 1)
    df_final.loc[mask_cons, "Operating_Expenses"] = \
        df_final.loc[mask_cons, "Operating_Expenses"].fillna(mean_cons)

    # Healthcare -> έξω (κόβει την TDOC)
    df_final = df_final.loc[df_final["sector"] != "Healthcare"].reset_index(drop=True)

    m = df_final["Ticker"] == "FSLY"
    df_final.loc[m, "Total_Revenue"] = round(df_final.loc[m, "Total_Revenue"].fillna(
        df_final["Cost_Of_Revenue"] + df_final["Gross_Profit"]), 1)

    m = df_final["Ticker"] == "PLAY"
    df_final.loc[m, "Total_Revenue"] = round(df_final.loc[m, "Total_Revenue"].fillna(
        df_final["EBIT"] + df_final["Cost_Of_Revenue"] + df_final["Operating_Expenses"]), 1)
    df_final.loc[m, "Gross_Profit"] = round(df_final.loc[m, "Gross_Profit"].fillna(
        df_final["Total_Revenue"] - df_final["Cost_Of_Revenue"]), 1)
    df_final.loc[m, "Total_Liabilities"] = round(
        df_final.loc[m, "Total_Liabilities"]).fillna(
        df_final["Total_Assets"] - df_final["Equity"])

    m = df_final["Ticker"] == "KSS"
    df_final.loc[m, "Cost_Of_Revenue"] = round(df_final.loc[m, "Cost_Of_Revenue"].fillna(
        df_final["Total_Revenue"] - df_final["EBIT"] - df_final["Operating_Expenses"]), 1)
    df_final.loc[m, "Gross_Profit"] = round(df_final.loc[m, "Gross_Profit"].fillna(
        df_final["Total_Revenue"] - df_final["Cost_Of_Revenue"]), 1)
    df_final.loc[m, "Total_Liabilities"] = round(
        df_final.loc[m, "Total_Liabilities"]).fillna(
        df_final["Total_Assets"] - df_final["Equity"])

    m = df_final["Ticker"] == "ROKU"
    df_final.loc[m, "Cost_Of_Revenue"] = round(df_final.loc[m, "Cost_Of_Revenue"].fillna(
        df_final["Total_Revenue"] - df_final["EBIT"] - df_final["Operating_Expenses"]), 1)

    m = df_final["Ticker"] == "SHAK"
    df_final.loc[m, "Total_Revenue"] = round(df_final.loc[m, "Total_Revenue"].fillna(
        df_final["EBIT"] + df_final["Cost_Of_Revenue"] + df_final["Operating_Expenses"]), 1)
    df_final.loc[m, "Cost_Of_Revenue"] = round(df_final.loc[m, "Cost_Of_Revenue"].fillna(
        df_final["Total_Revenue"] - df_final["EBIT"] - df_final["Operating_Expenses"]), 1)
    df_final.loc[m, "Gross_Profit"] = round(df_final.loc[m, "Gross_Profit"].fillna(
        df_final["Total_Revenue"] - df_final["Cost_Of_Revenue"]), 1)

    m = df_final["Ticker"] == "TEX"
    df_final.loc[m, "Operating_Expenses"] = round(df_final.loc[m, "Operating_Expenses"].fillna(
        df_final["Total_Revenue"] - df_final["EBIT"] - df_final["Cost_Of_Revenue"]), 1)
    df_final.loc[m, "Equity"] = round(df_final.loc[m, "Equity"]).fillna(
        df_final["Total_Assets"] - df_final["Total_Liabilities"])

    # TNC -> έξω
    df_final = df_final.loc[df_final["Ticker"] != "TNC"].reset_index(drop=True)
    return df_final


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Break-even, margins, Z-score, target — cells 8-10, 21."""
    # Break-Even variables (cell 8)
    df["Contribution_Margin"] = (df["Total_Revenue"] - df["Cost_Of_Revenue"]) / df["Total_Revenue"]
    df["BEP_Rev"] = df["Operating_Expenses"] / df["Contribution_Margin"]
    df["Margin of Safety%"] = round(
        ((df["Total_Revenue"] - df["BEP_Rev"]) / df["Total_Revenue"]) * 100, 1)

    # Margin analysis (cell 9)
    df["Operating_Margin%"] = round((df["EBIT"] / df["Total_Revenue"]) * 100, 1)
    df["YoY Revenue Growth"] = df.groupby("Ticker")["Total_Revenue"].pct_change()
    df["YoY Revenue Growth"] = df["YoY Revenue Growth"].fillna(0)
    df["Debt_to_Equity"] = df["Total_Liabilities"] / df["Equity"]

    # Altman Z-score (cell 10). Το round(,1) ανά συστατικό είναι του notebook.
    df["Working_Capital"] = df["Current_Assets"] - df["Current_Liabilities"]
    X1 = round(df["Working_Capital"] / df["Total_Assets"], 1)
    X2 = round(df["Retained_Earnings"] / df["Total_Assets"], 1)
    X3 = round(df["EBIT"] / df["Total_Assets"], 1)
    X4 = round(df["Equity"] / df["Total_Liabilities"], 1)
    X5 = round(df["Total_Revenue"] / df["Total_Assets"], 1)
    df["Z_score"] = round(1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5, 1)

    # Target (cell 21)
    df["Is_Distressed"] = np.where(df["Z_score"] > Z_THRESHOLD, 0, 1)
    return df


@st.cache_data(ttl=3600, show_spinner="Άντληση SEC & FRED δεδομένων...")
def load_data() -> pd.DataFrame:
    """SEC + FRED -> cleaning -> features. Cache 1 ώρα ώστε να μη χτυπάμε το SEC."""
    company_df = fetch_3yr_financial_data_sec(PORTFOLIO)
    macro_df = fetch_macro_data_fred()

    df_final = pd.merge_asof(
        company_df.sort_values("Fiscal_Year"), macro_df.sort_index(),
        left_on="Fiscal_Year", right_index=True, direction="backward")
    df_final = df_final.sort_values(["Ticker", "Fiscal_Year"]).reset_index(drop=True)

    return add_features(clean_data(df_final))


# =========================================================================
# ΦΑΣΗ 2 — Machine Learning (cells 24-39)
# =========================================================================

@st.cache_resource(show_spinner="Εκπαίδευση μοντέλου...")
def train_model(df_final: pd.DataFrame) -> dict:
    """Feature selection -> CV -> fit -> SHAP. Cells 24-39."""
    df_ml = df_final[[
        "Total_Revenue", "Cost_Of_Revenue", "EBIT", "Total_Assets",
        "Total_Liabilities", "Current_Assets", "Current_Liabilities",
        "Retained_Earnings", "Equity", "Gross_Profit", "Operating_Expenses",
        "SP500", "Interest_Rate_10Yr", "VIX", "sector", "Contribution_Margin",
        "BEP_Rev", "Margin of Safety%", "Operating_Margin%",
        "YoY Revenue Growth", "Working_Capital", "Is_Distressed",
    ]]

    y = df_ml["Is_Distressed"]
    X = df_ml.drop(["Is_Distressed"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # f_classif ΜΟΝΟ στο train (cell 29)
    num_f = X_train.select_dtypes(include="float64").columns.to_list()
    _, p_value = f_classif(X_train[num_f], y_train)
    selected = pd.Series(p_value, index=num_f)[lambda s: s < 0.05].index

    X_train_final = X_train[selected].copy()
    X_test_final = X_test[selected].copy()

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("scaler", StandardScaler())]),
                       X_train_final.columns.to_list())],
        remainder="drop")

    # Model selection — CV των 3 μοντέλων (cells 34-35)
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
            class_weight="balanced"),
        "XGBClassifier": XGBClassifier(
            n_estimators=500, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, max_depth=5, random_state=42, n_jobs=-1,
            tree_method="hist", objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=neg / max(pos, 1)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    cv_rows = []
    for name, model in models.items():
        res = cross_validate(
            Pipeline([("preprocessor", preprocessor), ("classifier", model)]),
            X_train_final, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows.append({"model": name,
                        **{s: float(res[f"test_{s}"].mean()) for s in scoring}})

    # Το τελικό μοντέλο (cell 36)
    model_log = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000,
                                          class_weight="balanced")),
    ]).fit(X_train_final, y_train)

    y_probs = model_log.predict_proba(X_test_final)[:, 1]
    y_pred = model_log.predict(X_test_final)
    y_pred_custom = (y_probs >= RUIN_THRESHOLD).astype(int)   # cell 38

    # SHAP (cell 39)
    X_train_scaled = preprocessor.transform(X_train_final)
    X_test_scaled = preprocessor.transform(X_test_final)
    explainer = shap.LinearExplainer(model_log.named_steps["classifier"],
                                     X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_values.feature_names = X_train_final.columns.tolist()

    return {
        "model": model_log, "df_ml": df_ml,
        "X_train_final": X_train_final, "X_test_final": X_test_final,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "y_probs": y_probs, "y_pred": y_pred, "y_pred_custom": y_pred_custom,
        "shap_values": shap_values,
        "features": X_train_final.columns.tolist(),
        "cv_summary": pd.DataFrame(cv_rows),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }


# =========================================================================
# ΦΑΣΗ 3 — Monte Carlo (cell 41)
# =========================================================================

def run_monte_carlo(base, model, feat_cols, sigma_rev=0.15, sigma_cogs=0.10,
                    threshold=RUIN_THRESHOLD, N=10_000, seed=42) -> dict:
    """Cell 41, παραμετροποιημένο.

    Σοκάρει τους οδηγούς (revenue, cogs, current assets, equity) και
    ξαναϋπολογίζει τα παράγωγα από τις λογιστικές ταυτότητες, ώστε κάθε
    σενάριο να στέκει.
    """
    rng = np.random.default_rng(seed)

    # Το OpEx από την ταυτότητα, όχι από τη στήλη
    opex = base["Total_Revenue"] - base["Cost_Of_Revenue"] - base["EBIT"]

    revenue = base["Total_Revenue"] * (1 + rng.normal(0, sigma_rev, N))
    cogs = base["Cost_Of_Revenue"] * (1 + rng.normal(0, sigma_cogs, N))
    ebit = revenue - cogs - opex
    op_margin = (ebit / revenue) * 100
    contrib_marg = np.clip((revenue - cogs) / revenue, None, 1.0)
    current_assets = (base["Current_Assets"] * (1 + rng.normal(0, 0.10, N))).clip(min=0)
    equity = base["Equity"] * (1 + rng.normal(0, 0.10, N))
    gross_profit = revenue - cogs
    mos = ((revenue - opex / contrib_marg) / revenue) * 100

    synthetic = pd.DataFrame({
        "Total_Revenue": revenue, "Cost_Of_Revenue": cogs, "EBIT": ebit,
        "Current_Assets": current_assets,
        "Current_Liabilities": base["Current_Liabilities"],
        "Retained_Earnings": base["Retained_Earnings"], "Equity": equity,
        "Gross_Profit": gross_profit, "Contribution_Margin": contrib_marg,
        "Margin of Safety%": mos, "Operating_Margin%": op_margin,
    })[feat_cols]

    probs = model.predict_proba(synthetic)[:, 1]

    # ΚΑΝΟΝΑΣ ΑΓΚΥΡΩΣΗΣ — τιμή, όχι assert: τα sliders δεν πρέπει να ρίχνουν το app.
    #
    # Το notebook ελέγχει |mean(EBIT) - base_EBIT| < 0.2*|base_EBIT|. Αυτό
    # καταρρέει σε εταιρεία που είναι σχεδόν breakeven: η ROKU έχει EBIT
    # -5.6M σε τζίρο 4.7 δισ. (-0.1%), οπότε το tolerance βγαίνει 1.1M ενώ
    # ο αναπόφευκτος θόρυβος του μέσου σε 10.000 σενάρια είναι 7.6M —
    # ο έλεγχος αποτυγχάνει πάντα, ακόμη κι όταν ο sampler είναι σωστός.
    # (Επαληθεύτηκε: σε 200 seeds το μέσο signed drift είναι ~0 -> αμερόληπτο.)
    #
    # Το πάτωμα στο 1% του τζίρου κρατάει τον έλεγχο ουσιαστικό για όλες τις
    # εταιρείες: εξακολουθεί να πιάνει λάθος κέντρο (stale μεταβλητή, λάθος
    # OpEx), αλλά δεν τιμωρεί τα λεπτά περιθώρια.
    anchor_drift = abs(synthetic["EBIT"].mean() - base["EBIT"])
    anchor_tol = max(0.2 * abs(base["EBIT"]), 0.01 * base["Total_Revenue"])
    anchor_ok = bool(anchor_drift < anchor_tol)

    return {"probs": probs, "synthetic": synthetic,
            "mc_mean": float(probs.mean()),
            "ruin": float((probs >= threshold).mean()),
            "anchor_drift": float(anchor_drift),
            "anchor_tol": float(anchor_tol), "anchor_ok": anchor_ok}


# =========================================================================
# Συμπεράσματα (markdown cells 16 & 19 + η analyze_margin_fluctuations)
# =========================================================================

# Τα κείμενα των markdown cells 16 & 19, με διορθωμένα τα τυπογραφικά.
HANDWRITTEN = {
    "GPRO": """
Παρατηρούμε ότι η εταιρεία **GPRO ακολουθεί πτωτική πορεία** στο διάστημα
2023-2025. Παρατηρείται αρχικά μεγάλη μείωση στα έσοδα. Επίσης δεν υπήρχε
ανάπτυξη, καθώς ο ρυθμός ανάπτυξής της ήταν αρνητικός μεταξύ 2024 και 2025.

Υπήρξε εντυπωσιακή μείωση στο **κεφάλαιο κίνησης**, το οποίο μειώθηκε κατά
186.475.000 $. Αυτό δείχνει ότι η εταιρεία έχει χάσει την ικανότητα άμεσης
ρευστοποίησης και από το 2024 και μετά αδυνατεί να πληρώσει τις
βραχυπρόθεσμες υποχρεώσεις της.

Ακόμη, ο συντελεστής **Z-score έπεσε πολύ κάτω από το όριο του 1.10**:
μάλιστα το 2024 ήταν -0.7 και το 2025 -1.7, τη στιγμή που το 2023 ήταν 1.3.
Αυτό δείχνει ξεκάθαρα ότι υπάρχει μεγάλη πιθανότητα σοβαρού προβλήματος
βιωσιμότητας.

Ακόμη, το γεγονός ότι η επιχείρηση είχε **αρνητικό Operating Margin** δείχνει
πως μεταξύ 2023-2025 υπήρχαν λειτουργικές ζημιές. Συνεπώς τα έσοδα δεν
έφταναν να καλύψουν το κόστος παραγωγής.

Τέλος, όπως φαίνεται και από το γράφημα, η εταιρεία μέσα στην τριετία δεν
κατάφερε να πιάσει το revenue το οποίο θα την έκανε βιώσιμη. **Αποτελεί
ξεκάθαρα τον αδύναμο κρίκο του χαρτοφυλακίου.**
""",
    "FSLY": """
Η εταιρεία δείχνει πως **αναπτύσσεται**, όμως ο δείκτης **Z-score την έχει
σταθερά κάτω του κρίσιμου ορίου 1.10** — δηλαδή η πιθανότητα χρεοκοπίας
είναι μονίμως υψηλή.

Η εταιρεία, όπως φαίνεται, δεν έχει καταφέρει να πιάσει το revenue που θα την
κάνει βιώσιμη και έχει επίσης **λειτουργικές ζημιές**. Το γεγονός αυτό δείχνει
ότι η επιχείρηση έχει θέμα ρευστότητας και αδυνατεί να ξεπληρώσει τις
βραχυπρόθεσμες υποχρεώσεις της.

Όπως φαίνεται και από τον δείκτη **Debt-to-Equity**, πάνω από το 50%
υποδηλώνει ότι η εταιρεία στηρίζεται σε δανεισμό ο οποίος, εν απουσία
λειτουργικών κερδών, λειτουργεί ως επιπλέον «βαρίδι», καθιστώντας την
κεφαλαιακή της δομή εξαιρετικά επισφαλή.
""",
}


def analyze_margin_fluctuations(df: pd.DataFrame) -> list[str]:
    """Η συνάρτηση του cell 11 — επιστρέφει γραμμές αντί να τυπώνει."""
    df = df.sort_values(by=["Ticker", "Fiscal_Year_Num"]).reset_index(drop=True)
    df["Revenue_Growth_%"] = df.groupby("Ticker")["Total_Revenue"].pct_change() * 100
    df["COGS_Growth_%"] = df.groupby("Ticker")["Cost_Of_Revenue"].pct_change() * 100
    df["Margin_Change"] = df.groupby("Ticker")["Operating_Margin%"].diff()

    out = []
    company_data = df.dropna(subset=["Revenue_Growth_%"])
    for _, row in company_data.iterrows():
        year, margin = row["Fiscal_Year_Num"], row["Operating_Margin%"]
        margin_change = row["Margin_Change"]
        rev_g, cogs_g = row["Revenue_Growth_%"], row["COGS_Growth_%"]

        cogs_word = "αυξήθηκε" if cogs_g >= 0 else "μειώθηκε"
        rev_word = "αυξήθηκαν" if rev_g >= 0 else "μειώθηκαν"

        if margin_change < -0.5:
            out.append(f"**[{year}] ΠΤΩΣΗ ΠΕΡΙΘΩΡΙΟΥ** στο {margin:.1f}% "
                       f"(έχασε {abs(margin_change):.1f} μονάδες)")
            if cogs_g > rev_g:
                out.append(f"→ Το μεταβλητό κόστος (COGS) {cogs_word} κατά "
                           f"{abs(cogs_g):.1f}%, ενώ τα έσοδα {rev_word} κατά "
                           f"{abs(rev_g):.1f}%.")
            else:
                out.append(f"→ Τα έσοδα {rev_word} ({abs(rev_g):.1f}%) ταχύτερα "
                           f"από όσο {cogs_word} τα κόστη ({abs(cogs_g):.1f}%).")
        elif margin_change > 0.5:
            out.append(f"**[{year}] ΑΝΑΚΑΜΨΗ ΠΕΡΙΘΩΡΙΟΥ** στο {margin:.1f}% "
                       f"(κέρδισε {margin_change:.1f} μονάδες)")
            if rev_g > cogs_g:
                out.append(f"→ Τα έσοδα {rev_word} κατά {abs(rev_g):.1f}%, ενώ "
                           f"το κόστος (COGS) {cogs_word} κατά {abs(cogs_g):.1f}%.")
            else:
                out.append(f"→ Επιτυχής συγκράτηση κόστους. Το κόστος (COGS) "
                           f"{cogs_word} ({abs(cogs_g):.1f}%) ταχύτερα από τα "
                           f"έσοδα ({abs(rev_g):.1f}%).")
    return out


def generate_insight(ticker: str, data: pd.DataFrame) -> list[str]:
    """Κανόνες πάνω στα νούμερα — για τις εταιρείες χωρίς γραμμένο συμπέρασμα."""
    d = data.sort_values("Fiscal_Year_Num")
    z, mos = d["Z_score"], d["Margin of Safety%"]
    wc, om, de = d["Working_Capital"], d["Operating_Margin%"], d["Debt_to_Equity"]
    growth = d["YoY Revenue Growth"]
    out = []

    if (z >= 3.0).all():
        out.append(f"Η **{ticker}** είναι σταθερά υγιής: το Z-score παραμένει "
                   f"πάνω από το 3.0 σε όλη την περίοδο (χαμηλότερο: {z.min()}).")
    elif (z <= Z_THRESHOLD).all():
        out.append(f"Το Z-score της **{ticker}** βρίσκεται **μόνιμα κάτω από το "
                   f"κρίσιμο όριο του {Z_THRESHOLD}** σε όλα τα έτη — η "
                   f"πιθανότητα χρεοκοπίας είναι διαρκώς υψηλή.")

    if len(z) >= 2 and z.iloc[-1] < z.iloc[0]:
        out.append(f"**Επιδεινούμενη πορεία:** το Z-score έπεσε από "
                   f"{z.iloc[0]} σε {z.iloc[-1]} μέσα στην περίοδο.")

    if len(growth) >= 2 and (z <= Z_THRESHOLD).all() and (growth.iloc[1:] > 0).all():
        out.append(f"Η εταιρεία **αναπτύσσεται** (έσοδα +"
                   f"{growth.iloc[1:].mean() * 100:.1f}% κατά μέσο όρο) αλλά "
                   f"παραμένει **επισφαλής**: η ανάπτυξη δεν έχει μεταφραστεί "
                   f"σε βιωσιμότητα.")

    if (mos < 0).all():
        out.append(f"**Τα έσοδα δεν φτάνουν το νεκρό σημείο** σε κανένα έτος "
                   f"(Margin of Safety από {mos.iloc[0]}% σε {mos.iloc[-1]}%).")
    elif (mos < 0).any():
        years = d.loc[d["Margin of Safety%"] < 0, "Fiscal_Year_Num"].tolist()
        out.append(f"Τα έσοδα έπεσαν κάτω από το νεκρό σημείο στα έτη: "
                   f"{', '.join(years)}.")

    if (wc < 0).any():
        out.append("**Αρνητικό κεφάλαιο κίνησης:** η εταιρεία αδυνατεί να "
                   "καλύψει τις βραχυπρόθεσμες υποχρεώσεις της.")
    if len(wc) >= 2 and wc.iloc[0] > 0 and wc.iloc[-1] < wc.iloc[0] * 0.5:
        out.append(f"**Απώλεια ρευστότητας:** το κεφάλαιο κίνησης μειώθηκε "
                   f"κατά {(wc.iloc[0] - wc.iloc[-1]):,.0f} $.")

    if (om < 0).all():
        out.append(f"**Διαρκείς λειτουργικές ζημιές** (Operating Margin από "
                   f"{om.iloc[0]}% σε {om.iloc[-1]}%): τα έσοδα δεν καλύπτουν "
                   f"το κόστος λειτουργίας.")

    if de.iloc[-1] > 1.0:
        out.append(f"Η εταιρεία **στηρίζεται σε δανεισμό** (Debt-to-Equity "
                   f"{de.iloc[-1]:.2f}) — χωρίς λειτουργικά κέρδη το χρέος "
                   f"λειτουργεί ως επιπλέον βαρίδι.")

    if not out:
        out.append(f"Δεν ενεργοποιήθηκε κανένας κανόνας συναγερμού για την "
                   f"**{ticker}**.")
    return out


# =========================================================================
# Γραφήματα — αυτούσια από το notebook
# =========================================================================

def latest_per_company(df: pd.DataFrame) -> pd.DataFrame:
    """Η τελευταία διαθέσιμη χρονιά ΑΝΑ εταιρεία.

    ΟΧΙ df[df.Fiscal_Year_Num == df.Fiscal_Year_Num.max()]: οι εταιρείες δεν
    καταθέτουν 10-K ταυτόχρονα. Η URBN έχει ήδη FY2026 ενώ οι υπόλοιπες
    σταματούν στο 2025, και η PLAY δεν έχει καθόλου 2024 — ένα κοινό max()
    θα κρατούσε 1 εταιρεία από τις 9. Το notebook το έλυνε με
    hardcoded "2025", που έσπασε μόλις βγήκε το filing της URBN.
    """
    return df.sort_values("Fiscal_Year_Num").groupby("Ticker").tail(1)


def fig_margin_of_safety(df_final: pd.DataFrame) -> go.Figure:
    """Cell 12: waterfall του Margin of Safety ανά εταιρεία."""
    df_margin = latest_per_company(df_final)[
        ["Ticker", "Margin of Safety%"]].copy()
    df_margin = df_margin.sort_values(by="Margin of Safety%", ascending=True)

    fig = go.Figure(go.Waterfall(
        name="Margin of Safety %", orientation="v",
        measure=["absolute"] * len(df_margin),
        x=df_margin["Ticker"], y=df_margin["Margin of Safety%"],
        text=df_margin["Margin of Safety%"].astype(str) + "%",
        textposition="outside",
        connector={"line": {"color": "gray", "dash": "dot"}},
        totals={"marker": {"color": "deep sky blue"}},
    ))
    fig.update_layout(
        title=dict(text="Margin of Safety (%) ανά Εταιρεία — τελευταίο "
                        "διαθέσιμο έτος", font=dict(size=18), x=0.01),
        showlegend=False, plot_bgcolor="white",
        yaxis=dict(title="Margin of Safety (%)", zeroline=True, zerolinewidth=2,
                   zerolinecolor="black", gridcolor="lightgray"),
        xaxis=dict(title="Εταιρείες", type="category"),
        height=520, margin=dict(t=70, b=50, l=60, r=30))
    return fig


def fig_six_panel(df_company: pd.DataFrame, ticker: str) -> go.Figure:
    """Cell 13: 6 μετρικές σε grid. Ήταν καρφωμένο σε GPRO."""
    metrics = ["Total_Revenue", "EBIT", "Margin of Safety%",
               "Operating_Margin%", "Z_score", "Working_Capital"]
    titles = ["Total Revenue", "EBIT", "Margin of Safety (%)",
              "Operating Margin", "Z-Score", "Working Capital"]
    colors = ["#1558d6", "#046e00", "#eaa937", "#c0151d", "#681da8", "#00CC96"]

    fig = make_subplots(rows=3, cols=2, subplot_titles=titles,
                        horizontal_spacing=0.12, vertical_spacing=0.13)
    row, col = 1, 1
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(x=df_company["Fiscal_Year_Num"], y=df_company[metric],
                       mode="lines+markers", name=metric,
                       line=dict(color=colors[i], width=2), marker=dict(size=7)),
            row=row, col=col)
        if metric in ["Margin of Safety%", "EBIT", "Operating_Margin%",
                      "Z_score", "Working_Capital"]:
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          row=row, col=col)
        col += 1
        if col > 2:
            col, row = 1, row + 1

    # Το κρίσιμο όριο του Z-score — δεν είναι στο notebook αλλά κάνει το
    # panel αναγνώσιμο χωρίς να θυμάσαι το 1.10
    fig.add_hline(y=Z_THRESHOLD, line_dash="dot", line_color="red", row=3, col=1)

    fig.update_layout(
        title=dict(text=f"Dashboard Απόδοσης & Βιωσιμότητας: {ticker}",
                   font=dict(size=18), x=0.01, y=0.98),
        height=880, showlegend=False, plot_bgcolor="white",
        hovermode="x unified", margin=dict(t=90, b=40, l=60, r=30))

    # type="category": τα Fiscal_Year_Num είναι strings ("2023"), αλλά το
    # Plotly τα κάνει parse ως αριθμούς και παρεμβάλλει ανύπαρκτα μισά έτη
    # (2,023.5) στον άξονα.
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray",
                     type="category")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_annotations(font_size=13)          # οι τίτλοι των subplots
    return fig


def fig_break_even(df_company: pd.DataFrame, ticker: str) -> go.Figure:
    """Cell 15: έσοδα vs όριο επιβίωσης."""
    d = df_company[np.isfinite(df_company["BEP_Rev"])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Fiscal_Year_Num"], y=d["Total_Revenue"], mode="lines+markers",
        name="Πραγματικά Έσοδα (Total Revenue)",
        line=dict(color="#1558d6", width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(
        x=d["Fiscal_Year_Num"], y=d["BEP_Rev"], mode="lines+markers",
        name="Όριο Επιβίωσης (Break-Even Revenue)",
        line=dict(color="#eaa937", width=3, dash="dash"), marker=dict(size=8)))

    fig.update_layout(
        title=dict(text=f"Ανάλυση Νεκρού Σημείου (Break-Even): {ticker}",
                   font=dict(size=18), x=0.01),
        xaxis_title="Οικονομικό Έτος", yaxis_title="Ποσό ($)",
        plot_bgcolor="white", hovermode="x unified",
        # Το legend οριζόντια ΚΑΤΩ από τον τίτλο, όχι πάνω στο plot area:
        # με y=1.1 και x=0.01 σκέπαζε τις πρώτες τιμές της καμπύλης.
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        height=520, margin=dict(t=100, b=50, l=70, r=30))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray",
                     type="category")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    return fig


# ---------------------------------------------------------------------------
# Τα matplotlib γραφήματα είναι ΟΛΑ cached.
#
# Το Streamlit ξανατρέχει ολόκληρο το script σε κάθε αλληλεπίδραση, και το
# st.tabs() ΔΕΝ είναι lazy: εκτελούνται και τα 4 tabs, ακόμη κι αυτά που δεν
# βλέπεις. Χωρίς cache, κάθε κούνημα ενός slider ξανάχτιζε το pairplot (25
# subplots), το 14x6 heatmap και τα δύο SHAP plots — γραφήματα που δεν
# εξαρτώνται καν από την επιλογή. Και χωρίς plt.close() τα figures έμεναν
# ανοιχτά στη μνήμη του matplotlib, οπότε το app γονάτιζε προοδευτικά.
#
# Οι _-prefixed παράμετροι δεν μπαίνουν στο cache key (δεν είναι hashable).
# Ό,τι ΔΕΝ έχει underscore είναι το πραγματικό key.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def fig_operating_margin(_df_company: pd.DataFrame, ticker: str):
    """Cell 11: seaborn lineplot με annotations — αυτούσιο."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=_df_company, x="Fiscal_Year_Num", y="Operating_Margin%",
                 marker="o", ax=ax)
    for _, row in _df_company.iterrows():
        ax.text(row["Fiscal_Year_Num"], row["Operating_Margin%"],
                f'{row["Operating_Margin%"]:.2f}%', ha="center", va="bottom",
                fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Operating Margin %")
    ax.set_title(f"Change of Operating Margin % — {ticker}")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_target_distribution(_df: pd.DataFrame):
    """Cell 22. Στατικό — δεν εξαρτάται από καμία επιλογή."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=_df, x="Is_Distressed", hue="Is_Distressed", ax=ax,
                  legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Distressed", "Distressed"])
    ax.set_title("Distribution of target value")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontweight="bold")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_correlation_heatmap(_X: pd.DataFrame):
    """Cell 28. Στατικό."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(_X.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_confusion(_y_true, _y_pred, title: str, cmap: str):
    """Cells 37-38. Το title είναι το cache key."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(_y_true, _y_pred), annot=True, fmt="d",
                cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_shap_bar(_shap_values):
    """Cell 39a. Στατικό."""
    fig = plt.figure(figsize=(8, 6))
    shap.plots.bar(_shap_values, show=False)
    plt.title("SHAP Feature Importance (Μέση Απόλυτη Επίδραση)",
              fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_shap_summary(_shap_values, _X_test_scaled, _features):
    """Cell 39b — το beeswarm. Στατικό."""
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(_shap_values.values, _X_test_scaled,
                      feature_names=_features, show=False)
    plt.title("SHAP Summary (Κατανομή Κινδύνου ανά Εταιρεία)",
              fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def fig_pairplot(_X: pd.DataFrame, cols: tuple):
    """Cell 40. Το ακριβότερο γράφημα του app: 5x5 = 25 subplots."""
    grid = sns.pairplot(_X[list(cols)])
    return grid.figure


# max_entries: τα sliders δίνουν 9 tickers x 9 sigma_rev x 9 sigma_cogs x 19
# thresholds συνδυασμούς. Χωρίς όριο, το cache θα φούσκωνε απεριόριστα καθώς
# ο χρήστης παίζει — το LRU κρατάει τους 24 τελευταίους.
@st.cache_resource(show_spinner=False, max_entries=24)
def fig_mc_histogram(_probs, ticker: str, sigma_rev: float, sigma_cogs: float,
                     threshold: float, mc_mean: float):
    """Cell 42. Cache key: ό,τι ελέγχουν τα sliders."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(_probs, bins=60, kde=True, color="#2c3e50", ax=ax)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2,
               label=f"Όριο Συναγερμού (Threshold = {threshold})")
    ax.axvline(x=mc_mean, color="green", linestyle="--", linewidth=2,
               label=f"Μέση πιθανότητα Distress βάση MC ({mc_mean:.2%})")
    ax.axvspan(threshold, 1.0, color="red", alpha=0.1, label="Ζώνη Χρεοκοπίας")
    ax.set_title(f"Monte Carlo: Κατανομή Πιθανότητας Χρεοκοπίας — {ticker}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Πιθανότητα Χρεοκοπίας (0 = Ασφάλεια, 1 = Χρεοκοπία)")
    ax.set_ylabel("Συχνότητα (Αριθμός Σεναρίων)")
    ax.legend()
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False, max_entries=24)
def cached_monte_carlo(ticker: str, sigma_rev: float, sigma_cogs: float,
                       threshold: float, _df: pd.DataFrame, _model,
                       _features: list):
    """10.000 σενάρια ανά συνδυασμό slider — τρέχει μία φορά, όχι σε κάθε rerun."""
    base = _df[_df["Ticker"] == ticker].sort_values("Fiscal_Year_Num").iloc[-1]
    return run_monte_carlo(base, _model, _features, sigma_rev=sigma_rev,
                           sigma_cogs=sigma_cogs, threshold=threshold)


# =========================================================================
# UI
# =========================================================================

def money(x: float) -> str:
    for div, suf in [(1e9, "δισ."), (1e6, "εκατ."), (1e3, "χιλ.")]:
        if abs(x) >= div:
            return f"{x / div:,.2f} {suf} $"
    return f"{x:,.0f} $"


st.title("📊 Dynamic Solvency & Risk Simulator")
st.caption("Early Warning System βασισμένο σε SEC 10-K financials + FRED macro data")

try:
    df_final = load_data()
except Exception as exc:
    st.error(f"Αποτυχία άντλησης δεδομένων: {exc}")
    st.info("Το SEC/FRED μπορεί να είναι προσωρινά μη διαθέσιμο. Δοκίμασε ξανά.")
    st.stop()

ml = train_model(df_final)
tickers = sorted(df_final["Ticker"].unique())

st.sidebar.header("Πλοήγηση")
st.sidebar.metric("Εταιρείες", len(tickers))
st.sidebar.metric("Παρατηρήσεις", len(df_final))
st.sidebar.metric("Έτη", df_final["Fiscal_Year_Num"].nunique())
st.sidebar.caption(f"Portfolio: {', '.join(tickers)}")
st.sidebar.divider()
st.sidebar.caption("Οι TDOC (Healthcare) και TNC αφαιρέθηκαν.")

tab_portfolio, tab_company, tab_model, tab_mc = st.tabs(
    ["🏦 Χαρτοφυλάκιο", "🏢 Εταιρεία", "🤖 Μοντέλο", "🎲 Monte Carlo"])


# --- Tab 1: Χαρτοφυλάκιο -------------------------------------------------
with tab_portfolio:
    latest = latest_per_company(df_final)
    n_distressed = int((latest["Z_score"] <= Z_THRESHOLD).sum())
    worst = latest.loc[latest["Z_score"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Σε κίνδυνο", f"{n_distressed}/{len(latest)}",
              help=f"Z-score ≤ {Z_THRESHOLD}")
    c2.metric("Μέσο Z-score", f"{latest['Z_score'].mean():.2f}")
    c3.metric("Χειρότερη", worst["Ticker"], f"Z-score {worst['Z_score']:.1f}",
              delta_color="inverse")
    c4.metric("Κάτω από νεκρό σημείο",
              int((latest["Margin of Safety%"] < 0).sum()),
              help="Margin of Safety < 0")

    st.plotly_chart(fig_margin_of_safety(df_final), use_container_width=True)

    st.subheader("Χαρτοφυλάκιο — τελευταίο διαθέσιμο έτος ανά εταιρεία")
    table = latest[["Ticker", "Fiscal_Year_Num", "sector", "Z_score",
                    "Margin of Safety%", "Operating_Margin%", "Debt_to_Equity",
                    "Working_Capital"]].sort_values("Z_score")
    st.dataframe(
        table.style
        .background_gradient(subset=["Z_score"], cmap="RdYlGn", vmin=-2, vmax=4)
        .format({"Z_score": "{:.1f}", "Margin of Safety%": "{:.1f}%",
                 "Operating_Margin%": "{:.1f}%", "Debt_to_Equity": "{:.2f}",
                 "Working_Capital": "{:,.0f}"}),
        use_container_width=True, hide_index=True)


# --- Tab 2: Εταιρεία -----------------------------------------------------
#
# @st.fragment: χωρίς αυτό, η αλλαγή εταιρείας ξανατρέχει ΟΛΟ το script και το
# Streamlit ξαναστέλνει στον browser και τα 12 γραφήματα — από όλα τα tabs,
# ακόμη κι αυτά που δεν βλέπεις (το st.tabs δεν είναι lazy). Το cache γλιτώνει
# το ΧΤΙΣΙΜΟ των figures, αλλά όχι τη μεταφορά τους: μετρήθηκε 4.3s ανά αλλαγή
# ακόμη και με 100% cache hits. Το fragment ξανατρέχει ΜΟΝΟ αυτό το μπλοκ.
@st.fragment
def render_company_tab():
    ticker = st.selectbox("Εταιρεία", tickers,
                          index=tickers.index("GPRO") if "GPRO" in tickers else 0)
    d = df_final[df_final["Ticker"] == ticker].sort_values("Fiscal_Year_Num")
    last = d.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Z-score", f"{last['Z_score']:.1f}",
              f"{last['Z_score'] - d.iloc[0]['Z_score']:+.1f} από {d.iloc[0]['Fiscal_Year_Num']}",
              delta_color="normal")
    c2.metric("Margin of Safety", f"{last['Margin of Safety%']:.1f}%")
    c3.metric("Operating Margin", f"{last['Operating_Margin%']:.1f}%")
    c4.metric("Debt / Equity", f"{last['Debt_to_Equity']:.2f}")

    st.subheader("Συμπέρασμα")
    if ticker in HANDWRITTEN:
        st.info(HANDWRITTEN[ticker].strip(), icon="📝")
    else:
        for line in generate_insight(ticker, d):
            st.markdown(f"- {line}")
        st.caption("Αυτόματα παραγόμενο από τα δεδομένα της εταιρείας.")

    st.plotly_chart(fig_six_panel(d, ticker), use_container_width=True)
    st.plotly_chart(fig_break_even(d, ticker), use_container_width=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.pyplot(fig_operating_margin(d, ticker))
    with col_right:
        st.markdown("**Ανάλυση μεταβολών περιθωρίου**")
        lines = analyze_margin_fluctuations(d.copy())
        if lines:
            for line in lines:
                st.markdown(line)
        else:
            st.caption("Καμία μεταβολή περιθωρίου πάνω από 0.5 μονάδες.")
        st.dataframe(d[["Fiscal_Year_Num", "Total_Revenue", "Cost_Of_Revenue"]]
                     .style.format({"Total_Revenue": "{:,.0f}",
                                    "Cost_Of_Revenue": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)


with tab_company:
    render_company_tab()


# --- Tab 3: Μοντέλο ------------------------------------------------------
with tab_model:
    rep = classification_report(ml["y_test"], ml["y_pred"], output_dict=True,
                                zero_division=0)
    rep_custom = classification_report(ml["y_test"], ml["y_pred_custom"],
                                       output_dict=True, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{ml['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{ml['roc_auc']:.4f}")
    c3.metric("Recall @ 0.5", f"{rep['1']['recall']:.2f}", help="κλάση distressed")
    c4.metric("Recall @ 0.35", f"{rep_custom['1']['recall']:.2f}",
              f"{rep_custom['1']['recall'] - rep['1']['recall']:+.2f}")

    st.warning(
        f"Το `Is_Distressed` ορίζεται ως `Z_score ≤ {Z_THRESHOLD}`, και τα "
        f"{len(ml['features'])} features περιέχουν τους αριθμητές του τύπου του "
        f"Altman (EBIT, Equity, Retained Earnings, Current Assets/Liabilities, "
        f"Revenue). Το μοντέλο ανακατασκευάζει τον δείκτη, δεν τον προβλέπει "
        f"ανεξάρτητα — γι' αυτό το ROC-AUC βγαίνει τόσο ψηλά. Test set: "
        f"**{len(ml['y_test'])} γραμμές**.", icon="⚠️")

    st.subheader("Cross-Validation — 3 μοντέλα")
    st.dataframe(ml["cv_summary"].style.format({
        c: "{:.3f}" for c in ml["cv_summary"].columns if c != "model"}),
        use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Κατανομή target")
        st.pyplot(fig_target_distribution(df_final))

    with col2:
        st.subheader("Επιλεγμένα features")
        st.caption(f"Από 20 αριθμητικά, το `f_classif` κράτησε "
                   f"{len(ml['features'])} με p < 0.05:")
        st.code("\n".join(ml["features"]), language=None)

    st.subheader("Correlation heatmap")
    st.pyplot(fig_correlation_heatmap(ml["X_train_final"]))

    st.subheader("Confusion Matrix — το κόστος του False Negative")
    st.caption("Το κατώφλι κατεβαίνει από 0.5 σε 0.35 επειδή μια αδιάγνωστη "
               "χρεοκοπία κοστίζει πολύ περισσότερο από έναν λάθος συναγερμό.")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_confusion(ml["y_test"], ml["y_pred"],
                                "Threshold = 0.5", "coolwarm"))
    with col2:
        st.pyplot(fig_confusion(ml["y_test"], ml["y_pred_custom"],
                                f"Threshold = {RUIN_THRESHOLD}", "Reds"))

    st.subheader("SHAP — ποιοι δείκτες ρυθμίζουν τη χρεοκοπία")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_shap_bar(ml["shap_values"]))
    with col2:
        st.pyplot(fig_shap_summary(ml["shap_values"], ml["X_test_scaled"],
                                   ml["features"]))

    st.subheader("Σχέσεις των top-5 features")
    top5 = tuple(c for c in ["Operating_Margin%", "Current_Assets", "Equity",
                             "Contribution_Margin", "EBIT"]
                 if c in ml["X_train_final"].columns)
    st.pyplot(fig_pairplot(ml["X_train_final"], top5))


# --- Tab 4: Monte Carlo --------------------------------------------------
# @st.fragment: τα sliders στέλνουν event σε κάθε κίνηση. Χωρίς fragment,
# καθένα ξανάτρεχε ολόκληρο το script και ξαναφόρτωνε όλα τα γραφήματα του
# app στον browser — γι' αυτό κολλούσε στο σύρσιμο.
@st.fragment
def render_mc_tab():
    st.markdown("Το ML μοντέλο απαντά «κινδυνεύει **σήμερα**;». Το Monte Carlo "
                "απαντά «τι γίνεται αν οι οδηγοί της **κινηθούν**;» — σοκάρει "
                "έσοδα και κόστη, ξαναϋπολογίζει τα παράγωγα από τις "
                "λογιστικές ταυτότητες, και μετράει σε πόσα από τα 10.000 "
                "σενάρια η εταιρεία περνάει στη ζώνη κινδύνου.")

    c1, c2, c3, c4 = st.columns(4)
    mc_ticker = c1.selectbox("Εταιρεία", tickers,
                             index=tickers.index("YETI") if "YETI" in tickers else 0)
    sigma_rev = c2.slider("σ Revenue", 0.0, 0.40, 0.15, 0.05,
                          help="Τυπική απόκλιση του σοκ στα έσοδα")
    sigma_cogs = c3.slider("σ COGS", 0.0, 0.40, 0.10, 0.05,
                           help="Τυπική απόκλιση του σοκ στο κόστος πωληθέντων")
    threshold = c4.slider("Όριο συναγερμού", 0.05, 0.95, RUIN_THRESHOLD, 0.05)

    base = df_final[df_final["Ticker"] == mc_ticker].sort_values(
        "Fiscal_Year_Num").iloc[-1]
    base_prob = ml["model"].predict_proba(
        ml["df_ml"].loc[[base.name], ml["features"]])[:, 1][0]

    res = cached_monte_carlo(mc_ticker, sigma_rev, sigma_cogs, threshold,
                             df_final, ml["model"], ml["features"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base probability", f"{base_prob:.2%}",
              help="Το μοντέλο πάνω στα πραγματικά στοιχεία της εταιρείας")
    c2.metric("MC mean distress", f"{res['mc_mean']:.2%}",
              f"{res['mc_mean'] - base_prob:+.2%}")
    c3.metric("Probability of Ruin", f"{res['ruin']:.2%}",
              help=f"P(distress ≥ {threshold}) στα 10.000 σενάρια")
    c4.metric("Anchor drift", money(res["anchor_drift"]),
              "εντός ορίου" if res["anchor_ok"] else "ΕΚΤΟΣ ΟΡΙΟΥ",
              delta_color="normal" if res["anchor_ok"] else "inverse")

    if not res["anchor_ok"]:
        st.error("Ο **anchor check** απέτυχε: ο μέσος EBIT των σεναρίων "
                 "απέχει πάρα πολύ από το base case. Σε αυτά τα σ, η "
                 "προσομοίωση δεν είναι πια αγκυρωμένη στην πραγματική "
                 "εταιρεία.", icon="⚠️")

    st.pyplot(fig_mc_histogram(res["probs"], mc_ticker, sigma_rev, sigma_cogs,
                               threshold, res["mc_mean"]))

    with st.expander("Τι σοκάρεται και τι υπολογίζεται"):
        st.markdown(f"""
**Οδηγοί (σοκάρονται):**
- `Total_Revenue` × (1 + N(0, {sigma_rev}))
- `Cost_Of_Revenue` × (1 + N(0, {sigma_cogs}))
- `Current_Assets` × (1 + N(0, 0.10)), clipped σε ≥ 0
- `Equity` × (1 + N(0, 0.10))

**Παράγωγα (από ταυτότητες, ποτέ δειγματοληψία):**
- `OpEx` = Revenue − COGS − EBIT  *(από το base, σταθερό)*
- `EBIT` = Revenue − COGS − OpEx
- `Operating_Margin%` = EBIT / Revenue × 100
- `Contribution_Margin` = (Revenue − COGS) / Revenue, clipped σε ≤ 1
- `Margin of Safety%` = (Revenue − OpEx/CM) / Revenue × 100

Γι' αυτό κάθε σενάριο στέκει λογιστικά: το margin δεν δειγματοληπτείται ποτέ,
υπολογίζεται. Ο **anchor check** επιβεβαιώνει ότι ο μέσος EBIT των 10.000
σεναρίων παραμένει κοντά στο base case — αν αποκλίνει, υπάρχει bug, όχι insight.
        """)
        st.dataframe(res["synthetic"].describe().T.style.format("{:,.2f}"),
                     use_container_width=True)


with tab_mc:
    render_mc_tab()
