"""
Concept Test Simulation Engine — Streamlit UI
==============================================

Self-contained Streamlit app for running concept test simulations.
Upload a concept test JSON, pick one or more digital twins, and run.

Usage:
    streamlit run app.py
"""

import io
import json
import re
from collections import Counter
from pathlib import Path

import anthropic
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths (relative to this file) ───────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CONCEPTS_DIR = DATA_DIR / "concepts"

# ── K-Dimension Definitions ─────────────────────────────────────────

K_DIMENSIONS = [
    {
        "key": "K1",
        "name": "Purchase Intent",
        "question": (
            "Based on the description, how likely would you be to buy "
            "this product if it were available where you shop?"
        ),
        "scale": {
            5: "Would definitely buy",
            4: "Would probably buy",
            3: "Not sure",
            2: "Would probably not buy",
            1: "Definitely would not buy",
        },
    },
    {
        "key": "K2",
        "name": "Value Perception",
        "question": (
            "Considering the price, how would you rate the value "
            "of this product?"
        ),
        "scale": {
            5: "Excellent value",
            4: "Good value",
            3: "Average value",
            2: "Poor value",
            1: "Very poor value",
        },
    },
    {
        "key": "K3",
        "name": "New & Different",
        "question": (
            "How new and different is this product compared to other "
            "products currently available?"
        ),
        "scale": {
            5: "Extremely new and different",
            4: "Very new and different",
            3: "Somewhat new and different",
            2: "Slightly new and different",
            1: "Not at all new and different",
        },
    },
    {
        "key": "K4",
        "name": "Believability",
        "question": "How believable are the claims made about this product?",
        "scale": {
            5: "Very believable",
            4: "Somewhat believable",
            3: "Neither believable nor unbelievable",
            2: "Somewhat unbelievable",
            1: "Not at all believable",
        },
    },
]

RATING_MODEL = "claude-opus-4-20250514"
RATING_TEMPERATURE = 0.3

DECISION_FACTORS = [
    "use_case", "value_price", "taste_flavor", "comparison",
    "health_nutrition", "repurchase_intent", "quality_issues", "convenience",
]


# ── Model / Data Loaders (cached) ───────────────────────────────────

@st.cache_resource
def load_xgb():
    data = joblib.load(DATA_DIR / "granola_model_full.joblib")
    model = data["model"]
    features = data["features"]
    booster = model.get_booster()
    trees = booster.get_dump(with_stats=True)
    feat_splits = {i: [] for i in range(len(features))}
    for t in trees:
        for m in re.finditer(r"\[f(\d+)<([+-]?\d+\.?\d*)\]", t):
            feat_splits[int(m.group(1))].append(float(m.group(2)))
    scales = {}
    for i, fname in enumerate(features):
        vals = sorted(set(feat_splits[i]))
        if len(vals) < 2:
            scales[fname] = vals[0] * 2 if vals else 0.3
            continue
        gaps = [vals[j + 1] - vals[j] for j in range(len(vals) - 1)]
        scales[fname] = max(vals) + min(gaps)
    return model, features, scales


@st.cache_data
def load_audiences():
    with open(DATA_DIR / "synthetic_audiences.json") as f:
        data = json.load(f)
    return data["audiences"]


# ── Engine Functions ─────────────────────────────────────────────────

def normalize_concept(raw):
    """Normalize different concept JSON formats into the engine's expected schema."""
    # Already in engine format
    if "product_headline" in raw:
        return raw
    # Nested {product: {...}, competitors: [...]} format
    if "product" in raw and isinstance(raw["product"], dict):
        p = raw["product"]
        return {
            "product_headline": p.get("name", ""),
            "brand_name": p.get("brand", ""),
            "consumer_insight": p.get("description", ""),
            "main_benefits": p.get("features", []),
            "reasons_to_believe": [],
            "price_point": f"${p['price']}" if "price" in p else "",
            "pack_size": p.get("category", ""),
            "competitors": raw.get("competitors", []),
        }
    return raw


def _join_items(items):
    """Join a list that may contain strings or dicts into a readable string."""
    parts = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            name = item.get("name", item.get("brand", ""))
            price = item.get("price", "")
            feat = item.get("key_feature", "")
            if price:
                parts.append(f"{name} (${price})" + (f" — {feat}" if feat else ""))
            else:
                parts.append(name + (f" — {feat}" if feat else ""))
        else:
            parts.append(str(item))
    return ", ".join(parts)


def format_concept_block(concept):
    benefits = _join_items(concept.get("main_benefits", []))
    rtb = _join_items(concept.get("reasons_to_believe", []))
    competitors = _join_items(concept.get("competitors", []))
    return (
        f"Product Headline: {concept.get('product_headline', '')}\n"
        f"Brand Name: {concept.get('brand_name', '')}\n"
        f"Consumer Insight: {concept.get('consumer_insight', '')}\n"
        f"Main Benefits: {benefits}\n"
        f"Reasons to Believe: {rtb}\n"
        f"Price Point: {concept.get('price_point', '')}\n"
        f"Pack Size: {concept.get('pack_size', '')}\n"
        f"Competitors/Alternatives: {competitors}"
    )


def get_audience_factors(audience):
    profile = audience.get("decision_factor_profile", {})
    return {f: round(max(0.0, min(1.0, float(profile.get(f, 0.5)))), 4)
            for f in DECISION_FACTORS}


def predict_buy(model, features, scales, factor_scores):
    feat_to_json = {f: f.replace("score_", "") for f in features}
    vec = [factor_scores[feat_to_json[f]] * scales[f] for f in features]
    proba = model.predict_proba(np.array([vec]))[0]
    return round(float(proba[1]) * 100, 1)


def build_k_dimension_block():
    lines = []
    for kd in K_DIMENSIONS:
        lines.append(f'{kd["key"]}. {kd["name"]}: "{kd["question"]}"')
        for score in sorted(kd["scale"].keys(), reverse=True):
            lines.append(f"  {score} = {kd['scale'][score]}")
        lines.append("")
    return "\n".join(lines)


def build_k_json_template():
    parts = [
        f'  "{kd["key"]}": {{"score": <int 1-5>, "rationale": "<1-2 sentences>"}}'
        for kd in K_DIMENSIONS
    ]
    return "{\n" + ",\n".join(parts) + "\n}"


def rate_audience(client, concept, buy_prob, factor_scores, audience):
    d = audience["demographics"]
    b = audience["behavioral"]
    concept_block = format_concept_block(concept)
    k_block = build_k_dimension_block()
    k_json = build_k_json_template()

    prompt = f"""\
You are simulating a real consumer's survey response to a product concept test.
Respond AS this specific consumer — adopt their perspective, priorities, biases,
and likely reactions based on their demographic and behavioral profile.

CONSUMER PROFILE:
  Segment: {audience['segment']}
  Age: {d.get('age', '')}
  Gender: {d.get('gender', '')}
  Household Income: {d.get('household_income', '')}
  Education: {d.get('education', '')}
  Marital Status: {d.get('marital_status', '')}
  Children: {d.get('children', '')}
  Employment: {d.get('employment_status', '')}
  Living Situation: {d.get('living_situation', '')}
  Neighborhood: {d.get('neighborhood_type', '')}
  Region: {d.get('region', '')}
  Home Owner: {d.get('home_owner', '')}
  Work Setting: {d.get('work_setting', '')}

  Health/Wellness Priorities: {', '.join(b.get('health_priorities', []))}
  Snack Approach: {b.get('snack_approach', '')}
  Shopping Location: {b.get('shopping_location', '')}

PRODUCT CONCEPT:
{concept_block}

PREDICTIVE MODEL CALIBRATION:
  A calibrated XGBoost model (trained on real concept test survey data) predicts
  that consumers with this profile have a {buy_prob:.1f}% likelihood of purchasing
  this product. The model identified these key factor scores for this concept:
    - Use Case: {factor_scores['use_case']:.2f} (how well it fits daily routine)
    - Quality Issues: {factor_scores['quality_issues']:.2f} (skepticism level — high = more doubt)
    - Value/Price: {factor_scores['value_price']:.2f} (perceived value for money)
    - Health/Nutrition: {factor_scores['health_nutrition']:.2f} (health claim credibility)

  Use this as a grounding signal. Your K1 (Purchase Intent) should broadly align
  with the {buy_prob:.1f}% buy probability, but individual variation is expected.
  A consumer at 80% buy probability would typically rate K1 as 4-5; at 50% as 3;
  at 30% as 1-2.

REAL SURVEY DISTRIBUTION PRIORS (from n=486 real concept test respondents):
  These are the actual score distributions observed in real consumer concept tests.
  Use these as base-rate priors — your individual consumer's score should reflect
  BOTH their specific profile AND these realistic population tendencies.
  Each dimension should be scored INDEPENDENTLY — in real surveys, these questions
  have near-zero correlation with each other (r ≈ 0.03 to 0.10).

  K1 Purchase Intent — real distribution:
    1 (Definitely would not buy): 6%
    2 (Would probably not buy): 12%
    3 (Not sure): 25%
    4 (Would probably buy): 31%
    5 (Would definitely buy): 27%
    Note: Calibrate primarily to the XGBoost buy probability above.

  K2 Value Perception — real distribution:
    1 (Very poor value): 1%
    2 (Poor value): 13%
    3 (Average value): ~0% (almost never chosen in real surveys)
    4 (Good value): 59%
    5 (Excellent value): 27%
    IMPORTANT: Real consumers almost never select "Average value." They form clear
    opinions — most see the value proposition favorably (86% score 4-5). Evaluate
    value holistically (benefits, ingredients, brand trust vs. price) — do NOT
    anchor narrowly on the price-per-unit or the Value/Price factor score above.

  K3 New & Different — real distribution:
    1 (Not at all new and different): 10%
    2 (Slightly new and different): 16%
    3 (Somewhat new and different): 34%
    4 (Very new and different): 23%
    5 (Extremely new and different): 17%
    Note: This is the most evenly spread dimension. Novelty perception varies
    strongly by consumer segment and product category.

Rate this concept on each survey question below. For each, provide a score (1-5)
and a brief rationale (1-2 sentences) explaining WHY this specific consumer
would give that rating given their profile. Score each dimension on its own merits —
do NOT let your rating on one dimension influence another.

{k_block}
Respond with ONLY a valid JSON object:
{k_json}"""

    response = client.messages.create(
        model=RATING_MODEL,
        max_tokens=600,
        temperature=RATING_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    ratings = json.loads(json_match.group() if json_match else text)

    for kd in K_DIMENSIONS:
        key = kd["key"]
        if key not in ratings:
            ratings[key] = {"score": 3, "rationale": "No response generated"}
        elif not isinstance(ratings[key], dict):
            ratings[key] = {"score": int(ratings[key]), "rationale": ""}
        ratings[key]["score"] = max(1, min(5, int(ratings[key]["score"])))

    return ratings


# ── Helper: build results DataFrame ─────────────────────────────────

def results_to_dataframe(results):
    rows = []
    for r in results:
        row = {
            "Audience ID": r["audience_id"],
            "Segment": r["segment"],
            "Age": r["demographics"].get("age", ""),
            "Gender": r["demographics"].get("gender", ""),
            "Income": r["demographics"].get("household_income", ""),
            "Buy Prob %": r["buy_prob"],
        }
        for kd in K_DIMENSIONS:
            key = kd["key"]
            rating = r["k_ratings"].get(key, {})
            score = rating.get("score", "")
            row[f"{key} Score"] = score
            row[f"{key} Label"] = kd["scale"].get(score, "")
            row[f"{key} Rationale"] = rating.get("rationale", "")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(df):
    summary = {}
    for kd in K_DIMENSIONS:
        key = kd["key"]
        col = f"{key} Score"
        scores = df[col].dropna().astype(int)
        if len(scores) == 0:
            continue
        c = Counter(scores)
        n = len(scores)
        dist = {s: c.get(s, 0) for s in range(1, 6)}
        dist_pct = {s: c.get(s, 0) / n * 100 for s in range(1, 6)}
        t2b = (c.get(4, 0) + c.get(5, 0)) / n * 100
        b2b = (c.get(1, 0) + c.get(2, 0)) / n * 100
        summary[key] = {
            "name": kd["name"],
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "n": n,
            "dist": dist,
            "dist_pct": dist_pct,
            "top2box": t2b,
            "bot2box": b2b,
        }
    return summary


# ── Streamlit UI ─────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Concept Test Simulator",
        page_icon="🧪",
        layout="wide",
    )

    st.title("Concept Test Simulation Engine")
    st.caption("Two-layer simulation: XGBoost buy probability + LLM K-dimension ratings")

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("API Key")
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Required to run simulations. Not stored anywhere.",
        )
        if api_key:
            st.session_state["api_key"] = api_key
        elif "api_key" not in st.session_state:
            st.session_state["api_key"] = ""

        st.divider()
        st.header("Concept Test")

        concept = None
        source = st.radio(
            "Concept source",
            ["Upload JSON", "Use sample"],
            horizontal=True,
        )

        if source == "Upload JSON":
            uploaded = st.file_uploader(
                "Upload concept test JSON",
                type=["json"],
                help="JSON with product_headline, brand_name, main_benefits, etc.",
            )
            if uploaded:
                try:
                    concept = normalize_concept(json.load(uploaded))
                    st.success(f"Loaded: {concept.get('product_headline', 'Unknown')}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
        else:
            def _is_concept_json(path):
                try:
                    with open(path) as fh:
                        d = json.load(fh)
                    if not isinstance(d, dict):
                        return False
                    return "product_headline" in d or "product" in d
                except Exception:
                    return False

            candidates = sorted(CONCEPTS_DIR.glob("*.json")) if CONCEPTS_DIR.exists() else []
            unique = [f for f in candidates if _is_concept_json(f)]

            if unique:
                # Build labels from actual product names
                label_map = []
                for f in unique:
                    with open(f) as fh:
                        d = json.load(fh)
                    n = normalize_concept(d)
                    label_map.append(n.get("product_headline", f.stem.replace("_", " ").title()))
                choice = st.selectbox("Select concept", label_map, index=0)
                idx = label_map.index(choice)
                with open(unique[idx]) as f:
                    concept = normalize_concept(json.load(f))
            else:
                st.warning("No concept JSONs found in data/concepts/")

        if concept:
            st.divider()
            st.subheader("Concept Preview")
            st.markdown(f"**{concept.get('product_headline', '')}**")
            st.markdown(f"Brand: {concept.get('brand_name', '')}")
            st.markdown(f"Price: {concept.get('price_point', '')}")
            benefits = concept.get("main_benefits", [])
            if benefits:
                st.markdown("**Benefits:** " + ", ".join(
                    b if isinstance(b, str) else str(b) for b in benefits
                ))
            with st.expander("Full concept details"):
                st.json(concept)

    if not concept:
        st.info("Upload or select a concept test JSON in the sidebar to begin.")
        return

    # ── Load model and audiences ─────────────────────────────────────
    model, features, scales = load_xgb()
    audiences = load_audiences()

    # ── Audience Selection ───────────────────────────────────────────
    st.header("Select Digital Twins")

    mode = st.radio(
        "Run mode",
        ["Single twin", "Batch"],
        horizontal=True,
        help="Single: pick one audience and see detailed results. Batch: run multiple.",
    )

    audience_summary = []
    for a in audiences:
        d = a["demographics"]
        audience_summary.append({
            "id": a["audience_id"],
            "segment": a["segment"],
            "age": d.get("age", ""),
            "gender": d.get("gender", ""),
            "income": d.get("household_income", ""),
            "label": (
                f"#{a['audience_id']} — {a['segment']}, "
                f"{d.get('age', '?')}{d.get('gender', '?')[0]}, "
                f"{d.get('household_income', '?')}"
            ),
        })

    selected_audiences = []

    if mode == "Single twin":
        col1, col2, col3 = st.columns(3)
        with col1:
            segments = sorted(set(a["segment"] for a in audiences))
            seg_filter = st.selectbox("Filter by segment", ["All"] + segments)
        with col2:
            gender_filter = st.selectbox("Filter by gender", ["All", "Man", "Woman"])
        with col3:
            age_filter = st.selectbox("Filter by age range", [
                "All", "18-24", "25-34", "35-44", "45-54", "55+"
            ])

        filtered = audience_summary
        if seg_filter != "All":
            filtered = [a for a in filtered if a["segment"] == seg_filter]
        if gender_filter != "All":
            filtered = [a for a in filtered if a["gender"] == gender_filter]
        if age_filter != "All":
            lo, hi = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
                      "45-54": (45, 54), "55+": (55, 100)}[age_filter]
            filtered = [a for a in filtered
                        if isinstance(a["age"], int) and lo <= a["age"] <= hi]

        if filtered:
            labels = [a["label"] for a in filtered]
            choice = st.selectbox("Select audience", labels)
            chosen_id = filtered[labels.index(choice)]["id"]
            selected_audiences = [a for a in audiences if a["audience_id"] == chosen_id]
        else:
            st.warning("No audiences match the filters")

    else:  # Batch
        col1, col2 = st.columns(2)
        with col1:
            batch_mode = st.selectbox("Batch selection", [
                "All 100 audiences",
                "By segment",
                "Custom IDs",
            ])
        with col2:
            if batch_mode == "By segment":
                segments = sorted(set(a["segment"] for a in audiences))
                seg_pick = st.multiselect("Segments", segments, default=segments)
                selected_audiences = [a for a in audiences if a["segment"] in seg_pick]
            elif batch_mode == "Custom IDs":
                id_input = st.text_input(
                    "Audience IDs (comma-separated)",
                    placeholder="1, 5, 10, 25",
                )
                if id_input:
                    try:
                        ids = [int(x.strip()) for x in id_input.split(",") if x.strip()]
                        selected_audiences = [a for a in audiences if a["audience_id"] in ids]
                    except ValueError:
                        st.error("Enter comma-separated numbers")
            else:
                selected_audiences = audiences

        if selected_audiences:
            seg_counts = Counter(a["segment"] for a in selected_audiences)
            st.caption(
                f"**{len(selected_audiences)} audiences selected** — "
                + ", ".join(f"{s}: {c}" for s, c in sorted(seg_counts.items()))
            )

    if not selected_audiences:
        return

    # ── Preview selected audience(s) ─────────────────────────────────
    if mode == "Single twin" and selected_audiences:
        a = selected_audiences[0]
        d = a["demographics"]
        b = a["behavioral"]
        with st.expander("Audience profile", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Demographics**")
                st.markdown(f"- Segment: {a['segment']}")
                st.markdown(f"- Age: {d.get('age', '')}")
                st.markdown(f"- Gender: {d.get('gender', '')}")
                st.markdown(f"- Income: {d.get('household_income', '')}")
                st.markdown(f"- Education: {d.get('education', '')}")
                st.markdown(f"- Children: {d.get('children', 'None')}")
            with c2:
                st.markdown("**Location & Living**")
                st.markdown(f"- Region: {d.get('region', '')}")
                st.markdown(f"- Neighborhood: {d.get('neighborhood_type', '')}")
                st.markdown(f"- Home owner: {d.get('home_owner', '')}")
                st.markdown(f"- Work setting: {d.get('work_setting', '')}")
                st.markdown(f"- Living: {d.get('living_situation', '')}")
            with c3:
                st.markdown("**Behavioral**")
                st.markdown(f"- Health priorities: {', '.join(b.get('health_priorities', []))}")
                st.markdown(f"- Snack approach: {b.get('snack_approach', '')}")
                st.markdown(f"- Shopping: {b.get('shopping_location', '')}")

    # ── Run Simulation ───────────────────────────────────────────────
    st.divider()

    run_btn = st.button(
        f"Run Simulation ({len(selected_audiences)} audience{'s' if len(selected_audiences) != 1 else ''})",
        type="primary",
        use_container_width=True,
    )

    if run_btn:
        if not st.session_state.get("api_key"):
            st.error("Enter your Anthropic API key in the sidebar to run simulations.")
            st.stop()
        client = anthropic.Anthropic(api_key=st.session_state["api_key"])
        results = []

        progress = st.progress(0, text="Starting simulation...")
        total = len(selected_audiences)

        for i, audience in enumerate(selected_audiences):
            aid = audience["audience_id"]
            seg = audience["segment"]
            age = audience["demographics"]["age"]
            gender = audience["demographics"]["gender"]

            progress.progress(
                i / total,
                text=f"Simulating audience #{aid} ({seg}, {age}{gender[0]}) — {i+1}/{total}",
            )

            # Layer 1
            factor_scores = get_audience_factors(audience)
            buy_prob = predict_buy(model, features, scales, factor_scores)

            # Layer 2
            try:
                ratings = rate_audience(client, concept, buy_prob, factor_scores, audience)
            except Exception as e:
                ratings = {
                    kd["key"]: {"score": 0, "rationale": f"Error: {e}"}
                    for kd in K_DIMENSIONS
                }

            results.append({
                "audience_id": aid,
                "segment": seg,
                "demographics": audience["demographics"],
                "behavioral": audience["behavioral"],
                "factor_scores": factor_scores,
                "buy_prob": buy_prob,
                "k_ratings": ratings,
            })

        progress.progress(1.0, text="Simulation complete!")

        st.session_state["results"] = results
        st.session_state["concept"] = concept

    # ── Display Results ──────────────────────────────────────────────
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    concept = st.session_state["concept"]
    df = results_to_dataframe(results)

    st.header("Results")

    if mode == "Single twin" and len(results) == 1:
        r = results[0]
        st.subheader(f"Audience #{r['audience_id']} — {r['segment']}")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Buy Probability", f"{r['buy_prob']}%")
        with col2:
            st.progress(r["buy_prob"] / 100)

        st.subheader("K-Dimension Ratings")
        cols = st.columns(len(K_DIMENSIONS))
        for j, kd in enumerate(K_DIMENSIONS):
            key = kd["key"]
            rating = r["k_ratings"].get(key, {})
            score = rating.get("score", 0)
            label = kd["scale"].get(score, "")
            with cols[j]:
                st.metric(f"{key}: {kd['name']}", f"{score}/5", help=label)
                st.caption(label)
                st.markdown(f"*{rating.get('rationale', '')}*")

        with st.expander("Decision Factor Scores (Layer 1)"):
            factor_df = pd.DataFrame([
                {"Factor": f.replace("_", " ").title(), "Score": v}
                for f, v in r["factor_scores"].items()
            ])
            st.bar_chart(factor_df, x="Factor", y="Score", horizontal=True)

    else:
        summary = compute_summary(df)

        st.subheader("Summary Metrics")
        cols = st.columns(len(summary))
        for j, (key, s) in enumerate(summary.items()):
            with cols[j]:
                st.metric(
                    f"{key}: {s['name']}",
                    f"Mean: {s['mean']:.2f}",
                    help=f"n={s['n']}, std={s['std']:.2f}",
                )
                st.caption(f"T2B: {s['top2box']:.0f}% | B2B: {s['bot2box']:.0f}%")

        st.subheader("Score Distributions")
        dist_cols = st.columns(len(summary))
        for j, (key, s) in enumerate(summary.items()):
            with dist_cols[j]:
                chart_df = pd.DataFrame({
                    "Score": list(s["dist_pct"].keys()),
                    "Percentage": list(s["dist_pct"].values()),
                })
                st.bar_chart(chart_df, x="Score", y="Percentage")
                st.caption(f"{key}: {s['name']}")

        segments_in_results = df["Segment"].unique()
        if len(segments_in_results) > 1:
            st.subheader("By Segment")
            seg_rows = []
            for seg in sorted(segments_in_results):
                seg_df = df[df["Segment"] == seg]
                row = {"Segment": seg, "n": len(seg_df)}
                for kd in K_DIMENSIONS:
                    key = kd["key"]
                    col = f"{key} Score"
                    scores = seg_df[col].dropna().astype(int)
                    if len(scores) > 0:
                        row[f"{key} Mean"] = round(float(scores.mean()), 2)
                        c = Counter(scores)
                        row[f"{key} T2B"] = f"{(c.get(4,0)+c.get(5,0))/len(scores)*100:.0f}%"
                seg_rows.append(row)
            st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

        st.subheader("Full Results")
        display_cols = ["Audience ID", "Segment", "Age", "Gender", "Income", "Buy Prob %"]
        for kd in K_DIMENSIONS:
            display_cols.extend([f"{kd['key']} Score", f"{kd['key']} Label"])
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    if len(results) > 1:
        with st.expander("View rationales"):
            for r in results:
                st.markdown(f"**#{r['audience_id']} — {r['segment']}** (buy={r['buy_prob']}%)")
                for kd in K_DIMENSIONS:
                    key = kd["key"]
                    rating = r["k_ratings"].get(key, {})
                    st.markdown(
                        f"- {key}={rating.get('score', '?')}: "
                        f"{rating.get('rationale', '')}"
                    )
                st.divider()

    # ── Download CSV ─────────────────────────────────────────────────
    st.divider()
    csv_buffer = io.StringIO()
    full_df = results_to_dataframe(results)
    full_df.to_csv(csv_buffer, index=False)
    slug = re.sub(r"[^a-z0-9]+", "_", concept.get("product_headline", "results").lower()).strip("_")[:30]

    st.download_button(
        label="Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"sim_results_{slug}.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
