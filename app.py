"""CineMatch: interactive recommender-systems demo (CSE4143).   Run with:  streamlit run app.py"""
from __future__ import annotations

import html
import inspect

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from recsys import attacks as A
from recsys import data as D
from recsys import evaluation as E
from recsys import models as M
from recsys import ui

AUTHOR = "Arnav Rathi"  

st.set_page_config(page_title="CineMatch · Recommender Systems", page_icon="🎬", layout="wide")
st.markdown(ui.inject_css(), unsafe_allow_html=True)

def show(fig, target=st):
    """Render a plotly figure full-width on both old and new Streamlit versions."""
    params = inspect.signature(target.plotly_chart).parameters
    if "width" in params:
        target.plotly_chart(fig, width="stretch")
    elif "use_container_width" in params:
        target.plotly_chart(fig, use_container_width=True)
    else:
        show(fig, target)


MODEL_ORDER = [M.POPULAR, M.TOP_RATED, M.CONTENT, M.USER_CF, M.ITEM_CF, M.HYBRID]
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10))


# --------------------------------------------------------------------------- #
# Data + models (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading MovieLens 100K (downloads on first run)…")
def load_data():
    # download=False: never block the page on a slow network; uses data/ml-100k if present, else synthetic data
    return D.get_dataset("data", download=False)


@st.cache_resource(show_spinner="Training models: content-based, user-CF, item-CF…")
def get_system(_ds, key):
    rm = M.RatingMatrix(_ds.ratings, _ds.n_users, _ds.n_items)
    X, names = M.build_item_features(_ds.items)
    return {"rm": rm, "X": X, "names": names, "models": M.fit_all(rm, X, names)}


@st.cache_data(show_spinner="Running the offline evaluation protocol (about 10-30 s)…")
def run_lab(_ds, key, test_frac, k, threshold, min_support, seed):
    X, names = M.build_item_features(_ds.items)
    return E.run_full_evaluation(_ds.ratings, _ds.n_users, _ds.n_items, X, names,
                                 test_frac, k, threshold, min_support, seed)


@st.cache_data(show_spinner="Injecting fake profiles, retraining, detecting, defending (about 15-40 s)…")
def run_security(_ds, key, model_name, kind, target, size, filler, detector, flag_pct):
    rm_ = M.RatingMatrix(_ds.ratings, _ds.n_users, _ds.n_items)
    factory = (lambda: M.UserKNN(40)) if model_name == M.USER_CF else (lambda: M.ItemKNN(40))
    sizes = sorted({0.0, 2.5, 5.0, 10.0, 15.0, float(size)})
    return A.attack_sweep(rm_, factory, target, kind, sizes, filler, detector, flag_pct)


ds, data_warning = load_data()
system = get_system(ds, ds.name)
rm, X, names, models = system["rm"], system["X"], system["names"], system["models"]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Recommender Systems · CSE4143")
    st.markdown("### Hybrid blend")
    st.caption("Weights of the three components in the Hybrid recommender.")
    w_c = st.slider("Content-based", 0.0, 1.0, M.DEFAULT_WEIGHTS[M.CONTENT], 0.05)
    w_u = st.slider("User-based CF", 0.0, 1.0, M.DEFAULT_WEIGHTS[M.USER_CF], 0.05)
    w_i = st.slider("Item-based CF", 0.0, 1.0, M.DEFAULT_WEIGHTS[M.ITEM_CF], 0.05)
    weights = {M.CONTENT: w_c, M.USER_CF: w_u, M.ITEM_CF: w_i}
    if sum(weights.values()) <= 0:
        weights = dict(M.DEFAULT_WEIGHTS)
    st.markdown("### Recommendations")
    top_n = st.slider("How many to show", 3, 20, 9)
    min_support = st.slider("Hide movies with fewer ratings than", 0, 100, 20)
    st.markdown("---")
    st.caption(f"Dataset: **{ds.name}**  \n{ds.n_users:,} users · {ds.n_items:,} movies · {len(ds.ratings):,} ratings")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(ui.hero(
    "CineMatch",
    "A movie recommender built from scratch: content-based, user-CF, item-CF and a tuned hybrid, "
    "with offline evaluation and a shilling-attack lab.",
    [ds.name, f"{ds.n_users:,} users", f"{ds.n_items:,} movies", f"{len(ds.ratings):,} ratings"]),
    unsafe_allow_html=True)
if data_warning:
    st.markdown('<div class="warn">⚠️ <b>Running on synthetic demo data</b>, not MovieLens. '
                f'Reason: {html.escape(data_warning)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_model(name: str, model_dict: dict):
    return M.make_hybrid(model_dict, weights) if name == M.HYBRID else model_dict[name]


def title_of(i: int) -> str:
    return str(ds.items.at[i, "title"])


def explain(name: str, model_dict: dict, rm_: M.RatingMatrix, u: int, i: int) -> str:
    def item_reason() -> str:
        js = model_dict[M.ITEM_CF].explain(u, i)
        return "Because you liked " + " & ".join(title_of(j) for j in js) if js else ""

    def content_reason() -> str:
        gs = model_dict[M.CONTENT].explain(u, i)
        return "Matches your taste for " + ", ".join(gs) if gs else ""

    def user_reason() -> str:
        liked, rated = model_dict[M.USER_CF].explain(u, i)
        return f"{liked} of {rated} users with similar taste rated it 4★+" if rated else ""

    if name == M.POPULAR:
        return f"Rated by {int(rm_.item_count[i])} users"
    if name == M.TOP_RATED:
        return f"Average rating {rm_.item_mean_raw[i]:.2f} from {int(rm_.item_count[i])} ratings"
    if name == M.CONTENT:
        return content_reason()
    if name == M.USER_CF:
        return user_reason()
    if name == M.ITEM_CF:
        return item_reason()
    return item_reason() or user_reason() or content_reason()


def rec_cards(name: str, model_dict: dict, rm_: M.RatingMatrix, u: int, n: int) -> str:
    model = get_model(name, model_dict)
    cards = []
    for rank, (i, pred) in enumerate(model.recommend(u, n, min_support), 1):
        cards.append(ui.movie_card(rank, title_of(i), ds.items.at[i, "year"], ds.items.at[i, "genres"], pred,
                                   explain(name, model_dict, rm_, u, i)))
    return ui.card_grid(cards) if cards else "<i>No candidates: lower the minimum-ratings filter.</i>"


def random_user():
    st.session_state["uid"] = int(np.random.randint(ds.n_users))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_rec, tab_new, tab_sim, tab_lab, tab_sec, tab_data, tab_map = st.tabs(
    ["🎯 Recommend", "🧑‍🎤 New user", "🔗 Similar movies", "📊 Model Lab", "🛡️ Security Lab", "🗂️ Data", "📚 Syllabus map"])

# ------------------------------ Recommend ----------------------------------- #
with tab_rec:
    st.session_state.setdefault("uid", 0)
    c1, c2, c3 = st.columns([1.1, 3, 0.9])
    c1.selectbox("User", list(range(ds.n_users)), format_func=lambda u: f"User {u + 1}", key="uid")
    algo = c2.radio("Algorithm", MODEL_ORDER, index=len(MODEL_ORDER) - 1, horizontal=True)
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c3.button("🎲 Surprise me", on_click=random_user)

    u = int(st.session_state["uid"])
    info = ds.users.loc[u]
    st.markdown(ui.kpi_row([
        ("User", f"#{u + 1}", f"{info['gender']} · age {info['age']} · {info['occupation']}"),
        ("Movies rated", int(rm.user_count[u]), "in the full history"),
        ("Average rating", f"{rm.user_mean[u]:.2f}", "how generous this user is"),
        ("Algorithm", algo, "change it above"),
    ]), unsafe_allow_html=True)

    view = st.radio("View", ["Single algorithm", "Compare algorithms"], horizontal=True, label_visibility="collapsed")
    if view == "Single algorithm":
        left, right = st.columns([2.3, 1])
        with left:
            st.markdown(f'<div class="section-title">Top {top_n} for User {u + 1} · {algo}</div>', unsafe_allow_html=True)
            st.markdown(rec_cards(algo, models, rm, u, top_n), unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-title">Learned taste profile</div>', unsafe_allow_html=True)
            prof = models[M.CONTENT].taste_profile(u)
            extremes = pd.concat([prof.nlargest(5), prof.nsmallest(5)])
            extremes = extremes[~extremes.index.duplicated()]
            tp = pd.DataFrame({"Genre": extremes.index, "Affinity": extremes.to_numpy()})
            tp["Direction"] = np.where(tp["Affinity"] >= 0, "likes", "dislikes")
            fig = px.bar(tp.sort_values("Affinity"), x="Affinity", y="Genre", orientation="h", color="Direction",
                         color_discrete_map={"likes": "#7C5CFF", "dislikes": "#FF5C8A"})
            fig.update_layout(**PLOT, height=330, showlegend=False)
            show(fig, st)
            with st.expander("This user's favourite movies"):
                hist = ds.ratings[ds.ratings["u"] == u].sort_values(["r", "ts"], ascending=False).head(10)
                st.dataframe(pd.DataFrame({
                    "Movie": [ds.label(int(i)) for i in hist["i"]], "Rating": hist["r"].to_numpy()}), hide_index=True)
    else:
        picked = st.multiselect("Algorithms to compare", MODEL_ORDER,
                                default=[M.POPULAR, M.CONTENT, M.USER_CF, M.ITEM_CF, M.HYBRID])
        if picked:
            cols = st.columns(len(picked))
            for col, name in zip(cols, picked):
                with col:
                    st.markdown(f'<div class="section-title">{html.escape(name)}</div>', unsafe_allow_html=True)
                    for i, pred in get_model(name, models).recommend(u, 6, min_support):
                        st.markdown(ui.mini_item(title_of(i), f"★ {pred:.2f} · " + ", ".join(ds.items.at[i, "genres"][:2])),
                                    unsafe_allow_html=True)

# ------------------------------ New user ------------------------------------ #
with tab_new:
    st.markdown("### Cold start: tell us what you like")
    st.caption("Pick a few movies you loved (5★) and disliked (1★). A new profile is added to the rating matrix, "
               "all models are re-trained, and you get personalised recommendations.")
    popular = [int(i) for i in np.argsort(-rm.item_count)[:500]]
    loved = st.multiselect("Movies you loved (5★)", popular, format_func=lambda i: ds.label(i))
    disliked = st.multiselect("Movies you disliked (1★)", popular, format_func=lambda i: ds.label(i))
    new_algo = st.radio("Algorithm ", [M.CONTENT, M.USER_CF, M.ITEM_CF, M.HYBRID], index=3, horizontal=True)
    if st.button("✨ Get my recommendations"):
        if len(loved) + len(disliked) < 3:
            st.warning("Please pick at least 3 movies in total so there is something to learn from.")
        else:
            with st.spinner("Adding you to the rating matrix and re-training…"):
                ratings_new = {i: 1.0 for i in disliked}
                ratings_new.update({i: 5.0 for i in loved})
                rm_new = rm.add_user(ratings_new)
                models_new = M.fit_all(rm_new, X, names)
            st.markdown(rec_cards(new_algo, models_new, rm_new, rm_new.n_users - 1, top_n), unsafe_allow_html=True)

# ------------------------------ Similar movies ------------------------------ #
with tab_sim:
    st.markdown("### Find similar movies")
    pool = np.flatnonzero(rm.item_count >= 5)
    pool = [int(i) for i in pool[np.argsort(-rm.item_count[pool])]]
    sc1, sc2 = st.columns([3, 1])
    seed_item = sc1.selectbox("Pick a movie (type to search)", pool, format_func=lambda i: ds.label(i))
    n_sim = sc2.slider("Neighbours", 5, 12, 8)
    st.markdown(ui.kpi_row([
        ("Movie", title_of(seed_item), ", ".join(ds.items.at[seed_item, "genres"])),
        ("Ratings", int(rm.item_count[seed_item]), "in the dataset"),
        ("Average rating", f"{rm.item_mean_raw[seed_item]:.2f}", "out of 5"),
    ]), unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="section-title">👥 People who liked it also liked (item-CF)</div>', unsafe_allow_html=True)
        for j, sim in models[M.ITEM_CF].similar_items(seed_item, n_sim):
            st.markdown(ui.mini_item(ds.label(j), f"similarity {sim:.2f} · " + ", ".join(ds.items.at[j, "genres"][:3])),
                        unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="section-title">🏷️ Similar by content (genres + era)</div>', unsafe_allow_html=True)
        for j, sim in models[M.CONTENT].similar_items(seed_item, n_sim):
            st.markdown(ui.mini_item(ds.label(j), f"similarity {sim:.2f} · " + ", ".join(ds.items.at[j, "genres"][:3])),
                        unsafe_allow_html=True)

# ------------------------------ Model Lab ----------------------------------- #
with tab_lab:
    st.markdown("### Offline evaluation on historical data")
    st.caption("Each user's ratings are split 80/20. Models train on the 80% and are scored on the hidden 20%.")
    l1, l2, l3, l4 = st.columns(4)
    lab_frac = l1.slider("Hidden test share", 0.1, 0.4, 0.2, 0.05)
    lab_k = l2.slider("K for top-K metrics", 5, 20, 10)
    lab_thr = l3.selectbox("Relevant if rating ≥", [3, 4, 5], index=1)
    lab_ms = l4.slider("Min ratings per candidate", 0, 100, 20)
    if st.button("▶️ Run evaluation"):
        st.session_state["lab"] = run_lab(ds, ds.name, lab_frac, lab_k, float(lab_thr), lab_ms, 42)

    res = st.session_state.get("lab")
    if res is None:
        st.info("Press **Run evaluation** to train on the training split and score every model on the hidden ratings.")
    else:
        table = res["table"]
        ndcg_col = next(c for c in table.columns if c.startswith("NDCG@"))
        prec_col = next(c for c in table.columns if c.startswith("Precision@"))
        rec_col = next(c for c in table.columns if c.startswith("Recall@"))
        best_rmse = table.loc[table["RMSE"].idxmin()]
        best_ndcg = table.loc[table[ndcg_col].idxmax()]
        st.markdown(ui.kpi_row([
            ("Training ratings", f"{res['n_train']:,}", f"test: {res['n_test']:,}"),
            ("Best RMSE", f"{best_rmse['RMSE']:.3f}", best_rmse["Model"]),
            (f"Best {ndcg_col}", f"{best_ndcg[ndcg_col]:.3f}", best_ndcg["Model"]),
            ("Tuned hybrid blend", " / ".join(f"{res['weights'][k]:.1f}" for k in (M.CONTENT, M.USER_CF, M.ITEM_CF)),
             "content / user-CF / item-CF"),
        ]), unsafe_allow_html=True)
        st.dataframe(table.round(4), hide_index=True)

        g1, g2 = st.columns(2)
        fig = px.bar(table, x="Model", y=["RMSE", "MAE"], barmode="group", title="Rating prediction error (lower is better)",
                     color_discrete_sequence=ui.PALETTE)
        fig.update_layout(**PLOT, legend_title_text="")
        show(fig, g1)
        melted = table.melt(id_vars="Model", value_vars=[prec_col, rec_col, ndcg_col], var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                     title="Top-N ranking quality (higher is better)", color_discrete_sequence=ui.PALETTE)
        fig.update_layout(**PLOT, legend_title_text="")
        show(fig, g2)

        g3, g4 = st.columns(2)
        fig = px.line(res["k_sweep"], x="k (neighbours)", y="RMSE", color="Model", markers=True,
                      title="Effect of neighbourhood size k", color_discrete_sequence=ui.PALETTE)
        fig.update_layout(**PLOT, legend_title_text="")
        show(fig, g3)
        fig = px.bar(res["metric_cmp"], x="Model", y="RMSE", color="Similarity", barmode="group",
                     title="Pearson vs cosine similarity", color_discrete_sequence=ui.PALETTE)
        fig.update_layout(**PLOT, legend_title_text="")
        show(fig, g4)
        with st.expander("Hybrid weight search (validation split carved out of the training data)"):
            st.dataframe(res["weight_grid"].round(3), hide_index=True)
        with st.expander("How to read these results"):
            st.markdown(
                "- **RMSE / MAE** measure how close predicted ratings are to the hidden true ratings.\n"
                "- **Precision/Recall/NDCG@K** measure the quality of the top-K list: a recommended movie counts as a "
                "hit if the user rated it at or above the relevance threshold in the hidden test set.\n"
                "- **Coverage** is the share of the (filtered) catalogue that appears in anyone's top-K list. "
                "Popularity baselines have very low coverage.\n"
                "- Hybrid weights are tuned on a validation split taken from the training data, so the test set "
                "stays untouched.")

# ------------------------------ Security Lab -------------------------------- #
with tab_sec:
    st.markdown("### Shilling attacks and robust recommendation")
    st.caption("An attacker injects fake users that all give one target movie 5★, hidden among filler ratings. "
               "We measure how much the target's predicted rating rises, then try to detect and remove the fake profiles.")
    cands = [int(i) for i in A.target_candidates(rm)]
    default_t = A.default_target(rm)
    k1, k2, k3 = st.columns(3)
    sec_model = k1.radio("Victim system", [M.USER_CF, M.ITEM_CF], horizontal=True)
    sec_kind = k2.selectbox("Attack type", A.ATTACK_KINDS, index=1)
    sec_det = k3.selectbox("Detector", A.DETECTORS)
    k4, k5, k6 = st.columns(3)
    sec_size = k4.slider("Attack size (% of users)", 1.0, 20.0, 10.0, 0.5)
    sec_fill = k5.slider("Filler ratings per fake profile (% of movies)", 1.0, 10.0, 5.0, 0.5)
    sec_flag = k6.slider("Detector flags top (% of users)", 2.0, 20.0, 8.0, 0.5)
    sec_target = st.selectbox(
        "Target movie (a niche title the attacker wants to promote)", cands, index=cands.index(default_t),
        format_func=lambda i: f"{ds.label(i)} · {int(rm.item_count[i])} ratings · avg {rm.item_mean_raw[i]:.2f}")
    with st.expander("What do the attacks and detectors do?"):
        st.markdown(
            "- **Random**: filler ratings drawn around the global mean. **Average**: filler drawn around each movie's mean "
            "(looks like a typical user, so it is stronger). **Bandwagon**: random filler plus 5★ on the most popular movies.\n"
            "- **RDMA / WDMA** score how far a profile deviates from the crowd, weighting rarely-rated movies more; "
            "the top-scoring profiles are flagged and **removed before re-training** (detect-and-filter defence).\n"
            "- The flag budget is a trade-off: flag too few and attackers slip through (low recall), flag too many and you "
            "remove honest users (low precision).")
    if st.button("💣 Launch attack simulation"):
        st.session_state["sec"] = (run_security(ds, ds.name, sec_model, sec_kind, sec_target, sec_size, sec_fill,
                                                sec_det, sec_flag), sec_size)
    packed = st.session_state.get("sec")
    if packed is None:
        st.info("Configure the attack and press **Launch attack simulation**.")
    else:
        sweep, chosen = packed
        row = sweep.iloc[(sweep["attack_pct"] - chosen).abs().argmin()]
        st.markdown(ui.kpi_row([
            ("Fake users injected", int(row["attackers"]), f"{row['attack_pct']:.1f}% of users"),
            ("Target's predicted rating", f"{row['pred_before']:.2f} → {row['pred_attack']:.2f}",
             f"shift {row['shift_attack']:+.3f} under attack"),
            ("After detect-and-filter", f"{row['pred_defended']:.2f}", f"residual shift {row['shift_defended']:+.3f}"),
            ("Detection", f"P {row['det_precision']:.2f} · R {row['det_recall']:.2f}", "precision · recall"),
            ("Hit ratio@10", f"{row['hit_before']:.3f} → {row['hit_attack']:.3f}", f"defended {row['hit_defended']:.3f}"),
        ]), unsafe_allow_html=True)
        long = sweep.melt(id_vars="attack_pct", value_vars=["shift_attack", "shift_defended"], var_name="Scenario",
                          value_name="Prediction shift")
        long["Scenario"] = long["Scenario"].map({"shift_attack": "Under attack", "shift_defended": "With detect-and-filter"})
        s1, s2 = st.columns(2)
        fig = px.line(long, x="attack_pct", y="Prediction shift", color="Scenario", markers=True,
                      title="Prediction shift of the target vs attack size", color_discrete_sequence=["#FF5C8A", "#2EC4B6"])
        fig.update_layout(**PLOT, xaxis_title="Attack size (% of users)", legend_title_text="")
        show(fig, s1)
        det = sweep[sweep["attackers"] > 0].melt(id_vars="attack_pct", value_vars=["det_precision", "det_recall"],
                                                 var_name="Metric", value_name="Score")
        det["Metric"] = det["Metric"].map({"det_precision": "Precision", "det_recall": "Recall"})
        fig = px.line(det, x="attack_pct", y="Score", color="Metric", markers=True, title="Detection quality vs attack size",
                      color_discrete_sequence=["#7C5CFF", "#FFB347"])
        fig.update_layout(**PLOT, xaxis_title="Attack size (% of users)", legend_title_text="", yaxis_range=[0, 1.05])
        show(fig, s2)

# ------------------------------ Data ---------------------------------------- #
with tab_data:
    st.markdown("### Dataset explorer")
    density = len(ds.ratings) / (ds.n_users * ds.n_items)
    st.markdown(ui.kpi_row([
        ("Users", f"{ds.n_users:,}", ""), ("Movies", f"{ds.n_items:,}", ""),
        ("Ratings", f"{len(ds.ratings):,}", "1-5 stars"),
        ("Sparsity", f"{(1 - density) * 100:.1f}%", "of the matrix is empty"),
        ("Mean rating", f"{rm.global_mean:.2f}", ""),
    ]), unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    dist = ds.ratings["r"].value_counts().sort_index().reset_index()
    dist.columns = ["Rating", "Count"]
    fig = px.bar(dist, x="Rating", y="Count", title="Rating distribution", color_discrete_sequence=["#7C5CFF"])
    fig.update_layout(**PLOT)
    show(fig, d1)
    gc = ds.items[D.GENRES[1:]].sum().sort_values(ascending=False).reset_index()
    gc.columns = ["Genre", "Movies"]
    fig = px.bar(gc, x="Movies", y="Genre", orientation="h", title="Movies per genre", color_discrete_sequence=["#FF5C8A"])
    fig.update_layout(**PLOT, yaxis=dict(autorange="reversed"))
    show(fig, d2)
    d3, d4 = st.columns(2)
    tail = pd.DataFrame({"Rank": np.arange(1, ds.n_items + 1), "Ratings": np.sort(rm.item_count)[::-1]})
    fig = px.line(tail, x="Rank", y="Ratings", title="Long tail: ratings per movie (sorted)", log_y=True,
                  color_discrete_sequence=["#FFB347"])
    fig.update_layout(**PLOT)
    show(fig, d3)
    fig = px.histogram(pd.DataFrame({"Ratings given": rm.user_count}), x="Ratings given", nbins=40,
                       title="User activity", color_discrete_sequence=["#2EC4B6"])
    fig.update_layout(**PLOT)
    show(fig, d4)
    top_items = np.argsort(-rm.item_count)[:10]
    st.markdown("**Most-rated movies**")
    st.dataframe(pd.DataFrame({
        "Movie": [ds.label(int(i)) for i in top_items], "Ratings": rm.item_count[top_items].astype(int),
        "Average": rm.item_mean_raw[top_items].round(2)}), hide_index=True)

# ------------------------------ Syllabus map -------------------------------- #
with tab_map:
    st.markdown("### Where each syllabus topic lives in this project")
    st.dataframe(pd.DataFrame([
        ("Traditional / non-personalized recommenders", "Most Popular and Top Rated baselines (damped mean rating)",
         "models.PopularityRecommender"),
        ("Content-based: item profiles, user profiles", "IDF-weighted genre + era vectors; ridge-regression user profile",
         "models.build_item_features, models.ContentBased"),
        ("Similarity-based retrieval", "Content similarity and 'Similar movies' tab", "ContentBased.similar_items"),
        ("User-based nearest-neighbour CF", "Mean-centred Pearson, top-k neighbours, shrinkage", "models.UserKNN"),
        ("Item-based CF", "Adjusted-cosine item similarities", "models.ItemKNN"),
        ("Attacks on CF; types of attacks", "Random, Average, Bandwagon push attacks", "attacks.make_attack_profiles"),
        ("Detecting attacks; robust design", "RDMA / WDMA detection and detect-and-filter retraining", "attacks.run_attack"),
        ("Evaluation design; offline evaluation", "Per-user hold-out split, validation-based tuning", "evaluation.holdout_split"),
        ("Error and accuracy metrics", "RMSE, MAE, Precision/Recall/NDCG/HitRate@K, coverage", "evaluation.ranking_metrics"),
        ("Hybrid approaches", "Weighted (parallelized) hybrid with tuned weights", "models.Hybrid, evaluation.tune_hybrid_weights"),
    ], columns=["Syllabus topic", "In this project", "Code"]), hide_index=True)

if AUTHOR:
    st.caption(AUTHOR)
st.caption("CSE4143 Recommender Systems · Manipal University Jaipur")
