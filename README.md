# CineMatch: a movie recommender system (CSE4143)

A from-scratch recommender built on **MovieLens 100K** (943 users, 1,682 movies, 100,000 ratings) with an interactive Streamlit app.
Only numpy/pandas do the maths: no recommender library, so every formula is visible in the code.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

* On first run the app downloads MovieLens 100K into `data/ml-100k/`. **Do this once with internet, before the evaluation.**
  No internet at the venue? Download <https://files.grouplens.org/datasets/movielens/ml-100k.zip> by hand and unzip it
  into `data/` so that `data/ml-100k/u.data` exists.
* If the dataset can't be found, the app falls back to *synthetic* data and shows an orange warning banner at the top.
  Don't present that as real results.
* Optional: set `AUTHOR = "Your name · roll no."` at the top of `app.py` to show it in the footer.
* Command-line evaluation: `python run_evaluation.py` (writes CSVs to `results/`).

## What's inside

| Algorithm | Idea | Code |
|---|---|---|
| Most Popular / Top Rated | non-personalized baselines (count, damped mean rating) | `models.PopularityRecommender` |
| Content-Based | IDF-weighted genre + era item profiles; a ridge-regression user profile learned per user | `models.ContentBased` |
| User-based CF | mean-centred Pearson similarity, top-k neighbours, significance shrinkage | `models.UserKNN` |
| Item-based CF | adjusted-cosine item similarity, top-k neighbours | `models.ItemKNN` |
| Hybrid | weighted blend of the three, weights tuned on a validation split | `models.Hybrid`, `evaluation.tune_hybrid_weights` |

**Evaluation** (`evaluation.py`): per-user 80/20 hold-out, RMSE and MAE for rating prediction, Precision/Recall/NDCG/HitRate@K
and catalogue coverage for top-N, a neighbourhood-size sweep, and Pearson vs cosine.

**Security lab** (`attacks.py`): Random, Average and Bandwagon push attacks; RDMA/WDMA detectors; a detect-and-filter defence
measured against a fair baseline (the same clean system with the same honest users removed).

## App tabs

1. **Recommend**: pick any user, choose an algorithm, see explained recommendations ("Because you liked X"), the user's learned genre taste profile, or compare algorithms side by side. Hybrid weights are live sliders in the sidebar.
2. **New user**: cold start. Pick movies you loved and disliked; the models retrain with you included.
3. **Similar movies**: item-CF neighbours vs content neighbours for any movie.
4. **Model Lab**: runs the full evaluation protocol and plots the results.
5. **Security Lab**: launch an attack, watch the target's predicted rating rise, then detect and filter the fake profiles.
6. **Data**: sparsity, rating distribution, long tail, genres.
7. **Syllabus map**: which lecture topic each part of the project covers.

## Design notes worth knowing for the viva

* **Rating prediction and top-N ranking are scored differently for the CF models.** Predicted ratings (used for RMSE) are
  mean-centred weighted averages. For ranking, ordering by predicted rating promotes obscure movies with one enthusiastic
  neighbour, so the CF models rank by similarity-weighted neighbour support (how many similar users or items liked it).
* **No test leakage:** hybrid weights are tuned on a validation split carved out of the training data only.
* **Candidate filter:** recommendations only include movies with at least N ratings (sidebar slider, default 20). Evaluation uses the same filter.
* **Known limitation:** MovieLens 100K has no rich item metadata (only genres and release year), so content-based filtering is
  expected to be the weakest model. That is the classic argument for the hybrid.

## Files

```
app.py              Streamlit app
run_evaluation.py   CLI evaluation
recsys/
  data.py           MovieLens loader (+ synthetic fallback)
  models.py         all recommenders
  evaluation.py     split, metrics, experiments
  attacks.py        attacks, detection, defence
  ui.py             HTML/CSS helpers for the app
```
