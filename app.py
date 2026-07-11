"""
=====================================================================
 MNIST Explorer — Machine Learning & Dimensionality Reduction
=====================================================================
A single-page Streamlit application that lets a user:

  1) Train one of five classic scikit-learn classifiers (LDA, QDA,
     SVM, KNN, Decision Tree) on the MNIST handwritten-digits dataset
     and inspect Train/Test accuracy plus a confusion matrix.

  2) Visualize the (high-dimensional) digit images in 2D using PCA,
     t-SNE, or UMAP.

Run with:
    streamlit run app.py

Required packages are listed in requirements.txt.
=====================================================================
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # from the `umap-learn` package

# QDA (and occasionally LDA) can emit "Variables are collinear" warnings
# on raw pixel data, because many border pixels are constant (always 0)
# across every image. This is harmless (it just means the covariance
# matrix is singular in some directions) so we silence it to keep the
# UI clean; the `reg_param` slider lets the user fix it properly instead.
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MNIST ML & Dimensionality Reduction Explorer",
    page_icon="🔢",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 3. DATA LOADING  (CACHED — this is the critical part that keeps the app
#    responsive: the expensive download/parsing step only ever runs once
#    per dataset choice, no matter how many times the user tweaks a widget)
# ---------------------------------------------------------------------------
DATASET_LABELS = {
    "MNIST (28×28, 70,000 imgs — via fetch_openml, needs internet on first run)": "mnist784",
    "Digits (8×8, 1,797 imgs — via load_digits, offline & fast)": "digits",
}


@st.cache_data(show_spinner="📥 Loading dataset (first load only — cached afterwards)...")
def load_raw_data(dataset_key: str):
    """
    Load the full raw (X, y) arrays for the chosen dataset and normalize
    pixel values to the [0, 1] range.

    Returns
    -------
    X       : ndarray, shape (n_samples, n_features)
    y       : ndarray, shape (n_samples,)   -- integer digit labels 0-9
    img_dim : int                            -- side length for reshaping
              a flat feature vector back into a square image (28 or 8)
    """
    if dataset_key == "mnist784":
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        except Exception as exc:  # no internet / OpenML unreachable
            st.error(
                "Could not download MNIST from OpenML (no internet access?). "
                "Switch the sidebar 'Dataset source' to the offline 'Digits' "
                f"option instead.\n\nDetails: {exc}"
            )
            st.stop()
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        img_dim = 28
    else:  # "digits" — sklearn's small, bundled, offline dataset
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0  # load_digits pixel range is 0-16
        y = digits.target.astype(np.int64)
        img_dim = 8

    return X, y, img_dim


@st.cache_data(show_spinner="🔀 Sampling & splitting data for training...")
def get_train_test_data(dataset_key: str, n_samples: int, test_size: float = 0.2, random_state: int = 42):
    """
    Take a stratified subsample of size `n_samples` from the full dataset,
    then split it 80/20 (or whatever `test_size` is) into train/test sets.

    Cached on (dataset_key, n_samples, test_size, random_state): changing
    the sample-size slider re-triggers only this cheap step, not the raw
    data load above.
    """
    X, y, _ = load_raw_data(dataset_key)

    if n_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=random_state
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


@st.cache_data(show_spinner="🔀 Sampling data for visualization...")
def get_viz_subset(dataset_key: str, n_samples: int, random_state: int = 42):
    """
    Grab a random stratified subset of the data purely for the 2D
    embedding plot (kept small since t-SNE/UMAP scale poorly with n).
    """
    X, y, _ = load_raw_data(dataset_key)
    if n_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=random_state
        )
    return X, y


# ---------------------------------------------------------------------------
# 4. MODEL FACTORY + TRAINING  (CACHED)
# ---------------------------------------------------------------------------
MODEL_OPTIONS = ["LDA", "QDA", "SVM", "KNN", "Decision Tree"]


def build_model(model_name: str, params: dict):
    """Instantiate an *untrained* scikit-learn estimator from a name + hyperparameter dict."""
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver=params["solver"])
    elif model_name == "QDA":
        return QuadraticDiscriminantAnalysis(reg_param=params["reg_param"])
    elif model_name == "SVM":
        return SVC(kernel=params["kernel"], C=params["C"])
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")


@st.cache_data(show_spinner="🤖 Training model — this can take a few seconds...")
def train_and_evaluate(model_name: str, params: dict, X_train, y_train, X_test, y_test):
    """
    Fit `model_name` on (X_train, y_train) and evaluate on both splits.

    Cached on all arguments (including the data arrays), so re-selecting a
    previously-seen (model, hyperparameters, dataset) combination is instant.

    Returns
    -------
    train_acc, test_acc : float, float
    cm                   : ndarray — confusion matrix on the TEST set
    fit_time             : float  — seconds spent in model.fit()
    """
    model = build_model(model_name, params)

    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds, labels=sorted(np.unique(y_train)))

    return train_acc, test_acc, cm, fit_time


# ---------------------------------------------------------------------------
# 5. DIMENSIONALITY REDUCTION  (CACHED)
# ---------------------------------------------------------------------------
DR_OPTIONS = ["PCA", "t-SNE", "UMAP"]


@st.cache_data(show_spinner="📉 Computing 2D embedding — this can take a little while...")
def compute_embedding(technique: str, X_subset, y_subset, params: dict):
    """
    Project `X_subset` down to 2 dimensions using the chosen technique.
    `y_subset` is accepted only so the cache key changes if the sampled
    subset changes; it is not otherwise used for the (unsupervised) fit.
    """
    if technique == "PCA":
        reducer = PCA(n_components=2, random_state=42)
    elif technique == "t-SNE":
        reducer = TSNE(
            n_components=2,
            perplexity=params["perplexity"],
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
    elif technique == "UMAP":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown technique: {technique}")

    embedding = reducer.fit_transform(X_subset)
    return embedding


# ---------------------------------------------------------------------------
# 6. PLOTTING HELPERS
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, labels):
    """Render a confusion matrix as an annotated seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    return fig


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, technique_name: str):
    """Render a 2D scatter plot of an embedding, colored by true digit label."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.85,
        linewidth=0,
    )
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Digit",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.add_artist(legend)
    ax.set_title(f"{technique_name} projection of digit images")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_sample_digits(X: np.ndarray, y: np.ndarray, img_dim: int, n: int = 10):
    """Show a small strip of sample digit images with their true labels."""
    fig, axes = plt.subplots(1, n, figsize=(1.1 * n, 1.4))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(img_dim, img_dim), cmap="gray_r")
        ax.set_title(str(y[i]), fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. SIDEBAR — USER CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

# --- Dataset source -----------------------------------------------------
st.sidebar.subheader("Dataset")
dataset_label = st.sidebar.radio(
    "Source",
    list(DATASET_LABELS.keys()),
    index=0,
    help=(
        "Full MNIST is downloaded once via OpenML and cached (needs internet "
        "the first time). 'Digits' is a small dataset bundled with "
        "scikit-learn — fast and fully offline."
    ),
)
dataset_key = DATASET_LABELS[dataset_label]

# We need to know how many rows are available before drawing the
# sample-size slider below, so load (cached ⇒ cheap after first call).
_X_full, _y_full, IMG_DIM = load_raw_data(dataset_key)
N_FULL = len(_X_full)

n_samples = st.sidebar.slider(
    "Number of samples to use (train + test)",
    min_value=min(500, N_FULL),
    max_value=N_FULL,
    value=min(5000, N_FULL),
    step=500 if N_FULL > 2000 else 100,
    help="MNIST has 70,000 images total. Slower models (SVM, KNN) run "
         "much faster on a smaller subsample.",
)

# --- Model selection ------------------------------------------------------
st.sidebar.subheader("🤖 Classification Model")
model_name = st.sidebar.selectbox("Choose a model", MODEL_OPTIONS)

# Per-model hyperparameter widgets — only the relevant ones are shown.
params = {}
if model_name == "LDA":
    params["solver"] = st.sidebar.selectbox("Solver", ["svd", "lsqr", "eigen"])
elif model_name == "QDA":
    params["reg_param"] = st.sidebar.slider(
        "Regularization (reg_param)", 0.01, 1.0, 0.1, 0.01,
        help="Raw pixel features are collinear (many border pixels are "
             "always 0); a small regularization value stabilizes QDA.",
    )
elif model_name == "SVM":
    params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    params["C"] = st.sidebar.slider("C (regularization strength)", 0.1, 10.0, 1.0, 0.1)
elif model_name == "KNN":
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 20, 5)
elif model_name == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 30, 10)

if model_name in ("SVM", "KNN") and n_samples > 6000:
    st.sidebar.warning(
        "⚠️ SVM/KNN can be slow with many samples. Consider lowering "
        "'Number of samples' above if training feels sluggish."
    )

# --- Dimensionality reduction ---------------------------------------------
st.sidebar.subheader("📊 Dimensionality Reduction")
dr_technique = st.sidebar.selectbox("Choose a visualization technique", DR_OPTIONS)

dr_params = {}
viz_default = 1000
if dr_technique == "PCA":
    viz_default = min(3000, N_FULL)  # PCA is fast — allow more points
elif dr_technique == "t-SNE":
    dr_params["perplexity"] = st.sidebar.slider("Perplexity", 5, 50, 30)
    viz_default = min(1000, N_FULL)
elif dr_technique == "UMAP":
    dr_params["n_neighbors"] = st.sidebar.slider("n_neighbors", 5, 50, 15)
    dr_params["min_dist"] = st.sidebar.slider("min_dist", 0.0, 0.99, 0.1)
    viz_default = min(1000, N_FULL)

viz_samples = st.sidebar.slider(
    "Samples for visualization",
    min_value=min(200, N_FULL),
    max_value=min(3000, N_FULL),
    value=viz_default,
    step=100,
    help="t-SNE and UMAP are computationally expensive — a smaller subset "
         "keeps the plot fast to generate.",
)


# ---------------------------------------------------------------------------
# 8. MAIN PAGE
# ---------------------------------------------------------------------------
st.title("🔢 MNIST: Machine Learning & Dimensionality Reduction Explorer")
st.markdown(
    "Train classic classifiers and explore 2D visualizations of the "
    "**MNIST handwritten digits** dataset — everything is controlled from "
    "the sidebar on the left."
)

# --- Load train/test split (cached) ---------------------------------------
X_train, X_test, y_train, y_test = get_train_test_data(dataset_key, n_samples)

st.info(
    f"Using **{len(X_train):,}** training and **{len(X_test):,}** testing "
    f"samples (an 80/20 split of a {n_samples:,}-sample subset of the "
    f"{N_FULL:,}-image **{dataset_label.split(' (')[0]}** dataset)."
)

with st.expander("👀 Peek at some sample digit images"):
    st.pyplot(plot_sample_digits(X_train, y_train, IMG_DIM))

with st.expander("📊 Class distribution in the training set"):
    counts = pd.Series(y_train).value_counts().sort_index()
    counts.index.name = "Digit"
    st.bar_chart(counts)

st.divider()

# ---------------- Section A: Classification --------------------------------
st.header(f"🤖 Model: {model_name}")

train_acc, test_acc, cm, fit_time = train_and_evaluate(
    model_name, params, X_train, y_train, X_test, y_test
)

col1, col2, col3 = st.columns(3)
col1.metric("Training Accuracy", f"{train_acc:.2%}")
col2.metric("Testing Accuracy", f"{test_acc:.2%}")
col3.metric("Training Time", f"{fit_time:.2f} s")

if train_acc - test_acc > 0.15:
    st.caption(
        "ℹ️ Training accuracy is notably higher than testing accuracy — "
        "this model may be overfitting on this sample size/hyperparameter combo."
    )

st.subheader("Confusion Matrix (Test Set)")
st.pyplot(plot_confusion_matrix(cm, labels=sorted(np.unique(y_train))))

st.divider()

# ---------------- Section B: Dimensionality Reduction -----------------------
st.header(f"📊 Visualization: {dr_technique}")

X_viz, y_viz = get_viz_subset(dataset_key, viz_samples)
embedding = compute_embedding(dr_technique, X_viz, y_viz, dr_params)

st.pyplot(plot_embedding(embedding, y_viz, dr_technique))
st.caption(
    f"2D {dr_technique} projection of {len(X_viz):,} randomly sampled digit "
    "images, colored by their true label. Tight, well-separated clusters "
    "indicate the digits are easy to distinguish in this embedding."
)
