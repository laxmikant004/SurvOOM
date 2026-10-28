"""
SurvOOM -- GRU survival model for proactive, per-container OOM-kill prediction.

This is the paper's canonical model file. One architecture only: a GRU that
reads a sliding LOOKBACK_WINDOW-row window of live cgroup/PSI metrics and
outputs a risk score, trained with a Cox proportional-hazards (ranking)
loss. (An earlier draft also had a "streaming" GRU that read a container's
full history and emitted a risk score at every row. We decided not to use
it for the paper -- the sliding-window model is simpler, matches how
`realtime_csv_oom_predictor.py` already deploys, and the streaming variant
added training cost without changing the paper's claims. It has been
removed from this file; if you ever want it back, it lived in
`gru_oom_predictor_only_csv_container_level_threshold_2.py` and `_3.py`.)

Train + evaluate every tier on the normal train/validation/test split:
    python3 gru_oom_predictor.py

Only run specific tiers:
    python3 gru_oom_predictor.py --tiers csv-only no-distance-strict

Unseen-workload generalization test (train WITHOUT 'spike', test on every
spike container):
    python3 gru_oom_predictor.py --holdout-workload spike --tiers csv-only

Hold out each workload in turn (produces one combined lowo_summary.csv):
    python3 gru_oom_predictor.py --holdout-workload all --tiers csv-only

Alert on ONE container's CSV with an already-trained model bundle (the
live / deployment path -- no meta.json, no retraining):
    python3 gru_oom_predictor.py predict --model-dir paper_run_<ts>/saved_models/csv-only --csv ep_123.csv

What this file does, per feature tier:
  1. Trains the GRU (Cox-PH survival loss) with early stopping on the
     validation Cox-PH loss, so the epoch count is chosen by the data,
     not hand-tuned once and hard-coded.
  2. Per-CONTAINER (episode-level) yes/no accuracy: for each test
     container, "will it die" (ANY window crossed the alert threshold),
     compared against ground truth.
  3. For each correctly-caught death, exactly how many seconds BEFORE
     the kill the first alert fired (the lead-time evidence).
  4. PER-CONTAINER ALERT + CONFIDENCE: every test container gets an
     alert level (CRITICAL / WARNING / OK), a calibrated probability of
     being OOM-killed, and a confidence level (High / Medium / Low). The
     probability comes from a Platt-scaling calibrator fitted on the
     TRAIN containers' peak risk scores (never on test data).
  5. C-index (peak-risk + landmark at 60/120/180 s) + Kaplan-Meier.
  6. A simple rule baseline: "alert when usage_ratio >= 0.99" (the
     obvious non-ML alternative a reviewer will ask about).
  7. Early-alarm checks: share of alarms in the first window, lead time
     as a fraction of the episode, median first-alarm time.

Splits: test = 20% of containers (stratified by workload x died); the
remaining 80% is split 85% train / 15% VALIDATION (also stratified). The
alert threshold and the confidence calibrator are BOTH chosen on the
validation containers, at CONTAINER level (peak risk), so they match the
"alarm if any window crosses" rule that is actually reported. Early
stopping during training is also decided on the validation set, using a
separate signal (Cox-PH loss, not peak risk) computed each epoch.

Tiers (what the model is allowed to see):
  full                 -- everything in the CSV plus meta.json fields.
  csv-only             -- every live CSV signal; EXCLUDES meta.json fields
                          (headroom_pct, contention_n) and memory_max (a
                          per-container constant that leaks the
                          experiment's headroom setting). THE DEPLOYABLE
                          TIER -- this is what `predict` mode and
                          `realtime_csv_oom_predictor.py` actually use.
  no-distance          -- drops raw byte-magnitude columns (memory_current,
                          memory_max, swap, anon/file/slab, usage_ratio).
  no-distance-strict   -- no-distance, and also no meta.json fields.
  psi-only             -- only the four psi_* pressure-stall columns.

Cumulative cgroup counters (pgfault, pgmajfault, pgrefill,
workingset_refault, events_low/high/max) are converted to per-second rates
(USE_RATES=True) before scaling, so a counter's raw magnitude can't leak
how long the container has already been running -- the model has to react
to the current RATE of trouble, not to "this counter is large so the
container must be old and close to the end of its episode".

Output, per tier, under <run_dir>/results/<tier>/:
  container_alerts.csv             -- ONE ROW PER CONTAINER: alert level, P(OOM-kill),
                                      confidence, alert message, actual outcome, lead time
  episode_level_accuracy.csv       -- per-container yes/no + lead-time table
  episode_level_summary.csv        -- accuracy/precision/recall/F1 + lead-time + Brier
  accuracy_by_confidence_level.csv -- is "High confidence" actually more accurate?
  per_workload_summary.csv         -- same metrics, broken out by workload type
  c_index_variants.csv             -- peak / landmark / first-window C-index
  confusion_matrix.png             -- per-CONTAINER confusion matrix
  container_risk_distribution.png  -- max risk score per container, survived vs died
  container_roc.png                -- container-level ROC curve
  confidence_calibration.png       -- reliability diagram: predicted P(OOM) vs observed
  alert_levels.png                 -- containers per alert level, split by actual outcome
  per_workload_breakdown.png       -- accuracy + lead-time, one bar per workload type
  example_risk_trajectories.png    -- risk-over-time for a TP/FP/TN example container
  lead_time_distribution.png       -- histogram of seconds-of-warning, caught deaths only
  kaplan_meier_by_risk.png         -- survival curves by risk known at the 60 s landmark
Output, per tier, under <run_dir>/saved_models/<tier>/:
  gru_survival.pt                  -- trained weights (best validation epoch, not last)
  bundle.joblib                    -- everything `predict` mode needs: scaler, threshold,
                                      calibrator, feature list, lookback, arch metadata
  training_curve.csv / .png        -- train vs. validation Cox-PH loss per epoch, with
                                      the early-stopping epoch marked
Output under <run_dir>/results/:
  baseline-usage99/                -- same container tables, for the usage-ratio rule
  tier_comparison.png / .csv       -- all tiers + baseline, side by side
  lead_time_vs_accuracy.png
  lead_time_boxplot_by_tier.png
"""
import os
import sys
import json
import glob
import copy
import random
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

try:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAVE_LIFELINES = True
except ImportError:
    HAVE_LIFELINES = False

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = "./data"
LOOKBACK_WINDOW = 30            # rows per sliding window the GRU sees
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4              # L2 regularization; cheap insurance against overfitting
GRAD_CLIP = 1.0                  # clip gradient norm every step, for training stability

# Early stopping (replaces the old hard-coded EPOCHS=10 "verified by eye"
# cutoff): train up to MAX_EPOCHS, but stop as soon as validation Cox-PH
# loss hasn't improved by MIN_DELTA for PATIENCE epochs in a row, and keep
# the weights from whichever epoch had the best validation loss -- not
# whatever epoch training happened to stop on.
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
# Batch size used ONLY for computing the per-epoch validation loss (early
# stopping). Large enough that the Cox-PH ranking loss within one batch is
# a stable estimate, small enough to fit a GRU forward pass on a modest
# GPU -- a window is (LOOKBACK_WINDOW, num_features) floats, so a batch of
# a few thousand windows is a few hundred MB even with cuDNN workspace,
# nowhere near "the whole validation set in one GRU call" (which is what
# the previous version did, and which asked a 6 GB GPU for ~38 GB).
VAL_EVAL_BATCH_SIZE = 1024

SEED = 42
ZERO_TIME_PREFIX = "0001-01-01"

TIERS = ['full', 'csv-only', 'no-distance', 'no-distance-strict', 'psi-only']

# Cumulative cgroup counters -> per-second rates (see module docstring).
USE_RATES = True
COUNTER_COLS = ['pgfault', 'pgmajfault', 'pgrefill', 'workingset_refault',
                'events_low', 'events_high', 'events_max']
META_COLS = ['headroom_pct', 'contention_n']     # come from meta.json, not the live CSV

# Alert + confidence settings
CONF_HIGH = 0.80        # confidence >= 0.80 -> "High"
CONF_MEDIUM = 0.60      # 0.60 - 0.80 -> "Medium", below -> "Low"
CRITICAL_P = 0.80       # alert fired AND P(OOM) >= 0.80 -> CRITICAL, else WARNING

# Rule baseline
BASELINE_USAGE_RATIO = 0.99

# Container-level alert threshold. It is chosen on a VALIDATION set of
# containers held out from training (the model has never seen them), using
# each container's peak risk -- which is exactly the "any window crosses"
# alert rule. "precision": lowest threshold whose validation precision is
# >= CONTAINER_MIN_PRECISION (max recall at that precision); falls back to
# best F1 if no threshold reaches it. "f1": best F1.
VAL_FRAC = 0.15
THRESHOLD_MODE = "precision"
CONTAINER_MIN_PRECISION = 0.90

# Landmark C-index: ranking quality using only what was seen up to time L
LANDMARKS_S = [60, 120, 180]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_features_for_tier(df_columns, tier):
    excluded_cols = ['t_wall', 't_rel', 'events_oom', 'events_oom_kill']
    all_features = [c for c in df_columns if c not in excluded_cols]
    raw_bytes = ['memory_current', 'memory_max', 'memory_swap_current', 'anon',
                 'file', 'kernel_stack', 'slab', 'usage_ratio']
    if tier == 'full':
        return all_features
    elif tier == 'csv-only':
        # Everything observable live from the container's cgroup. No meta.json
        # values, and no memory_max (a per-container constant that encodes the
        # experiment's headroom setting; usage_ratio carries the same info live).
        return [c for c in all_features if c not in META_COLS and c != 'memory_max']
    elif tier == 'no-distance':
        return [c for c in all_features if c not in raw_bytes]
    elif tier == 'no-distance-strict':
        return [c for c in all_features if c not in raw_bytes and c not in META_COLS]
    elif tier == 'psi-only':
        return [c for c in all_features if c.startswith('psi_')]
    else:
        raise ValueError(f"Unknown tier: {tier}")


def prepare_df(csv_path, ep=None, use_rates=USE_RATES):
    """Read one episode CSV the same way everywhere (training, test,
    baseline and live prediction). `ep` carries the meta.json values;
    pass None in live use -- the meta columns are then NaN, which is fine
    for any tier that doesn't use them (csv-only, no-distance-strict,
    psi-only)."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.fillna(0)
    if use_rates and "t_rel" in df.columns:
        dt = df["t_rel"].diff()
        dt = dt.where(dt > 0, np.nan)
        for c in COUNTER_COLS:
            if c in df.columns:
                rate = df[c].diff() / dt
                df[c] = rate.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    df["headroom_pct"] = ep["headroom_pct"] if ep is not None else np.nan
    df["contention_n"] = ep["contention_n"] if ep is not None else np.nan
    return df


# ==========================================
# 2. Episode discovery
# ==========================================
def discover_episodes(data_dir):
    """Find every ep_*.csv / ep_*.meta.json pair under data_dir, recursing
    into subfolders (so a data/part_1 .. data/part_N layout works), and
    read ground truth (did it actually get OOM-killed?) from the meta."""
    meta_paths = glob.glob(os.path.join(data_dir, "**", "ep_*.meta.json"), recursive=True)
    meta_paths = [p for p in meta_paths if "smoke" not in p]
    episodes = []
    for mp in meta_paths:
        csv_path = mp.replace(".meta.json", ".csv")
        if not os.path.exists(csv_path):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        flag = meta.get("docker_oom_killed_flag")
        exit_code = meta.get("exit_code")
        finished_real = not str(meta.get("finished_at", "")).startswith(ZERO_TIME_PREFIX)
        died = (flag == "true") or (exit_code == "137" and finished_real)
        episodes.append({
            "episode_id": meta.get("episode_id", os.path.basename(mp).replace(".meta.json", "")),
            "csv_path": csv_path,
            "workload": meta.get("workload"),
            "died": died,
            "headroom_pct": meta.get("headroom_pct"),
            "contention_n": meta.get("contention_n"),
        })
    return episodes


def make_episode_split(episodes, test_frac=0.2, seed=SEED):
    """Stratified split by (workload, died), so every workload type and
    both outcomes are represented in both halves even with few episodes."""
    rng = random.Random(seed)
    groups = {}
    for ep in episodes:
        groups.setdefault((ep["workload"], ep["died"]), []).append(ep)
    train_ids, test_ids = set(), set()
    for group in groups.values():
        ids = [e["episode_id"] for e in group]
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_frac))
        test_ids.update(ids[:n_test])
        train_ids.update(ids[n_test:])
    return train_ids, test_ids


def make_validation_split(episodes, train_ids, val_frac=VAL_FRAC, seed=SEED + 1):
    """Carve a validation set out of the TRAIN containers, stratified the
    same way (workload x died). The test set is untouched, so test results
    stay comparable with earlier runs. Validation containers are never
    trained on; they set the alert threshold, the confidence calibrator,
    AND the early-stopping epoch."""
    train_eps = [e for e in episodes if e["episode_id"] in train_ids]
    fit_ids, val_ids = make_episode_split(train_eps, test_frac=val_frac, seed=seed)
    return fit_ids, val_ids


def make_holdout_split(episodes, workload):
    """Unseen-workload generalization test: EVERY container of `workload`
    (died and survived) goes to test; the model, the alert threshold and
    the confidence calibrator only ever see the other workloads."""
    test_ids = {e["episode_id"] for e in episodes if e["workload"] == workload}
    train_ids = {e["episode_id"] for e in episodes if e["workload"] != workload}
    return train_ids, test_ids


# ==========================================
# 3. Dataset
# ==========================================
class OOMSurvivalDataset(Dataset):
    """One item = one LOOKBACK_WINDOW-row sliding window from one episode,
    paired with (time remaining in the episode after this window, whether
    the episode eventually died). Windows from the same episode share the
    episode's scaler-fitted features; `episodes` holds one entry per
    episode and `window_indices` maps a flat window index back to
    (episode_idx, window_start_row)."""

    def __init__(self, episodes, episode_ids, tier, scaler=None, lookback=LOOKBACK_WINDOW):
        self.lookback = lookback
        self.tier = tier
        self.episodes = []
        self.window_indices = []
        self.episode_ids_in_order = []

        selected = [e for e in episodes if e["episode_id"] in episode_ids]
        if not selected:
            raise ValueError("No episodes matched the given episode_ids.")

        sample_df = prepare_df(selected[0]["csv_path"], selected[0])
        self.features_list = get_features_for_tier(sample_df.columns, tier)
        print(f"[{tier}] Using {len(self.features_list)} features.")

        if scaler is None:
            # Scaling statistics (mean / std per feature) from ALL training containers,
            # one container at a time (partial_fit) so memory stays low. Validation and
            # test reuse this scaler (else-branch), so no test data is ever used here.
            self.scaler = StandardScaler()
            n_fit = 0
            for ep in selected:
                df = prepare_df(ep["csv_path"], ep)
                if all(c in df.columns for c in self.features_list):
                    self.scaler.partial_fit(df[self.features_list].values)
                    n_fit += 1
            print(f"[{tier}] Scaler fitted on {n_fit} training containers.")
        else:
            self.scaler = scaler

        skipped = 0
        for ep in selected:
            df = prepare_df(ep["csv_path"], ep)
            if not all(c in df.columns for c in self.features_list) or len(df) <= self.lookback:
                skipped += 1
                continue
            if "t_rel" not in df.columns:
                skipped += 1
                continue

            scaled_feat = self.scaler.transform(df[self.features_list].values).astype(np.float32)
            t_rel = df["t_rel"].values.astype(np.float64)
            is_oom = 1 if ep["died"] else 0

            self.episodes.append((scaled_feat, is_oom, t_rel, ep["workload"]))
            self.episode_ids_in_order.append(ep["episode_id"])
            ep_idx = len(self.episodes) - 1
            for i in range(len(scaled_feat) - self.lookback):
                self.window_indices.append((ep_idx, i))

        if skipped:
            print(f"[{tier}] Skipped {skipped} episode(s) (schema mismatch, too short, or no t_rel).")
        print(f"[{tier}] {len(self.episodes)} episodes, {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        ep_idx, start_step = self.window_indices[idx]
        scaled_feat, is_oom, t_rel, _workload = self.episodes[ep_idx]
        window = scaled_feat[start_step: start_step + self.lookback]
        window_end_t = t_rel[start_step + self.lookback - 1]
        time_remaining_s = float(t_rel[-1] - window_end_t)
        return (torch.tensor(window),
                torch.tensor(time_remaining_s, dtype=torch.float32),
                torch.tensor(is_oom, dtype=torch.float32))


# ==========================================
# 4. GRU model
# ==========================================
class GRUSurvival(nn.Module):
    """Reads one LOOKBACK_WINDOW-row window and outputs a single risk
    score from the GRU's final hidden state. Higher score = higher
    predicted hazard of an OOM-kill. The score is NOT a probability -- it
    is a Cox-PH ranking score, meaningful only relative to other windows'
    scores (see ConfidenceCalibrator below for how we turn it into a
    calibrated P(OOM-kill))."""

    def __init__(self, num_inputs, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=num_inputs, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.linear(last_hidden)


# ==========================================
# 5. Cox PH loss
# ==========================================
def cox_ph_loss(risk_scores, time, events):
    """Cox partial-likelihood ranking loss: pushes risk scores to rank
    windows correctly by time-to-event, among windows where the event
    (OOM-kill) is actually observed (events==1). Windows from survived
    episodes are censored observations -- they contribute to the ranking
    denominator (as "still at risk") but never directly to the numerator."""
    time, events, risk_scores = time.view(-1), events.view(-1), risk_scores.view(-1)
    idx = torch.argsort(time, descending=True)
    time_sorted, events_sorted, risk_scores_sorted = time[idx], events[idx], risk_scores[idx]
    max_risk = torch.max(risk_scores_sorted)
    hazard_ratio = torch.exp(risk_scores_sorted - max_risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8) + max_risk
    uncensored_likelihood = risk_scores_sorted - log_risk
    censored_likelihood = uncensored_likelihood * events_sorted
    return -torch.sum(censored_likelihood) / (torch.sum(events_sorted) + 1e-8)


# ==========================================
# 6. Training (with early stopping)
# ==========================================
def _run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains (backward + step + grad clip) when
    `optimizer` is given, otherwise just evaluates under no_grad. Returns
    the mean per-batch Cox-PH loss."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_x, batch_t, batch_e in loader:
            batch_x, batch_t, batch_e = batch_x.to(device), batch_t.to(device), batch_e.to(device)
            if train_mode:
                optimizer.zero_grad()
            loss = cox_ph_loss(model(batch_x), batch_t, batch_e)
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_model(tier, episodes, train_ids, val_ids, test_ids, run_dir):
    """Build the three datasets for this tier, train the GRU with early
    stopping on validation Cox-PH loss, and save the best-epoch weights
    plus a training-curve plot/CSV. Returns the trained model (best
    validation epoch, not necessarily the last one trained) and the three
    datasets, for the caller to score and evaluate."""
    train_dataset = OOMSurvivalDataset(episodes, train_ids, tier, lookback=LOOKBACK_WINDOW)
    val_dataset = OOMSurvivalDataset(episodes, val_ids, tier, scaler=train_dataset.scaler, lookback=LOOKBACK_WINDOW)
    test_dataset = OOMSurvivalDataset(episodes, test_ids, tier, scaler=train_dataset.scaler, lookback=LOOKBACK_WINDOW)

    dataset_dir = os.path.join(run_dir, 'saved_datasets', tier)
    os.makedirs(dataset_dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(dataset_dir, 'train.pt'))
    torch.save(val_dataset, os.path.join(dataset_dir, 'val.pt'))
    torch.save(test_dataset, os.path.join(dataset_dir, 'test.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Fixed, GPU-memory-safe batch size for validation loss (see
    # VAL_EVAL_BATCH_SIZE above). The reported val_loss is the mean over
    # these batches -- a stable-enough signal for early stopping without
    # trying to run the whole validation set through the GRU in one call.
    val_loader = DataLoader(val_dataset, batch_size=VAL_EVAL_BATCH_SIZE, shuffle=False)

    model = GRUSurvival(num_inputs=len(train_dataset.features_list), hidden_size=128, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"[{tier}] Training on {device} (max {MAX_EPOCHS} epochs, "
          f"early stopping: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA})...")

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{tier}] Epoch {epoch}/{MAX_EPOCHS} - train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[{tier}] Early stopping at epoch {epoch} "
                      f"(no val-loss improvement for {EARLY_STOP_PATIENCE} epochs). "
                      f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.")
                break
    else:
        print(f"[{tier}] Reached MAX_EPOCHS={MAX_EPOCHS} without triggering early stopping. "
              f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.")

    if best_state is not None:
        model.load_state_dict(best_state)   # always evaluate/deploy the best validation epoch

    model_dir = os.path.join(run_dir, 'saved_models', tier)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'gru_survival.pt'))
    _save_training_curve(history, best_epoch, model_dir, tier)

    return model, train_dataset, val_dataset, test_dataset, device


def _save_training_curve(history, best_epoch, model_dir, tier):
    """CSV + PNG of train/val Cox-PH loss per epoch, with the chosen
    (early-stopped) epoch marked -- useful evidence for the paper that
    training actually converged rather than being cut off arbitrarily."""
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(model_dir, "training_curve.csv"), index=False)
    plt.figure(figsize=(7, 4.5))
    plt.plot(df.epoch, df.train_loss, label="train loss", color="#2E5395")
    plt.plot(df.epoch, df.val_loss, label="validation loss", color="#C00000")
    plt.axvline(best_epoch, color="black", linestyle="--", lw=1,
                label=f"best epoch ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Cox-PH loss")
    plt.title(f"Training Curve -- {tier}"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(model_dir, "training_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 7. Scoring + threshold + confidence calibration
# ==========================================
def score_all_windows(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch_x, _, _ in loader:
            scores.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(scores).flatten()


def pick_container_threshold(peak_risk, died, mode=THRESHOLD_MODE, min_precision=CONTAINER_MIN_PRECISION):
    """Alert threshold chosen per CONTAINER, on validation containers.

    A container alarms iff some window's risk >= threshold, i.e. iff its
    PEAK risk >= threshold -- so choosing the threshold on peak risks
    optimises exactly the decision we report. (An earlier version chose it
    on individual windows: with ~1,000 windows per container, one noisy
    window was enough to flag a survivor.)

    Returns (threshold, rule_used)."""
    if len(np.unique(died)) < 2:
        return float(np.max(peak_risk)) + 1e-6, "degenerate (one class in validation)"
    precision, recall, thresholds = precision_recall_curve(died, peak_risk)
    p, r = precision[:-1], recall[:-1]
    if mode == "precision":
        ok = np.where(p >= min_precision)[0]
        if len(ok):
            return float(thresholds[ok[0]]), f"lowest with val precision >= {min_precision:.2f}"
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(thresholds[int(np.argmax(f1))]), "best val F1"


def container_peak_risk(dataset, window_risk):
    """One number per container: its highest window risk score, plus its
    true outcome. Same ordering as dataset.episodes."""
    peak = np.full(len(dataset.episodes), -np.inf)
    for w, (ep_idx, _s) in enumerate(dataset.window_indices):
        if window_risk[w] > peak[ep_idx]:
            peak[ep_idx] = window_risk[w]
    died = np.array([ep[1] for ep in dataset.episodes])
    keep = np.isfinite(peak)
    return peak[keep], died[keep]


class ConfidenceCalibrator:
    """Platt scaling: maps a container's peak risk score to a calibrated
    probability P(OOM-kill). Fitted on TRAIN containers only. A model
    score of 2.3 means nothing to an operator; "87% chance of OOM-kill"
    does."""

    def __init__(self, coef=None, intercept=None, constant_p=None):
        self.coef, self.intercept, self.constant_p = coef, intercept, constant_p

    @classmethod
    def fit(cls, peak_risk, died):
        if len(np.unique(died)) < 2:      # degenerate: only one class in train
            return cls(constant_p=float(np.mean(died)))
        lr = LogisticRegression(C=1.0).fit(peak_risk.reshape(-1, 1), died)
        return cls(coef=float(lr.coef_[0][0]), intercept=float(lr.intercept_[0]))

    def p_oom(self, risk):
        risk = np.asarray(risk, dtype=np.float64)
        if self.constant_p is not None:
            return np.full_like(risk, self.constant_p)
        return 1.0 / (1.0 + np.exp(-(self.coef * risk + self.intercept)))

    def to_dict(self):
        return {"coef": self.coef, "intercept": self.intercept, "constant_p": self.constant_p}


def make_alert(p_oom, alerted):
    """Turn a calibrated probability + the threshold decision into the
    operator-facing alert.

    - alert_level: CRITICAL (alert fired and P(OOM) >= CRITICAL_P),
      WARNING (alert fired, lower probability), OK (no alert).
    - confidence: how sure we are of the call that was made --
      P(OOM) for an alert, 1 - P(OOM) for an OK.
    """
    if alerted:
        level = "CRITICAL" if p_oom >= CRITICAL_P else "WARNING"
        confidence = p_oom
    else:
        level = "OK"
        confidence = 1.0 - p_oom
    conf_level = "High" if confidence >= CONF_HIGH else ("Medium" if confidence >= CONF_MEDIUM else "Low")
    return level, float(confidence), conf_level


def alert_message(episode_id, workload, level, p_oom, confidence, conf_level, first_alert_time_s):
    who = f"{episode_id} ({workload})" if workload else str(episode_id)
    if level == "OK":
        return (f"OK: {who} -- no OOM-kill expected "
                f"(P(OOM)={p_oom:.0%}, confidence {confidence:.0%}, {conf_level})")
    return (f"{level}: {who} -- OOM-kill predicted, alert raised at t={first_alert_time_s:.1f}s "
            f"(P(OOM)={p_oom:.0%}, confidence {confidence:.0%}, {conf_level})")


def summarize_predictions(df):
    """Accuracy/precision/recall/F1 + lead time from a per-container table
    with columns actual_died, predicted_die, lead_time_s."""
    tp = int(((df.predicted_die == 1) & (df.actual_died == 1)).sum())
    fp = int(((df.predicted_die == 1) & (df.actual_died == 0)).sum())
    tn = int(((df.predicted_die == 0) & (df.actual_died == 0)).sum())
    fn = int(((df.predicted_die == 0) & (df.actual_died == 1)).sum())
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (tp and (precision + recall)) else float("nan")
    lead_times = df.loc[(df.actual_died == 1) & (df.predicted_die == 1), "lead_time_s"].dropna().astype(float)
    n_died = int(df.actual_died.sum())
    lt = (float(lead_times.median()), float(lead_times.quantile(0.25)), float(lead_times.quantile(0.75))) \
        if len(lead_times) else (None, None, None)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "lead_times": lead_times, "n_died": n_died,
            "never_alerted": n_died - len(lead_times), "lt_median": lt[0], "lt_p25": lt[1], "lt_p75": lt[2]}


def early_alarm_stats(df):
    """How 'early' are the caught deaths really? Computed over true
    positives only.
      - alarm_at_first_window_frac: share of alarms that fired at the very
        first moment the method could speak. A high value means the method
        recognised "this run will die" from its start, rather than detecting
        trouble building up -- and lead time then mostly reflects episode
        length.
      - lead_fraction_median: lead time / episode duration. ~1.0 means
        "alarm at the start", small values mean "alarm near the end".
      - first_alarm_time_median_s: when (seconds after start) alarms fire."""
    tp = df[(df.actual_died == 1) & (df.predicted_die == 1)]
    if not len(tp):
        return {"alarm_at_first_window_frac": None, "lead_fraction_median": None,
                "first_alarm_time_median_s": None}
    return {
        "alarm_at_first_window_frac": float(tp.alarm_at_first_window.astype(float).mean()),
        "lead_fraction_median": float(tp.lead_fraction.astype(float).median()),
        "first_alarm_time_median_s": float(tp.first_alert_time_s.astype(float).median()),
    }


# ==========================================
# 8. Episode-level (per-container) analysis -- THE reported result
# ==========================================
def episode_level_analysis(test_dataset, test_risk, threshold, calibrator, plot_dir):
    """Per-container yes/no, lead time, alert level and calibrated
    confidence. A container is "predicted: will die" if ANY of its windows
    crosses the alert threshold; lead time is measured from the FIRST such
    window to the container's actual end. Its P(OOM-kill) is the calibrated
    probability of its peak risk score."""
    windows_by_ep = {}
    for w, (ep_idx, start_step) in enumerate(test_dataset.window_indices):
        windows_by_ep.setdefault(ep_idx, []).append((start_step, test_risk[w]))
    for ep_idx in windows_by_ep:
        windows_by_ep[ep_idx].sort(key=lambda x: x[0])

    rows = []
    for ep_idx, (scaled_feat, is_oom, t_rel, workload) in enumerate(test_dataset.episodes):
        ep_id = test_dataset.episode_ids_in_order[ep_idx]
        windows = windows_by_ep.get(ep_idx, [])
        if not windows:
            continue
        episode_duration_s = float(t_rel[-1])
        first_alert_time_s = None
        risk_at_alert = None
        alarm_at_first_window = 0
        max_risk = max(risk for _, risk in windows)
        for start_step, risk in windows:
            if risk >= threshold:
                first_alert_time_s = float(t_rel[start_step + test_dataset.lookback - 1])
                risk_at_alert = float(risk)
                alarm_at_first_window = int(start_step == windows[0][0])
                break
        predicted_die = int(first_alert_time_s is not None)
        lead_time_s = (episode_duration_s - first_alert_time_s) if (predicted_die and is_oom) else None
        lead_fraction = (lead_time_s / episode_duration_s) if (lead_time_s is not None and episode_duration_s > 0) else None

        p_oom = float(calibrator.p_oom(max_risk))
        # What an operator would have seen at the moment the alarm went off
        p_at_alert = float(calibrator.p_oom(risk_at_alert)) if risk_at_alert is not None else None
        level, confidence, conf_level = make_alert(p_oom, predicted_die)
        rows.append({
            "episode_id": ep_id, "workload": workload,
            "alert_level": level,
            "predicted_outcome": "OOM-kill" if predicted_die else "survive",
            "p_oom_kill": p_oom, "confidence": confidence, "confidence_level": conf_level,
            "p_oom_at_first_alert": p_at_alert,
            "alert_message": alert_message(ep_id, workload, level, p_oom, confidence, conf_level,
                                           first_alert_time_s),
            "actual_died": int(is_oom), "predicted_die": predicted_die,
            "correct": int(predicted_die == int(is_oom)),
            "max_risk_score": float(max_risk),
            "first_alert_time_s": first_alert_time_s, "episode_duration_s": episode_duration_s,
            "lead_time_s": lead_time_s, "lead_fraction": lead_fraction,
            "alarm_at_first_window": alarm_at_first_window if predicted_die else None,
        })

    df = pd.DataFrame(rows)
    m = summarize_predictions(df)
    early = early_alarm_stats(df)
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    accuracy, precision, recall, f1 = m["accuracy"], m["precision"], m["recall"], m["f1"]
    lead_times, n_died, never_alerted = m["lead_times"], m["n_died"], m["never_alerted"]
    brier = float(brier_score_loss(df.actual_died, df.p_oom_kill)) if len(df) else None

    print(f"\n[{plot_dir}] EPISODE-LEVEL (per-container) results:")
    print(f"  Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}   Brier score (confidence quality)={brier:.4f}")
    if early["alarm_at_first_window_frac"] is not None:
        print(f"  Early alarms: {early['alarm_at_first_window_frac']:.0%} of caught deaths alarmed in the FIRST "
              f"window; median alarm at t={early['first_alarm_time_median_s']:.1f}s; "
              f"median lead = {early['lead_fraction_median']:.0%} of the episode")

    # ---- per-container alerts: the operator-facing table ----
    alert_cols = ["episode_id", "workload", "alert_level", "predicted_outcome", "p_oom_kill",
                  "confidence", "confidence_level", "p_oom_at_first_alert", "first_alert_time_s",
                  "alert_message", "actual_died", "correct", "lead_time_s", "lead_fraction",
                  "alarm_at_first_window", "episode_duration_s"]
    level_order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    alerts_df = df[alert_cols].sort_values(by=["alert_level", "p_oom_kill"],
                                           key=lambda s: s.map(level_order) if s.name == "alert_level" else -s)
    alerts_df.to_csv(os.path.join(plot_dir, "container_alerts.csv"), index=False)
    counts = df.alert_level.value_counts()
    print(f"  Alerts: CRITICAL={counts.get('CRITICAL', 0)}  WARNING={counts.get('WARNING', 0)}  "
          f"OK={counts.get('OK', 0)}  (full list: container_alerts.csv)")
    for msg in alerts_df.alert_message.head(5):
        print(f"    {msg}")

    # How reliable is each confidence level? (key table for the paper)
    conf_rows = []
    for cl in ["High", "Medium", "Low"]:
        g = df[df.confidence_level == cl]
        conf_rows.append({"confidence_level": cl, "n_containers": len(g),
                          "accuracy": float(g.correct.mean()) if len(g) else None})
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(os.path.join(plot_dir, "accuracy_by_confidence_level.csv"), index=False)
    print("  Accuracy by confidence level: " + ", ".join(
        f"{r.confidence_level}={r.accuracy:.2f} (n={r.n_containers})" for r in conf_df.itertuples()
        if r.n_containers))

    # ---- container-level confusion matrix ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: survive", "Pred: die"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: survive", "Actual: die"])
    ax.set_title(f"Confusion Matrix (per-container)\naccuracy={accuracy:.2f} precision={precision:.2f} "
                 f"recall={recall:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- container-level risk-score distribution ----
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df.actual_died == 0, "max_risk_score"], bins=30, alpha=0.6,
              label="Survived", color="#375623")
    plt.hist(df.loc[df.actual_died == 1, "max_risk_score"], bins=30, alpha=0.6,
              label="Died", color="#C00000")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Decision threshold = {threshold:.3f}")
    plt.xlabel("Container max risk score"); plt.ylabel("Container count")
    plt.title("Container-level Risk Score Distribution"); plt.legend()
    plt.savefig(os.path.join(plot_dir, 'container_risk_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- container-level ROC curve ----
    container_roc_auc = None
    if df.actual_died.nunique() > 1:
        fpr, tpr, _ = roc_curve(df.actual_died, df.max_risk_score)
        container_roc_auc = float(sk_auc(fpr, tpr))
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2, color="#2E5395", label=f"AUC = {container_roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Container-level ROC Curve"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'container_roc.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- confidence calibration (reliability diagram) ----
    if df.actual_died.nunique() > 1:
        n_bins = max(2, min(10, len(df) // 10))
        frac_pos, mean_pred = calibration_curve(df.actual_died, df.p_oom_kill, n_bins=n_bins,
                                                strategy="quantile")
        plt.figure(figsize=(6.5, 6))
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfectly calibrated")
        plt.plot(mean_pred, frac_pos, marker="o", lw=2, color="#2E5395",
                 label=f"SurvOOM (Brier = {brier:.3f})")
        plt.xlabel("Predicted P(OOM-kill)"); plt.ylabel("Observed fraction OOM-killed")
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.title("Confidence Calibration (per container)"); plt.legend(loc="upper left"); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'confidence_calibration.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- alert levels vs actual outcome ----
    levels = ["CRITICAL", "WARNING", "OK"]
    died_counts = [int(((df.alert_level == lv) & (df.actual_died == 1)).sum()) for lv in levels]
    surv_counts = [int(((df.alert_level == lv) & (df.actual_died == 0)).sum()) for lv in levels]
    x = np.arange(len(levels))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - 0.2, died_counts, width=0.4, color="#C00000", label="Actually OOM-killed")
    plt.bar(x + 0.2, surv_counts, width=0.4, color="#375623", label="Actually survived")
    plt.xticks(x, levels); plt.ylabel("Container count")
    plt.title("Alert Level vs. Actual Outcome"); plt.legend(); plt.grid(alpha=0.3, axis="y")
    plt.savefig(os.path.join(plot_dir, 'alert_levels.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- per-workload breakdown ----
    workload_rows = []
    for wl, g in df.groupby("workload"):
        wm = summarize_predictions(g)
        workload_rows.append({
            "workload": wl, "n_containers": len(g), "n_died": wm["n_died"],
            "accuracy": wm["accuracy"], "TP": wm["tp"], "FP": wm["fp"], "TN": wm["tn"], "FN": wm["fn"],
            "lead_time_median_s": wm["lt_median"],
            "mean_confidence": float(g.confidence.mean()),
            **early_alarm_stats(g),
        })
    workload_df = pd.DataFrame(workload_rows).sort_values("workload")
    workload_df.to_csv(os.path.join(plot_dir, "per_workload_summary.csv"), index=False)

    if len(workload_df) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(workload_df.workload, workload_df.accuracy, color="#2E5395")
        axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.0)
        axes[0].set_title("Per-workload Accuracy"); axes[0].tick_params(axis='x', rotation=30)
        axes[1].bar(workload_df.workload, workload_df.lead_time_median_s.fillna(0), color="#C00000")
        axes[1].set_ylabel("Median lead time (s)")
        axes[1].set_title("Per-workload Lead Time"); axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'per_workload_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- example risk-score-over-time trajectories ----
    tp_rows = df[(df.actual_died == 1) & (df.predicted_die == 1)]
    tn_rows = df[(df.actual_died == 0) & (df.predicted_die == 0)]
    fp_rows = df[(df.actual_died == 0) & (df.predicted_die == 1)]
    examples = []
    if len(tp_rows):
        examples.append(("Died, caught early (TP)", tp_rows.iloc[0], "#C00000"))
    if len(fp_rows):
        examples.append(("Survived, false alarm (FP)", fp_rows.iloc[0], "#BF8F00"))
    if len(tn_rows):
        examples.append(("Survived, correctly quiet (TN)", tn_rows.iloc[0], "#375623"))

    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(9, 3.2 * len(examples)), sharex=False)
        if len(examples) == 1:
            axes = [axes]
        id_to_idx = {eid: i for i, eid in enumerate(test_dataset.episode_ids_in_order)}
        for ax, (label, row, color) in zip(axes, examples):
            ep_idx = id_to_idx[row.episode_id]
            _, is_oom, t_rel, _ = test_dataset.episodes[ep_idx]
            windows = windows_by_ep.get(ep_idx, [])
            times = [float(t_rel[s + test_dataset.lookback - 1]) for s, _ in windows]
            risks = [r for _, r in windows]
            ax.plot(times, risks, color=color, lw=1.5)
            ax.axhline(threshold, color="black", linestyle="--", lw=1, label="threshold")
            if pd.notna(row.first_alert_time_s):
                ax.axvline(row.first_alert_time_s, color="gray", linestyle=":", lw=1, label="first alert")
            if is_oom:
                ax.axvline(row.episode_duration_s, color="black", lw=1, label="actual death")
            ax.set_title(f"{label} — {row.episode_id} ({row.workload}) — "
                         f"{row.alert_level}, P(OOM)={row.p_oom_kill:.0%}")
            ax.set_ylabel("risk score")
            ax.legend(fontsize=7, loc="upper left")
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'example_risk_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close()

    lt_median, lt_p25, lt_p75 = m["lt_median"], m["lt_p25"], m["lt_p75"]
    if len(lead_times):
        print(f"  Of {n_died} deaths, {len(lead_times)} caught early. "
              f"Lead time median={lt_median:.1f}s (p25={lt_p25:.1f}s, p75={lt_p75:.1f}s)")
        if never_alerted:
            print(f"  WARNING: {never_alerted} death(s) never alerted.")
        plot_lead_time_hist(lead_times, n_died, never_alerted, plot_dir)

    df.drop(columns=["alert_message"]).to_csv(os.path.join(plot_dir, "episode_level_accuracy.csv"), index=False)
    pd.DataFrame([{
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "lead_time_median_s": lt_median, "lead_time_p25_s": lt_p25, "lead_time_p75_s": lt_p75,
        "n_died": n_died, "n_caught_early": len(lead_times), "never_alerted": never_alerted,
        "brier_score": brier,
        "n_critical": int(counts.get("CRITICAL", 0)), "n_warning": int(counts.get("WARNING", 0)),
        "n_ok": int(counts.get("OK", 0)), **early,
    }]).to_csv(os.path.join(plot_dir, "episode_level_summary.csv"), index=False)

    return {
        "episode_accuracy": accuracy, "episode_precision": precision,
        "episode_recall": recall, "episode_f1": f1,
        "episode_TP": tp, "episode_FP": fp, "episode_TN": tn, "episode_FN": fn,
        "episode_lead_time_median_s": lt_median, "episode_lead_time_p25_s": lt_p25,
        "episode_lead_time_p75_s": lt_p75, "n_died_episodes": n_died, "never_alerted": never_alerted,
        "container_roc_auc": container_roc_auc, "brier_score": brier, **early,
        "_lead_times_raw": lead_times.tolist(),  # internal: used by plot_tier_comparison's boxplot
    }


def plot_lead_time_hist(lead_times, n_died, never_alerted, plot_dir):
    lt_median = float(lead_times.median())
    plt.figure(figsize=(8, 5))
    plt.hist(lead_times, bins=25, color="#2E5395", edgecolor="white")
    plt.axvline(lt_median, color="#C00000", linestyle="--", label=f"median = {lt_median:.0f}s")
    plt.xlabel("Seconds of advance warning before death"); plt.ylabel("Container count")
    plt.title(f"Lead Time Distribution — per container\n{never_alerted} of {n_died} deaths never alerted")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'lead_time_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 9. C-index + Kaplan-Meier -- also per-episode
# ==========================================
def _safe_cindex(times, risk, events):
    times, risk, events = np.asarray(times), np.asarray(risk), np.asarray(events)
    if len(times) < 10 or events.sum() < 2:
        return None
    try:
        return float(concordance_index(times, -risk, events))
    except ZeroDivisionError:      # no comparable pairs
        return None


def container_level_survival_plot(test_dataset, test_risk, plot_dir):
    """Three C-index variants, because the plain "first window only" one
    is misleading: early on, PSI is ~0 for every container, so psi-only
    gets exactly 0.5 from ties, while usage-based tiers score high just
    from where usage started.

      c_index            -- peak risk over the whole episode vs. time to
                            death. Matches the alert rule. Uses windows up
                            to the end, so it is optimistic.
      c_index_at_<L>s    -- LANDMARK C-index (the honest, prospective one):
                            only containers still running at L seconds,
                            risk = highest score seen up to L, ranked
                            against the time remaining after L.
      c_index_first_window -- the old metric, kept for transparency.
    Kaplan-Meier groups use the first landmark (risk known at L s, survival
    after L s), so the plot doesn't peek at the end of the episode either."""
    if not HAVE_LIFELINES:
        print("[!] lifelines not installed — skipping C-index and Kaplan-Meier plot.")
        return {}

    lookback = test_dataset.lookback
    per_ep = {}   # ep_idx -> list of (window_end_time, risk), in time order
    for w, (e, s) in enumerate(test_dataset.window_indices):
        t_rel = test_dataset.episodes[e][2]
        per_ep.setdefault(e, []).append((float(t_rel[s + lookback - 1]), float(test_risk[w])))

    peak, first, t_end, event = [], [], [], []
    for e, wins in per_ep.items():
        wins.sort()
        _, is_oom, t_rel, _ = test_dataset.episodes[e]
        peak.append(max(r for _, r in wins)); first.append(wins[0][1])
        t_end.append(float(t_rel[-1])); event.append(is_oom)
    out = {"c_index": _safe_cindex(t_end, peak, event),
           "c_index_first_window": _safe_cindex(t_end, first, event)}

    rows = [{"variant": "peak risk (whole episode)", "n_containers": len(t_end),
             "n_deaths": int(np.sum(event)), "c_index": out["c_index"]},
            {"variant": "first window only (old metric)", "n_containers": len(t_end),
             "n_deaths": int(np.sum(event)), "c_index": out["c_index_first_window"]}]
    landmark_data = {}
    for L in LANDMARKS_S:
        r_l, t_l, e_l = [], [], []
        for e, wins in per_ep.items():
            _, is_oom, t_rel, _ = test_dataset.episodes[e]
            seen = [r for t, r in wins if t <= L]
            if float(t_rel[-1]) > L and seen:          # still running at L, and scorable by L
                r_l.append(max(seen)); t_l.append(float(t_rel[-1]) - L); e_l.append(is_oom)
        c = _safe_cindex(t_l, r_l, e_l)
        out[f"c_index_at_{L}s"] = c
        landmark_data[L] = (np.array(r_l), np.array(t_l), np.array(e_l))
        rows.append({"variant": f"landmark at {L}s", "n_containers": len(t_l),
                     "n_deaths": int(np.sum(e_l)), "c_index": c})
    pd.DataFrame(rows).to_csv(os.path.join(plot_dir, "c_index_variants.csv"), index=False)
    print("  C-index: " + ", ".join(f"{r['variant']}={r['c_index']:.3f} (n={r['n_containers']})"
                                    for r in rows if r["c_index"] is not None))

    # Kaplan-Meier at the first landmark with enough data
    L = next((L for L in LANDMARKS_S if out.get(f"c_index_at_{L}s") is not None), None)
    if L is not None:
        r_l, t_l, e_l = landmark_data[L]
        risk_group = np.digitize(r_l, np.quantile(r_l, [1 / 3, 2 / 3]))
        plt.figure(figsize=(8, 6))
        kmf = KaplanMeierFitter()
        colors = {0: "#375623", 1: "#BF8F00", 2: "#C00000"}
        names = {0: "Low predicted risk", 1: "Medium predicted risk", 2: "High predicted risk"}
        for g in [0, 1, 2]:
            mask = risk_group == g
            if mask.sum() < 2:
                continue
            kmf.fit(t_l[mask], event_observed=e_l[mask], label=names[g])
            kmf.plot_survival_function(color=colors[g])
        plt.xlabel(f"Seconds after t = {L}s"); plt.ylabel("Survival probability")
        plt.title(f"Kaplan-Meier by risk known at t = {L}s (landmark C-index = {out[f'c_index_at_{L}s']:.3f})")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'kaplan_meier_by_risk.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return out


# ==========================================
# 10. Rule baseline: alert when usage_ratio >= 0.99
# ==========================================
def usage_threshold_baseline(test_episode_ids, episodes, run_dir, ratio=BASELINE_USAGE_RATIO):
    """The obvious non-ML alternative a reviewer will ask about. Evaluated on
    exactly the same test containers as the GRU (the ones its test set kept),
    with the same container-level rules. It has no confidence score."""
    tier = f"baseline-usage{int(round(ratio * 100))}"
    plot_dir = os.path.join(run_dir, 'results', tier)
    os.makedirs(plot_dir, exist_ok=True)
    by_id = {e["episode_id"]: e for e in episodes}
    rows = []
    for ep_id in test_episode_ids:
        ep = by_id[ep_id]
        df = pd.read_csv(ep["csv_path"], skipinitialspace=True)
        df.columns = df.columns.str.strip()
        if "usage_ratio" not in df.columns or "t_rel" not in df.columns:
            continue
        t_rel = df["t_rel"].astype(float).values
        hits = np.where(df["usage_ratio"].fillna(0).astype(float).values >= ratio)[0]
        first_alert = float(t_rel[hits[0]]) if len(hits) else None
        died = int(ep["died"])
        duration = float(t_rel[-1])
        lead = (duration - first_alert) if (first_alert is not None and died) else None
        rows.append({
            "episode_id": ep_id, "workload": ep["workload"], "actual_died": died,
            "predicted_die": int(first_alert is not None), "first_alert_time_s": first_alert,
            "episode_duration_s": duration, "lead_time_s": lead,
            "lead_fraction": (lead / duration) if (lead is not None and duration > 0) else None,
            # "first window" for the rule = within the first LOOKBACK samples, i.e. no later
            # than the GRU's earliest possible alarm -- so the two are comparable
            "alarm_at_first_window": int(hits[0] <= LOOKBACK_WINDOW - 1) if len(hits) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(plot_dir, "episode_level_accuracy.csv"), index=False)
    m = summarize_predictions(df)
    early = early_alarm_stats(df)
    print(f"\n[{tier}] Rule baseline (alert when usage_ratio >= {ratio:.2f}):")
    print(f"  Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  median lead time={m['lt_median']}")
    if early["alarm_at_first_window_frac"] is not None:
        print(f"  Early alarms: {early['alarm_at_first_window_frac']:.0%} within the first {LOOKBACK_WINDOW} samples; "
              f"median lead = {early['lead_fraction_median']:.0%} of the episode")
    if len(m["lead_times"]):
        plot_lead_time_hist(m["lead_times"], m["n_died"], m["never_alerted"], plot_dir)
    return {
        "tier": tier, "episode_accuracy": m["accuracy"], "episode_precision": m["precision"],
        "episode_recall": m["recall"], "episode_f1": m["f1"],
        "episode_TP": m["tp"], "episode_FP": m["fp"], "episode_TN": m["tn"], "episode_FN": m["fn"],
        "episode_lead_time_median_s": m["lt_median"], "episode_lead_time_p25_s": m["lt_p25"],
        "episode_lead_time_p75_s": m["lt_p75"], "n_died_episodes": m["n_died"],
        "never_alerted": m["never_alerted"], "_lead_times_raw": m["lead_times"].tolist(),
        "threshold_rule": f"usage_ratio >= {ratio:.2f}", **early,
    }


# ==========================================
# 11. Put it together per tier
# ==========================================
def evaluate_and_plot(model, train_dataset, val_dataset, test_dataset, device, tier, run_dir):
    plot_dir = os.path.join(run_dir, 'results', tier)
    os.makedirs(plot_dir, exist_ok=True)
    summary = {"tier": tier}

    # Alert threshold + confidence calibrator: VALIDATION containers only
    # (held out from training, never seen by the model; test is untouched)
    val_risk = score_all_windows(model, val_dataset, device)
    val_peak, val_died = container_peak_risk(val_dataset, val_risk)
    threshold, rule = pick_container_threshold(val_peak, val_died)
    calibrator = ConfidenceCalibrator.fit(val_peak, val_died)
    val_pred = val_peak >= threshold
    summary["decision_threshold"] = float(threshold)
    summary["threshold_rule"] = rule
    summary["val_precision"] = float(val_died[val_pred].mean()) if val_pred.any() else None
    summary["val_recall"] = float(val_pred[val_died == 1].mean()) if (val_died == 1).any() else None
    print(f"[{tier}] Container-level threshold = {threshold:.3f} ({rule}); "
          f"on {len(val_peak)} validation containers: precision={summary['val_precision']}, "
          f"recall={summary['val_recall']}")

    # Everything needed to alert on a new container later (predict mode)
    model_dir = os.path.join(run_dir, 'saved_models', tier)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({
        "tier": tier, "features": train_dataset.features_list, "scaler": train_dataset.scaler,
        "threshold": float(threshold), "calibrator": calibrator.to_dict(),
        "lookback": train_dataset.lookback, "use_rates": USE_RATES,
        "hidden_size": 128, "num_layers": 2,
        "conf_high": CONF_HIGH, "conf_medium": CONF_MEDIUM, "critical_p": CRITICAL_P,
    }, os.path.join(model_dir, 'bundle.joblib'))

    test_risk = score_all_windows(model, test_dataset, device)

    summary.update(episode_level_analysis(test_dataset, test_risk, threshold, calibrator, plot_dir))
    summary.update(container_level_survival_plot(test_dataset, test_risk, plot_dir))

    printable = {k: v for k, v in summary.items() if k != "_lead_times_raw"}
    print(f"[{tier}] Summary: {json.dumps(printable, indent=2, default=str)}")
    print(f"Saved all figures to {plot_dir}/")
    return summary


def plot_tier_comparison(all_summaries, run_dir):
    out_path = os.path.join(run_dir, 'results', 'tier_comparison.png')
    tiers = [s["tier"] for s in all_summaries]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ci = [s for s in all_summaries if s.get("c_index") is not None]
    if ci:
        axes[0].bar([s["tier"] for s in ci], [s["c_index"] for s in ci], color="#2E5395")
        axes[0].set_ylabel("C-index"); axes[0].set_title("Concordance Index by Tier")
        axes[0].set_ylim(0.5, 1.0); axes[0].axhline(0.5, color="gray", linestyle="--")
        axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(tiers, [s.get("episode_accuracy") or 0 for s in all_summaries], color="#C00000")
    axes[1].set_ylabel("Episode-level accuracy"); axes[1].set_title("Per-container accuracy by Tier")
    axes[1].set_ylim(0, 1.0); axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ---- lead-time vs. accuracy trade-off, one point per tier ----
    plt.figure(figsize=(7, 6))
    colors = ["#2E5395", "#7030A0", "#BF8F00", "#375623", "#C00000", "#7F7F7F"]
    for i, s in enumerate(all_summaries):
        x = s.get("episode_lead_time_median_s")
        y = s.get("episode_accuracy")
        if x is None or y is None:
            continue
        marker = "s" if s["tier"].startswith("baseline") else "o"
        plt.scatter(x, y, s=140, marker=marker, color=colors[i % len(colors)], edgecolor="black", zorder=3)
        plt.annotate(s["tier"], (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    plt.xlabel("Median lead time (s)"); plt.ylabel("Episode-level accuracy")
    plt.title("Lead Time vs. Accuracy Trade-off, by Tier")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, 'results', 'lead_time_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- lead-time boxplots, all tiers side by side ----
    lead_time_lists = [s.get("_lead_times_raw", []) for s in all_summaries]
    if any(lead_time_lists):
        plt.figure(figsize=(10, 5))
        bp = plt.boxplot([lt if lt else [np.nan] for lt in lead_time_lists], tick_labels=tiers,
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        plt.ylabel("Lead time (s), caught deaths only")
        plt.title("Lead Time Distribution by Tier")
        plt.xticks(rotation=15)
        plt.grid(alpha=0.3, axis='y')
        plt.savefig(os.path.join(run_dir, 'results', 'lead_time_boxplot_by_tier.png'), dpi=300, bbox_inches='tight')
        plt.close()

    rows = []
    for s in all_summaries:
        rows.append({k: s.get(k) for k in
                     ["tier", "threshold_rule", "decision_threshold",
                      "episode_accuracy", "episode_precision", "episode_recall", "episode_f1",
                      "episode_TP", "episode_FP", "episode_TN", "episode_FN",
                      "episode_lead_time_median_s", "episode_lead_time_p25_s", "episode_lead_time_p75_s",
                      "alarm_at_first_window_frac", "lead_fraction_median", "first_alarm_time_median_s",
                      "container_roc_auc", "brier_score",
                      "c_index", *[f"c_index_at_{L}s" for L in LANDMARKS_S], "c_index_first_window",
                      "n_died_episodes", "never_alerted"]})
    csv_path = os.path.join(run_dir, 'results', 'tier_comparison.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved tier_comparison.png, lead_time_vs_accuracy.png, "
          f"lead_time_boxplot_by_tier.png, and tier_comparison.csv under {os.path.join(run_dir, 'results')}/")


# ==========================================
# 12. Predict mode: alert + confidence for ONE container's CSV
# ==========================================
def predict_container(model_dir, csv_path):
    """Score one container from its CSV alone (no meta.json), exactly as it
    would run live. Works on a finished episode or a partial CSV that is
    still being written: in that case "OK" means "no alert so far"."""
    bundle = joblib.load(os.path.join(model_dir, 'bundle.joblib'))
    features, lookback = bundle["features"], bundle["lookback"]
    meta_needed = [c for c in features if c in META_COLS]
    if meta_needed:
        sys.exit(f"Tier '{bundle['tier']}' needs {meta_needed} from meta.json, which isn't available "
                 f"for a live container. Use a csv-only, no-distance-strict or psi-only model.")

    df = prepare_df(csv_path, None, use_rates=bundle["use_rates"])
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns the model needs: {missing}")
    if len(df) < lookback:
        sys.exit(f"Need at least {lookback} samples to score; this CSV has {len(df)}.")

    model = GRUSurvival(num_inputs=len(features), hidden_size=bundle["hidden_size"],
                        num_layers=bundle["num_layers"])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'gru_survival.pt'), map_location="cpu"))
    model.eval()

    X = bundle["scaler"].transform(df[features].values).astype(np.float32)
    t_rel = df["t_rel"].values.astype(np.float64)
    with torch.no_grad():
        windows = np.stack([X[i: i + lookback] for i in range(len(X) - lookback + 1)])
        risk = model(torch.tensor(windows)).numpy().flatten()

    cal = ConfidenceCalibrator(**bundle["calibrator"])
    crossed = np.where(risk >= bundle["threshold"])[0]
    first_alert_time = float(t_rel[crossed[0] + lookback - 1]) if len(crossed) else None
    p_oom = float(cal.p_oom(risk.max()))
    level, confidence, conf_level = make_alert(p_oom, first_alert_time is not None)
    name = os.path.basename(csv_path).replace(".csv", "")
    result = {
        "container": name, "alert_level": level, "p_oom_kill": p_oom,
        "confidence": confidence, "confidence_level": conf_level,
        "first_alert_time_s": first_alert_time,
        "p_oom_now": float(cal.p_oom(risk[-1])), "observed_until_s": float(t_rel[-1]),
        "message": alert_message(name, None, level, p_oom, confidence, conf_level, first_alert_time),
    }
    print(result["message"])
    print(json.dumps(result, indent=2))
    return result


# ==========================================
# 13. Leave-one-workload-out driver + CLI entry point
# ==========================================
def run_training(holdout=None, tiers=None, run_dir=None):
    tiers = tiers or TIERS
    # Every run gets its own folder, so nothing from a previous run is
    # ever overwritten: saved_datasets/, saved_models/, and results/
    # all live underneath this one run_dir.
    if run_dir is None:
        run_dir = f"paper_run_{datetime.now():%Y%m%d_%H%M%S}" + (f"_holdout-{holdout}" if holdout else "")
    os.makedirs(run_dir, exist_ok=True)
    print(f"This run's outputs will be saved under: {run_dir}/")
    print(f"Model input: sliding window of {LOOKBACK_WINDOW} rows.")

    episodes = discover_episodes(DATA_DIR)
    print(f"Discovered {len(episodes)} episodes "
          f"({sum(e['died'] for e in episodes)} died, from meta.json ground truth).")

    if holdout:
        train_ids, test_ids = make_holdout_split(episodes, holdout)
        n_died = sum(e["died"] for e in episodes if e["episode_id"] in test_ids)
        print(f"UNSEEN-WORKLOAD TEST: training on {sorted({e['workload'] for e in episodes} - {holdout})}; "
              f"testing on all {len(test_ids)} '{holdout}' containers ({n_died} died, {len(test_ids) - n_died} survived).")
    else:
        train_ids, test_ids = make_episode_split(episodes)              # same test set as earlier runs
    fit_ids, val_ids = make_validation_split(episodes, train_ids)   # validation carved from train
    print(f"Episode-level split: {len(fit_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

    all_summaries = []
    baseline_ids = None
    for tier in tiers:
        print(f"\n{'=' * 50}\nGRU Survival Analysis — tier: {tier}\n{'=' * 50}")
        model, train_dataset, val_dataset, test_dataset, device = train_model(
            tier, episodes, fit_ids, val_ids, test_ids, run_dir)
        summary = evaluate_and_plot(model, train_dataset, val_dataset, test_dataset, device, tier, run_dir)
        all_summaries.append(summary)
        if baseline_ids is None:
            baseline_ids = list(test_dataset.episode_ids_in_order)

    all_summaries.append(usage_threshold_baseline(baseline_ids, episodes, run_dir))

    plot_tier_comparison(all_summaries, run_dir)
    print(f"\nAll tiers completed! Everything is under {run_dir}/")
    return all_summaries


def run_leave_one_workload_out(tiers=None):
    """Hold out each workload in turn. One sub-folder per held-out workload,
    plus one combined table: lowo_summary.csv."""
    parent = f"paper_run_{datetime.now():%Y%m%d_%H%M%S}_lowo"
    workloads = sorted({e["workload"] for e in discover_episodes(DATA_DIR)})
    rows = []
    for wl in workloads:
        print(f"\n{'#' * 60}\n# HOLD-OUT WORKLOAD: {wl}\n{'#' * 60}")
        for s in run_training(holdout=wl, tiers=tiers, run_dir=os.path.join(parent, f"holdout-{wl}")):
            rows.append({"held_out_workload": wl, **{k: v for k, v in s.items() if not k.startswith("_")}})
    out = pd.DataFrame(rows)
    keep = ["held_out_workload", "tier", "episode_accuracy", "episode_precision", "episode_recall",
            "episode_FP", "episode_FN", "n_died_episodes", "episode_lead_time_median_s",
            "container_roc_auc", "brier_score", "c_index_at_60s", "c_index_at_120s"]
    out = out[[c for c in keep if c in out.columns]]
    out.to_csv(os.path.join(parent, "lowo_summary.csv"), index=False)
    print(f"\nLeave-one-workload-out finished. Combined table: {parent}/lowo_summary.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SurvOOM GRU: train/evaluate, or alert on one container.")
    parser.add_argument("--holdout-workload", default=None, metavar="NAME",
                        help="unseen-workload test: train WITHOUT this workload, test on all of its "
                             "containers (e.g. spike). 'all' = hold out each workload in turn")
    parser.add_argument("--tiers", nargs="+", choices=TIERS, default=None,
                        help="only run these feature tiers (default: all), e.g. --tiers csv-only")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("predict", help="alert + confidence for one container CSV")
    p.add_argument("--model-dir", required=True,
                   help="e.g. paper_run_<ts>/saved_models/csv-only")
    p.add_argument("--csv", required=True, help="the container's metrics CSV")
    args = parser.parse_args()
    if args.cmd == "predict":
        predict_container(args.model_dir, args.csv)
    elif args.holdout_workload == "all":
        run_leave_one_workload_out(tiers=args.tiers)
    elif args.holdout_workload:
        known = sorted({e["workload"] for e in discover_episodes(DATA_DIR)})
        if args.holdout_workload not in known:
            sys.exit(f"Unknown workload '{args.holdout_workload}'. Workloads in {DATA_DIR}: {known}")
        run_training(holdout=args.holdout_workload, tiers=args.tiers)
    else:
        run_training(tiers=args.tiers)
