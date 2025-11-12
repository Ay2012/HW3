# HW3 — IMDb Sentiment with RNN/LSTM/BiLSTM

Binary sentiment classification on the IMDb 50k dataset, comparing RNN variants and training choices. The pipeline follows the assignment spec: fixed 25k/25k (50-50) split, top-10k vocab (train-only), and sequence lengths 25/50/100.

## Repo Layout

Data/
Raw/IMDB Dataset.csv
processed/
vocab.json, vocab.txt
imdb_train_len{25,50,100}.csv
imdb_test_len{25,50,100}.csv

results/
metrics.csv
summary_metrics.csv
loss_curve_*.csv

src/
Hw3.ipynb

README.md
requirements.txt
ReportHW3.pdf

# Data 
the IMDb datset is placed inside the Raw folder inside the Data folder

Preprocessing (per spec)

 Lowercase, strip HTML/punct/special chars
 Whitespace tokenization
 Vocab from TRAIN only (top 10,000; 0 <pad>, 1 <unk>)
 Convert to token IDs
 Pad/Truncate to lengths 25, 50, 100
 Save to Data/processed/

# Experiments

I vary one factor at a time (others fixed):
Architecture: RNN / LSTM / BiLSTM
Activation: Sigmoid / ReLU / Tanh
Optimizer: Adam / SGD / RMSProp
Seq length: 25 / 50 / 100
Stability: No clipping vs. Gradient Clipping (e.g., max-norm 1.0)

Design defaults (when not being varied):
embed_dim=100, hidden_size=64, num_layers=2, dropout=0.5, batch_size=32, loss=BCEWithLogitsLoss, output sigmoid threshold=0.5, epochs=5.

Metrics logged per run to results/metrics.csv:

accuracy, f1_macro, final_test_loss, epoch_time_s(last_train)
Plus configuration columns (model, activation, optimizer, seq_len, grad_clip, seed, device).
Per-epoch loss curves are saved to results/loss_curve_*.csv for the required plots.

# Reproduce

Preprocess (once) to create Data/processed/*.
Run training notebook/cells (the ones I provided) to execute the full matrix:
 Architectures sweep
 Activations sweep
 Optimizers sweep
 Sequence lengths sweep
 Gradient clipping on/off

# Outputs You Should See

results/metrics.csv — one row per experiment (as in the assignment’s example table).
results/summary_metrics.csv — sorted view for the report.
results/loss_curve_*.csv

# Reproducibility


Seed: 1337 for Python/NumPy/Torch
Device: CPU by default 
State your hardware (CPU, RAM) in the report’s reproducibility section.

# Notes & Troubleshooting
Don’t build vocab before the split. All frequency statistics must be train-only.
If you see repeated rows in the summary (e.g., many LSTM/ReLU/50), that’s expected when sweeping other factors (optimizer, clipping, length) — each sweep includes a baseline config for fair comparison.
On laptops, seq_len=100 is slower but may produce the best F1/accuracy.#