# CLAUDE.md — Mia Chess Engine

## What this project is

**Mia** is a chess engine written from scratch in Rust, built as a hands-on **machine learning learning project**.
The engine starts with a classic hand-written evaluation, then moves to a learned one: first by tuning the
hand-written weights with data (Texel tuning), then by replacing the evaluation with a small neural network (NNUE)
trained on the engine's own self-play games.

- **Owner:** Lucas Romero-Wyhnanskyj
- **Main goal:** learn ML deeply by building every step by hand (data, training, quantization, inference, evaluation).
- **Second goal:** a strong engine. Strength is the scoreboard that proves the ML worked.
- **Machine:** MacBook with Apple Silicon (M5). Local training uses PyTorch MPS; SIMD on this machine is ARM NEON.

## How Claude should help on this project

- **Explain in extremely simple language.** Short sentences, plain words, define any jargon the first time it appears.
- **Learning comes first.** Don't swap in a library that skips an ML step this plan says to write by hand
  (gradient descent in NumPy, the PyTorch training loop, NNUE inference in Rust). Point out shortcuts, but don't take them unasked.
- **One change at a time.** Every strength change is its own branch and its own SPRT test.
- **Correctness before speed.** Never optimize move generation or search until perft and tests pass.
- **Follow the decisions below.** If a decision should change, say why and update `docs/decisions.md` in the same change.
- **Measure, don't guess.** Claims that something is "stronger" need an SPRT result. Claims that something is "faster" need a benchmark.

---

## Roadmap

| Phase | What | Main ML lesson |
|---|---|---|
| 1 | Classic engine: board, movegen, search, hand-written eval, UCI | (foundation, no ML yet) |
| 2 | Texel tuning of the hand-written eval (NumPy) | Loss functions, sigmoid, gradient descent, train/val split, overfitting |
| 3 | Self-play data pipeline | Data quality, filtering, labeling, storage, lineage |
| 4 | NNUE: train in PyTorch, quantize, run in Rust | Feature encoding, architecture, LR schedules, quantization, inference |
| 5 | Testing + experiment loop | SPRT, experiment tracking, ablations, the self-play improvement loop |
| 6 | Stretch: policy net for move ordering, mini AlphaZero (MCTS), cloud GPU training | Deeper ML, RL, distributed training |

Rough pace (part-time): Phase 1 a few weeks · Phases 2–3 a couple of weeks each · Phase 4 a month or more · 5–6 open-ended.

---

## Technical decisions

### 1. Engine language
- **Rust** (stable toolchain, edition 2024).
- Release profile: `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`. Local builds may use `-C target-cpu=native`.

### 2. Board representation
- **Bitboards + a 64-square piece array** (written by hand, no chess library in the engine).
  - One `u64` per (color, piece type), plus occupancy per color.
  - `[Piece; 64]` array (or `Option<Piece>`) for fast "what is on square X?".
- Square indexing: `a1 = 0, b1 = 1, ..., h8 = 63` (little-endian rank-file).
- Sliding attacks: **start with simple ray loops**, then upgrade to **magic bitboards** once perft passes. Keep the loop version for tests.

### 3. Move generation
- **Pseudo-legal generation + legality check** (make the move, reject it if own king is attacked).
  Optional later upgrade: fully legal generation, verified against the pseudo-legal version.
- **Copy-make:** the `Board` struct is `Copy`/`Clone` and small. Playing a move returns a new board; no unmake code.
- **16-bit packed moves:**
  - bits 0–5: from square · bits 6–11: to square · bits 12–15: flags
  - flags: `0` quiet · `1` double pawn push · `2` king castle · `3` queen castle · `4` capture · `5` en passant ·
    `8–11` promotion (N, B, R, Q) · `12–15` promotion-capture (N, B, R, Q)
  - Accessors like `mv.from()`, `mv.to()`, `mv.is_capture()` so code stays readable.
- Zobrist hash is updated incrementally on every move.

### 4. Search
- **Negamax alpha-beta.** MCTS is reserved for the Phase 6 stretch.
- **Single-threaded first.** Lazy SMP only later, and only if wanted for real games.
- Add features **one at a time, each with its own SPRT test**, in this order:
  1. Iterative deepening
  2. Quiescence search (captures only)
  3. Basic time management (≈ remaining/20 + increment/2)
  4. Transposition table (Zobrist keys, size set by UCI `Hash` option)
  5. Move ordering: TT move → MVV-LVA captures → killer moves → history heuristic
  6. Principal Variation Search (PVS)
  7. Null move pruning
  8. Late move reductions (LMR)
  9. Check extensions, aspiration windows

### 5. Evaluation design
- **`Evaluator` trait + generics.** The search is generic over `E: Evaluator` (static dispatch, no speed cost).
  - `fn evaluate(&self, board: &Board) -> i32`
  - Move hooks (e.g. `on_move`) for incremental updates. The hand-written eval may ignore them; NNUE uses them for its accumulator.
- **Hand-written eval (HCE) to start:** material + piece-square tables + **tapered eval** (middlegame/endgame blend by game phase).
  Optional later terms: bishop pair, passed/doubled/isolated pawns, mobility, simple king safety.
- **All eval weights live in one data structure, loaded from a file.** No magic numbers scattered in code.
  This is what Texel tuning reads and writes.
- **Starting values are our own simple guesses** (not copied tuned tables), so the gain from tuning is visible.
- Scores: `i32` **centipawns**, always from the **side to move's** view. Mate scores use a reserved high range
  (e.g. `MATE = 32000`, "mate in N" = `MATE - ply`).

### 6. Interface and testing
- **UCI protocol.** Custom debug commands:
  - `perft <depth>` — count leaf nodes (with per-move breakdown)
  - `d` — print the board, FEN, and Zobrist key
  - `eval` — print the evaluation and its parts
  - `bench` — fixed-depth search on a fixed position set; prints total nodes (the behavior fingerprint) and NPS
- **Correctness tests:**
  - Perft suite with known counts, for example:
    - Start position: depth 4 = 197,281 · depth 5 = 4,865,609 · depth 6 = 119,060,324
    - Kiwipete `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`: depth 3 = 97,862 · depth 4 = 4,085,603
    - Position 3 `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1`: depth 5 = 674,624
  - `cargo test` unit tests: FEN round-trip, move encoding, special moves.
  - Debug-only assertions: after each move, recompute Zobrist hash (and later the NNUE accumulator) from scratch and compare.
  - Reference comparison (tests only): random games compared against `shakmaty`'s legal move lists. `shakmaty` is a
    **dev-dependency only**, never used by the engine itself.
- **Strength tests:** **fastchess** + **SPRT**.
  - Typical time control: ~8s + 0.08s.
  - Bounds: `[0, 5]` Elo normally; `[0, 10]` early on when changes are large.
  - Games start from a balanced opening book (positions stored in `testing/books/`).
- Tests run **locally** on the Mac. OpenBench is an optional later project.

### 7. ML tooling
- **Python for training, Rust for the engine.** They talk **through files only** (no PyO3 for now).
- Libraries, in order: **NumPy** (Texel, gradient descent written by hand) → **PyTorch** (NNUE, own training loop) · **Matplotlib** for plots.
- Environment: **uv** (Python 3.12+), with a committed lock file.
- Experiment tracking: **Weights & Biases**.
- Real training code lives in **scripts driven by config files** (YAML/TOML) with a **fixed random seed**.
  Notebooks are only for exploring data and quick plots.
- Device selection is automatic: `mps` → `cuda` → `cpu`.

### 8. Training data
- **Sources:** high-rated **Lichess database games** for Texel tuning · **self-play** for NNUE.
- **Self-play setup:** ~8 random opening plies (or book positions) · **fixed node count per move** (e.g. 5,000) ·
  adjudication (resign when clearly lost for several moves, draw when clearly drawn) · one game per CPU core.
- **Labels per position:** search score (centipawns) + game result (win/draw/loss), both from the **side to move's** view.
- **Filter out:** opening plies, positions in check, positions where the best move is a capture or promotion (not quiet),
  mate/extreme scores. **Deduplicate and shuffle** the whole dataset before training.
- **Formats:**
  - Stage 1 (text): one line per position, `FEN | score | result`.
  - Stage 2 (binary): fixed-size ~32-byte records (packed board + score + result), read in Python with NumPy memmap.
    Exact byte layout goes in `docs/data-format.md` when Phase 3 starts.
- **Every dataset is versioned** with an info file (engine version, net used, search settings, position count, filters).
- **Sizes:** ~1M positions for Texel · tens of millions for the first NNUE, scaling to hundreds of millions.

### 9. Neural network (NNUE)
- **Inputs:** 768 = 6 piece types × 2 colors × 64 squares, one-hot. King buckets come later.
- **Two perspectives** (White view and flipped Black view), concatenated **side to move first**.
- **Shape:** `768 → 256 (×2 perspectives) → 1`. Later: 512/1024 hidden, output buckets.
- **Activation:** clipped ReLU (clamp 0..1). Later: try SCReLU.
- **Loss:** `sigmoid(output / 400)` vs. target `λ · sigmoid(score / 400) + (1 − λ) · result`, mean squared error.
  λ is an experiment knob.
- **Optimizer:** AdamW · batches of ~16k positions · decaying learning rate (step or cosine) ·
  weight clipping so weights fit the quantized range.
- **Quantization:** train in float32; export to **int16** (hidden layer scale `QA = 255`, output scale `QB = 64`);
  integer-only inference in the engine.
- **Inference in Rust:** an **accumulator** (hidden layer values) updated incrementally on each move
  (subtract old-square weights, add new-square weights), stored with the board copy (fits copy-make).
  Plain loops first; explicit SIMD (NEON locally, AVX2 on x86) later, with the plain version kept as the reference.
- **Weights file:** header (format version, architecture, quantization scales) + weights. Engine refuses mismatched headers.
  Loaded from a file in development; embedded in release builds.
- **Lineage:** nets are **named by content hash** and linked to their W&B run and dataset info file.
- **Python vs. Rust check:** on a fixed test position set, the float PyTorch output and the int Rust output must closely match.

### 10. Compute
- **Data generation + SPRT:** the local Mac, all cores.
- **Texel + early NNUE training:** the local Mac (PyTorch MPS).
- **Bigger training runs:** **Modal** (rented GPUs). **UF HiPerGator** if access is granted through a professor/lab.
- **Docker** for the training code so runs are identical locally and in the cloud.
- Cloud object storage (S3/R2/GCS) once datasets reach tens of GB.

### 11. Repo structure and workflow
- **Monorepo:**
  ```
  mia/
  ├── CLAUDE.md
  ├── README.md
  ├── engine/                 # Rust Cargo workspace
  │   ├── crates/
  │   │   ├── core/           # board, bitboards, movegen, Zobrist, FEN, move encoding
  │   │   ├── engine/         # search, Evaluator trait, HCE, NNUE inference, UCI → `mia` binary
  │   │   └── datagen/        # self-play data generator → `mia-datagen` binary
  │   └── tests/              # perft suite, shakmaty comparison
  ├── training/               # Python (uv project)
  │   ├── texel/              # Texel tuning (NumPy)
  │   ├── nnue/               # model, training loop, export (PyTorch)
  │   ├── data/               # loaders, filtering, shuffling
  │   └── configs/            # one config per experiment
  ├── testing/                # fastchess scripts, SPRT settings, opening books
  ├── docs/                   # decisions.md, experiments.md, data-format.md
  ├── data/                   # datasets (gitignored; info files committed)
  └── nets/                   # trained nets (gitignored; info files committed)
  ```
- **Git workflow:**
  - `main` always builds and passes tests.
  - One change per branch.
  - Strength changes merge only after **SPRT passes**. Non-functional changes (refactors, speedups) merge only if the
    **bench node count is unchanged**.
  - Every commit message ends with `Bench: <nodes>`.
- **Big files:** `data/` and `nets/` are gitignored; their small info files are committed. Release nets go in GitHub Releases.
- **CI (GitHub Actions)** on push/PR:
  - Rust: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, quick perft, bench check.
  - Python: `ruff check`, `ruff format --check`, `pytest`.
  - SPRT runs locally, not in CI.
- **Docs:**
  - `docs/decisions.md` — these decisions plus any later changes and why.
  - `docs/experiments.md` — one entry per experiment: idea, settings, Elo result (± error), W&B link.

---

## Common commands

```bash
# Engine (from engine/)
cargo build --release                     # build `mia` and `mia-datagen`
cargo test                                # unit + perft + reference tests
cargo run --release -p engine             # start the UCI engine
echo "bench" | ./target/release/mia       # behavior fingerprint (node count)
cargo clippy -- -D warnings && cargo fmt --check

# Training (from training/)
uv sync                                   # install Python deps from the lock file
uv run python -m texel.tune --config configs/<exp>.yaml
uv run python -m nnue.train --config configs/<exp>.yaml
uv run python -m nnue.export --checkpoint <path> --out ../nets/
uv run pytest && uv run ruff check

# Strength testing (from testing/)
./sprt.sh <new-binary> <base-binary>      # fastchess SPRT wrapper script
```

## Definition of done for a change
1. `cargo test` passes (including perft).
2. `bench` count is unchanged, or the change is a strength change with a passing SPRT.
3. Commit message ends with `Bench: <nodes>`.
4. If it was an experiment: an entry in `docs/experiments.md`.
5. If it changed a decision: `docs/decisions.md` updated.