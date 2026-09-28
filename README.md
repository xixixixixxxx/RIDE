# RIDE: RL-Induced Direction Extrapolation for On-Policy Distillation

Code for the ICLR 2027 submission *The Teacher Is a Direction, Not a Destination: Extrapolating
RL-Induced Representation Residuals in On-Policy Distillation*. Anonymous supplementary material.

RIDE is implemented on top of the open-source OPRD / OPD training stack (verl v0.7.0, FSDP + vLLM).
The code base is inherited from those projects; **all RIDE-specific code is marked `[RIDE]`** and is
summarised in the table below. Internally the method is sometimes referred to by its development
codename in configuration keys (`rep_extrapolation_*`); these are the RIDE options.

## What is in this archive

| Path | Role |
| --- | --- |
| `run_ride.sh` | Main launcher: RIDE, and the matched OPD top-1 / top-16 baselines (`DISTILLATION_MODE`). Defaults to the Qwen3-4B pair; used for the Llama-3.2-3B and Phi-4-mini pairs by changing the model paths. |
| `run_ride_r1_distill_1p5b.sh` | RIDE on the R1-Distill-1.5B -> JustRL-1.5B pair (Table 1 first block, Table 2, Table 3, Figures 4-7). `REP_LAMBDA=1` reproduces OPRD exactly. |
| `run_opd_top1.sh`, `run_opd_top16.sh` | Thin wrappers around `run_ride.sh` for the output-space teacher-matching baselines. |
| `run_lambda_sweep.sh` | Extrapolation-coefficient sweep (Table 3 / Appendix B). |
| `run_grpo_teacher.sh` | Trains the RL teacher from a base checkpoint (JustRL-style GRPO) for the Qwen3-4B, Llama-3.2-3B and Phi-4-mini pairs. |
| `run_eval_only.sh` | Evaluates a base model / teacher (Avg@16 on AIME24, AIME25, AIMO) with the same validation path used during training. |
| `rep_distillation.sh`, `on_policy_distillation.sh` | Generic OPRD / OPD drivers inherited from the OPRD release (kept for reference; the `run_*.sh` launchers above are self-contained). |
| `verl/verl/utils/rep_distillation.py` | **[RIDE]** target construction `build_extrapolated_target` (Eq. 6), `compute_extrapolation_lambda`, extrapolation diagnostics, plus the OPRD representation extraction / masked loss. |
| `verl/verl/workers/fsdp_workers.py` | **[RIDE]** the teacher ("reward model") worker loads the frozen pre-RL checkpoint (`base_model_path`), runs the second forward pass on the same student prefix, forms `h* = lambda h_T + (1 - lambda) h_B` in place and ships one hidden-state tensor to the trainer (Appendix E). |
| `verl/verl/workers/actor/dp_actor.py` | Student update: masked multi-layer regression toward the shipped target (Eq. 7). Unchanged from OPRD except for the target. |
| `verl/tests/utils/test_rep_distillation.py` | Unit tests for the target construction and lambda schedule (33 tests). |
| `scripts/val/` | Offline evaluation (vLLM generation + grading, JustRL pipeline) and the AIME24 / AIME25 / AIMO prompt sets. |
| `datasets/` | DAPO-Math-17K training prompts (parquet) and the three evaluation sets (`test_data/`). |
| `requirements-verified.txt` | Exact package versions of the environment used for all experiments. |

Everything not listed (upstream verl docs/examples/recipes, SFT tooling, raw evaluation outputs) has
been removed to keep the archive small. Model weights are not included; see "Models".

## Mapping between the paper and the code

| Paper | Code |
| --- | --- |
| RL-induced residual `Delta = h_T - h_B` on identical student prefixes (Sec. 4.2) | `fsdp_workers.py`: teacher and base forward passes on the same `input_ids`; `build_extrapolated_target` |
| Target `h* = h_T + (lambda - 1) Delta` (Eq. 6) | `build_extrapolated_target(teacher, base, lambda)` in `rep_distillation.py` |
| Objective, all `L` layers, last `k = 2000` positions (Eq. 7) | `REP_DISTILLATION_LAYERS=all`, `REP_DISTILLATION_POSITIONS=last_k`, `REP_DISTILLATION_LAST_K=2000` |
| `lambda = 1` recovers OPRD exactly | `REP_LAMBDA=1` (the base model is then not loaded) |
| Loss coefficient scaled by `lambda^-2` (Remark E.1) | `REP_DISTILLATION_COEF` defaults to `1/lambda^2` in the launchers |
| Single global constant `lambda = 1.25` | `REP_LAMBDA=1.25 REP_LAMBDA_SCHEDULE=constant` (the `warmup_cosine` schedule is an unused option) |
| OPD top-1 / top-16 baselines | `DISTILLATION_MODE=opd-top1` / `opd-top16` (`LOG_PROB_TOP_K`, `only_stu`, `student_p`) |
| Hyperparameters (Table 4) | Section 3 of `run_ride.sh` and the Hydra overrides in Section 5 |

The ExOPD baseline (output-space reward extrapolation, Yang et al. 2026) was run with the authors'
official implementation using the same prompts, rollout settings, optimizer schedule, `lambda = 1.25`
and the same pre-RL reference checkpoint; it is not part of this archive.

## Environment

The experiments used Python 3.12, PyTorch 2.8.0 (CUDA 12.8), vLLM 0.11.0, transformers 4.56.1,
Ray 2.58.0, FlashAttention 2.8.1 and FlashInfer 0.3.1 on 4x NVIDIA H200. `requirements-verified.txt`
lists the exact versions; `verl/requirements.txt` is the upstream (looser) list.

```bash
conda create -n ride python=3.12 -y && conda activate ride
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
python -m pip install vllm==0.11.0
python -m pip install -r requirements-verified.txt
# FlashAttention wheel for torch 2.8 / cp312 (see https://github.com/Dao-AILab/flash-attention/releases)
python -m pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
python -m pip install flashinfer-python==0.3.1
cd verl && python -m pip install -e . --no-deps && cd ..
# sanity check of the RIDE target construction
PYTHONPATH=$PWD/verl python -m pytest -q verl/tests/utils/test_rep_distillation.py   # 33 passed
```

## Models

Set `AGENTRL_ROOT` to a directory that contains a `models/` folder with the checkpoints
(or pass `STUDENT_MODEL`, `TEACHER_MODEL`, `BASE_MODEL` explicitly). In every pair the student is
initialised from the base checkpoint and `BASE_MODEL == STUDENT_MODEL`.

| Pair | Base = student init (`STUDENT_MODEL`, `BASE_MODEL`) | RL teacher (`TEACHER_MODEL`) |
| --- | --- | --- |
| R1-Distill-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `hbx/JustRL-DeepSeek-1.5B` (public) |
| Qwen3-4B | `Qwen/Qwen3-4B` | Just-Qwen3-4B: trained with `run_grpo_teacher.sh` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | Just-Llama-3.2-3B: `run_grpo_teacher.sh` |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | Just-Phi-4-mini: `run_grpo_teacher.sh` |

The three RL teachers we trained will be released with the camera-ready version.

## Reproducing the tables

All launchers expect 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3` by default) and 500 optimizer steps;
every run writes checkpoints, terminal logs and per-validation JSON under `OUTPUT_ROOT`.

```bash
export AGENTRL_ROOT=/path/to/workspace        # contains models/

# Table 1, R1-Distill-1.5B pair
REP_LAMBDA=1.25 bash run_ride_r1_distill_1p5b.sh      # RIDE
REP_LAMBDA=1    bash run_ride_r1_distill_1p5b.sh      # OPRD

# Table 1, Qwen3-4B pair (RIDE, OPD top-1, OPD top-16)
bash run_ride.sh
bash run_opd_top1.sh
bash run_opd_top16.sh
REP_LAMBDA=1 bash run_ride.sh                         # OPRD

# Table 1, Llama-3.2-3B / Phi-4-mini pairs: same launcher, different checkpoints
STUDENT_MODEL=$AGENTRL_ROOT/models/Llama-3.2-3B \
BASE_MODEL=$STUDENT_MODEL TEACHER_MODEL=$AGENTRL_ROOT/models/Just-Llama-3.2-3B \
OUTPUT_ROOT=$AGENTRL_ROOT/runs/ride_llama3b bash run_ride.sh

# Table 3 / Appendix B: extrapolation-coefficient sweep on the R1-Distill-1.5B pair
bash run_lambda_sweep.sh                              # lambda in {0.5, 0.75, 1, 1.15, 1.25, 1.35, 1.5, 2}

# RL teachers (JustRL-style GRPO) for the Qwen3 / Llama / Phi pairs
ACTOR_MODEL_PATH=$AGENTRL_ROOT/models/Qwen3-4B MODE=full CONFIRM_FULL=1 bash run_grpo_teacher.sh
```

Direction controls (Table 2) reuse `run_ride_r1_distill_1p5b.sh`: the reversed direction is
`REP_LAMBDA=0.75`; the mismatched-origin control passes `BASE_MODEL=<Qwen2.5-Math-1.5B-Instruct>`;
the random-direction and trajectory-mismatched controls are not exposed as switches and were run by
modifying the target construction in `fsdp_workers.py` (Gaussian direction of equal norm; base forward
on a separately generated trajectory).

Smoke test (about 5 minutes on 4 GPUs):

```bash
VAL_BEFORE_TRAIN=False TOTAL_STEPS=2 SAVE_FREQ=1000 REP_LAMBDA=1.25 bash run_ride.sh
```

## Evaluation

Validation during training reports Avg@16 (16 samples, temperature 0.7, top-p 0.95, up to 15,360
response tokens) on AIME 2024 (30 problems), AIME 2025 (30) and AIMO / AMC 2022-2023 (83), graded by
exact match on the boxed answer (`verl/verl/utils/reward_score/ttrl_math`). Saved checkpoints can also
be evaluated offline with the JustRL pipeline:

```bash
cd scripts/val/eval
python gen_vllm.py      # set MODEL_NAMES and workers
python grade.py
```

`scripts/val/eval/evaluate_checkpoint.py` / `run_eval_10bench.sh` evaluate a checkpoint on the
ten benchmarks shipped in `scripts/val/data`; `summarize_training_val.py` aggregates the
per-validation JSON files written during training.

## Acknowledgements

This code base extends the open-source OPRD implementation (Yang et al. 2026), which in turn extends
the OPD release (Li et al. 2026) built on verl. Token-level OPD, the evaluation pipeline and the
training infrastructure follow their design; RIDE adds the frozen pre-RL forward pass and the
extrapolated representation target described above.
