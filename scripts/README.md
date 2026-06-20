# Scripts Reference

## Main Pipeline

### `run_text2hugs.py` — end-to-end text/speech → video

```bash
# Text prompt
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center --tz 1.0

# Speech from microphone
python scripts/run_text2hugs.py \
  --speech-input \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center

# Speech from browser (remote sessions)
python scripts/run_text2hugs.py \
  --browser-input \
  --out_root ./output_text2hugs \
  --center
# → opens http://localhost:9876

# Dry run — print commands without executing
python scripts/run_text2hugs.py \
  --prompt "a person walks" \
  --out_root ./test \
  --dry_run
```

### `rotate_hugs_motion_v2.py` — standalone coordinate rotation

Rotates only the root joint (global_orient + transl), preserving relative body pose.
Use this to fix avatar orientation when the body pose is correct but the whole figure is horizontal.

```bash
python scripts/rotate_hugs_motion_v2.py \
  --input motion.npz \
  --output motion_upright.npz \
  --rx 90 --rz 180 \
  --center --tz 1.0
```

The standard MDM → HUGS conversion is always `--rx 90 --rz 180`.

---

## Evaluation & Plotting

| Script | Purpose |
|--------|---------|
| `evaluate.py` | Quantitative evaluation metrics |
| `evaluate_llm_normalization.py` | Evaluate LLM prompt normalization quality |
| `plot_llm_normalization.py` | Plot normalization similarity results |
| `plot_pipeline_latency_v2.py` | Per-stage latency breakdown chart |
| `plot_subsampling.py` | Gaussian subsampling size analysis |
| `benchmark_root_drift.py` | Root translation drift benchmark |

---

## Setup

```bash
# Create the HUGS conda environment
source scripts/conda_setup.sh

# Download SMPL models, NeuMan dataset, and pretrained HUGS checkpoints
source scripts/prepare_data_models.sh
```
