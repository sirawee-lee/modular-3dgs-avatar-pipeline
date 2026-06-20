# Modular Neural Pipelines for Text-to-3DGS Avatar Generation

> **Modular Neural Pipelines for Text-to-3DGS Avatar Generation: System Implementation and Preliminary Optimization**
>
> Napat Leesaksakul · Nahathai Wonganawat · Sirawee Wutinarongtrakul · Buachompoo Kaewchuay · Cheng-Hsin Hsu
>
> *National Tsing Hua University, Hsin-Chu, Taiwan*
>
> [ImmerCom '26](https://immersivecommconference.org/), October 30, 2026 · Austin, Texas, USA

Generate a photorealistic animated 3D human avatar from a **text prompt or spoken voice** — no motion-capture hardware, no manual animation, no 3D graphics expertise required.

```
Audio / Text → Whisper STT → LLM Normalization → MDM → Coord. Rotation → HUGS → 2D Video
```

Built on [HUGS: Human Gaussian Splats](https://arxiv.org/abs/2311.17910) (CVPR 2024) and [MDM: Human Motion Diffusion Model](https://arxiv.org/abs/2209.14916).

---

## Abstract

This paper presents an end-to-end modular neural pipeline that orchestrates speech recognition, Large Language Model (LLM)-based prompt normalization, diffusion-based motion generation, and 3D Gaussian Splatting (3DGS) avatar rendering into a single, yet modularized workflow. To bridge the heterogeneous coordinate systems between the motion generator and the avatar renderer, we design a rotation connector to ensure proper upright and camera-facing alignments. Furthermore, to mitigate the computational and storage bottlenecks inherent to streaming heavy per-frame 3D Gaussians over resource-constrained networks, we implement and evaluate a preliminary optimization approach using keyframe subsampling.

---

## Pipeline Overview

The system consists of **six sequential modules** (Figure 4 in the paper):

```
┌────────────────────────────────────────────────────────────────────┐
│  Audio ──► Speech-to-Text ──► Text                                 │
│            (OpenAI Whisper)                                        │
│                                                                    │
│  Text ───► Input Generator                                         │
│            (LLM Normalization: Ollama/llama3.2)                    │
│                   │                                                │
│                   ▼                                                │
│            Motion Generator                                        │
│            (MDM — Human Motion Diffusion Model)                    │
│                   │ SMPL                                           │
│                   ▼                                                │
│            Coordinates Converter                                   │
│            (RX=+90°, RZ=+180°  MDM Y-up → HUGS Z-up)               │
│                   │ SMPL                                           │
│                   ▼                                                │
│  Avatar ──► 3DGS Generator                                         │
│  Templates  (HUGS — Human Gaussian Splats)                         │
│                   │ 3DGS                                           │
│                   ▼                                                │
│            3DGS Renderer ──────────────────────────► 2D Video      │
│            (Three.js / Web-based)                                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## System Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3080 Ti (tested) — ≥10 GB VRAM recommended |
| CUDA | 12.2 driver / 11.8 toolkit |
| OS | Ubuntu 20.04 / 22.04 |
| Conda | Two isolated environments: `hugs` and `mdm` |
| RAM | ≥32 GB recommended |
| Storage | ≥50 GB — per-frame PLY ≈101.7 MB; 1 baseline sequence ≈12.2 GB |

---

## Pre-Flight Checklist

Before running the pipeline, verify you have everything. Each item is marked **[REQUIRED]** or **[OPTIONAL]**:

| Item | Required for | Status check |
|------|-------------|--------------|
| `hugs` conda env | All stages | `conda env list \| grep hugs` |
| `mdm` conda env | Motion generation | `conda env list \| grep mdm` |
| MDM repo at `~/motion-diffusion-model` | Motion generation | `ls ~/motion-diffusion-model` |
| MDM checkpoint `model000750000.pt` | Motion generation | `ls ~/motion-diffusion-model/save/humanml_enc_512_50steps/` |
| `data/smpl/SMPL_NEUTRAL.pkl` | HUGS rendering | `ls data/smpl/` |
| `data/neuman/dataset/bike/` | HUGS rendering | `ls data/neuman/dataset/` |
| `output/pretrained_models/bike/` | HUGS rendering | `ls output/pretrained_models/` |
| `pip install ollama` (in `hugs` env) | LLM normalization | `conda run -n hugs pip show ollama` |
| `pip install openai-whisper` | **[OPTIONAL]** Speech input | `conda run -n hugs pip show openai-whisper` |
| HiggsAudio v2 | **[OPTIONAL]** TTS readback | `conda run -n hugs pip show higgs-audio` |

---

## Setup (Step-by-Step)

### Step 1 — Clone both repositories

Clone this repo and the MDM repo **into the same parent directory**, then symlink MDM to the home directory where the pipeline expects it:

```bash
git clone <this-repo> modular-3dgs-avatar-pipeline
git clone https://github.com/GuyTevet/motion-diffusion-model
ln -s $(pwd)/motion-diffusion-model ~/motion-diffusion-model
```

> **Why the symlink?** The pipeline script defaults to `~/motion-diffusion-model`. You can skip the symlink and always pass `--mdm_repo /path/to/motion-diffusion-model` instead.

### Step 2 — Create the HUGS conda environment

```bash
cd modular-3dgs-avatar-pipeline
source scripts/conda_setup.sh
```

This script:
- Creates the `hugs` conda env (Python 3.8, PyTorch 1.13.1, pytorch3d)
- Clones and installs the CUDA Gaussian Splatting kernels (`diff-gaussian-rasterization`, `simple-knn`) into `submodules/`
- Installs all Python dependencies from `requirements.txt`
- Installs the `ollama` Python client (needed for LLM prompt normalization)

### Step 3 — Create the MDM conda environment

```bash
cd ~/motion-diffusion-model
conda env create -f environment.yml
conda activate mdm
pip install -e .
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
```

Download the MDM checkpoint (750k iterations, the one used in this paper) from [Google Drive](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view) and place it at:

```
~/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt
```

### Step 4 — Install the Ollama LLM (for prompt normalization)

The LLM normalization stage — which maps raw speech like *"uh, like someone kind of doing a happy dance"* to `"a person doing a happy dance"` — requires Ollama running locally.

```bash
# 1. Install the Ollama service
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the llama3.2 model (≈2 GB)
ollama pull llama3.2
```

> The `ollama` Python client is installed automatically by `conda_setup.sh` (Step 2). Only the service and model download are needed here.

### Step 5 — Download the SMPL body model

> **SMPL requires free registration.** Go to [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de), register, and download **SMPL for Python** (`SMPL_python_v.1.1.0.zip`).

Extract and copy the model files into the repo:

```bash
# After extracting the SMPL zip:
mkdir -p data/smpl
cp /path/to/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl data/smpl/SMPL_NEUTRAL.pkl
cp /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl      data/smpl/SMPL_MALE.pkl
cp /path/to/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl      data/smpl/SMPL_FEMALE.pkl
```

Expected result:

```
data/smpl/
├── SMPL_NEUTRAL.pkl
├── SMPL_MALE.pkl
└── SMPL_FEMALE.pkl
```

### Step 6 — Download NeuMan dataset and HUGS pretrained models

```bash
cd modular-3dgs-avatar-pipeline
source scripts/prepare_data_models.sh
```

This downloads from Apple ML Research servers:
- NeuMan dataset (6 scenes) → `data/neuman/dataset/`
- Pretrained HUGS checkpoints (6 scenes) → `output/pretrained_models/`

Expected result after download:

```
data/neuman/dataset/
├── bike/   citron/   jogging/   lab/   parkinglot/   seattle/

output/pretrained_models/
├── bike/   citron/   jogging/   lab/   parkinglot/   seattle/
    └── human_final.pth  +  scene_final.pth  (each scene)
```

### Step 7 — (Optional) Install speech I/O dependencies

Needed only if you want to use `--speech-input`, `--browser-input`, `--audio-file`, or `--speech-output`:

```bash
conda activate hugs

# STT — Whisper speech recognition
pip install openai-whisper sounddevice scipy

# TTS — HiggsAudio v2 text-to-speech
git clone https://github.com/boson-ai/higgs-audio
cd higgs-audio && pip install -e .
```

### Step 8 — Verify the setup

```bash
conda activate hugs
cd modular-3dgs-avatar-pipeline

python -c "
from hugs.cfg.constants import SMPL_PATH, NEUMAN_PATH
import os
print('SMPL:  ', 'OK' if os.path.exists(SMPL_PATH) else 'MISSING — recheck Step 5')
print('NeuMan:', 'OK' if os.path.exists(NEUMAN_PATH) else 'MISSING — recheck Step 6')
import sys; sys.path.insert(0, 'scripts')
from speech_io import normalize_prompt
print('Ollama:', 'OK')
"
```

All three lines should print `OK` before running the full pipeline.

---

## Quick Start

```bash
conda activate hugs
cd modular-3dgs-avatar-pipeline

python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --scene bike \
  --out_root ./output_text2hugs \
  --center
```

The final video will be saved to `./output_text2hugs/<timestamp>_a_person_jumps/final/result.mp4`.

---

## All Usage Examples

### Text prompt → avatar video

```bash
python scripts/run_text2hugs.py \
  --prompt "a person does a latin dance" \
  --scene bike \
  --out_root ./output_text2hugs \
  --center
```

### Speech input — local microphone

```bash
python scripts/run_text2hugs.py \
  --speech-input \
  --out_root ./output_text2hugs \
  --center
```

### Speech input — browser (for remote/AnyDesk/VNC sessions)

```bash
python scripts/run_text2hugs.py \
  --browser-input \
  --out_root ./output_text2hugs \
  --center
# Opens a recording page at http://localhost:9876
```

### Speech input — pre-recorded audio file

```bash
python scripts/run_text2hugs.py \
  --audio-file recording.wav \
  --out_root ./output_text2hugs \
  --center
```

### With TTS readback (HiggsAudio v2)

```bash
python scripts/run_text2hugs.py \
  --speech-input \
  --speech-output \
  --out_root ./output_text2hugs \
  --center
```

### Keyframe subsampling — 4× storage reduction (paper Table 5)

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./output_text2hugs \
  --subsample-k 4
```

### Dry run — print all commands without executing

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./test \
  --dry_run
```

---

## Arguments Reference

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | — | Text description of the motion |
| `--out_root` | *(required)* | Root output directory |
| `--scene` | `bike` | Background scene (see table below) |
| `--center` | `False` | Zero-center root translation |
| `--tz` | `1.0` | Depth offset — distance in front of camera |
| `--tx` | `0.0` | Horizontal offset |
| `--ty` | `0.0` | Vertical offset |
| `--seed` | `10` | MDM random seed |
| `--steps` | `50` | MDM diffusion steps (paper uses 50) |
| `--subsample-k` | `1` | Export every k-th frame: `1`=all, `2`=half, `4`=quarter |
| `--save_ply` | `False` | Save per-frame `.ply` Gaussian Splat files |
| `--bg_color` | `white` | Background for human-only mode (`white`/`black`) |
| `--dry_run` | `False` | Print commands without executing |

### Paths

| Argument | Default | Description |
|----------|---------|-------------|
| `--mdm_repo` | `~/motion-diffusion-model` | Path to MDM repository |
| `--mdm_py` | `~/anaconda3/envs/mdm/bin/python` | MDM environment Python executable |
| `--hugs_repo` | *(this repo)* | Path to this repository |
| `--hugs_py` | *(current Python)* | HUGS environment Python executable |

### Speech I/O

| Argument | Default | Description |
|----------|---------|-------------|
| `--speech-input` | `False` | Record from local mic (Whisper STT) |
| `--browser-input` | `False` | Record from browser page (remote sessions) |
| `--audio-file` | — | Pre-recorded audio file to transcribe |
| `--speech-output` | `False` | Read status aloud (HiggsAudio v2 TTS) |
| `--tts-save-wav` | — | Save TTS audio to this WAV file |
| `--whisper-model` | `base` | Whisper model size: `tiny/base/small/medium/large` |
| `--refine-prompt` | `False` | Extra LLM cleanup pass on top of auto-normalization |
| `--ollama-model` | `llama3.2` | Ollama model for prompt normalization |
| `--record-duration` | `8.0` | Microphone recording duration in seconds |

---

## Available Scenes

| Scene | Description |
|-------|-------------|
| `bike` | Outdoor — person near a bicycle (paper default) |
| `citron` | Indoor scene |
| `jogging` | Outdoor jogging path |
| `lab` | Lab environment |
| `parkinglot` | Outdoor parking lot |
| `seattle` | Outdoor urban scene |

Pretrained checkpoints live in `output/pretrained_models/<scene>/human_final.pth`.

---

## Pipeline Stages

```
Speech / Text Prompt
       │
       ▼  [Whisper STT — optional, requires openai-whisper]
  Raw Transcript / Text
       │
       ▼  [LLM Normalization — requires ollama + llama3.2]
  "a person [motion]"
       │
       ▼  Stage 1: MDM Motion Generation  (~13.6 s)
  results.npy
       │
       ▼  Stage 2: SMPL Extraction  (MDM extract_smpl_params.py)  (~49.6 s)
  hugs_smpl_original.npz
       │
       ▼  Stage 3: Coordinate Rotation  (RX=+90°, RZ=+180°)  (~0.15 s)
  hugs_smpl_upright.npz
       │
       ▼  Stage 4: HUGS Rendering  (3D Gaussian Splatting)  (~369.6 s)
  anim_*.mp4  +  anim_ply/*.ply  (if --save_ply)
       │
       ▼  Stage 5: Copy to output  (~11.2 s)
  final/result.mp4
       │
       ▼  [HiggsAudio TTS readback — optional]
```

Total pipeline time: **≈444 s (≈7.4 min)** on RTX 3080 Ti.

---

## Output Structure

```
output_text2hugs/<timestamp>_<prompt>/
├── mdm_out/
│   ├── mdm.log
│   └── extract_smpl.log
├── smpl_npz/
│   └── hugs_smpl_original.npz       # Raw SMPL from MDM (Y-up coordinate space)
├── rotated_npz/
│   └── hugs_smpl_upright.npz        # Converted to HUGS Z-up space
├── hugs_logs/
│   └── hugs.log
├── final/
│   ├── result.mp4
│   └── anim_ply/                    # Per-frame posed .ply files (if --save_ply)
├── benchmark_timing.json            # Per-stage wall-clock timing
├── benchmark_timing.csv
└── run_record.json                  # Full run metadata and parameters
```

---

## Experimental Results (from the paper)

### LLM Prompt Normalization (Table 3)

Cosine similarity between raw Whisper transcripts and LLM-normalized prompts
using `all-MiniLM-L6-v2` sentence embeddings:

| Raw Input | Normalized Prompt | Similarity |
|-----------|-------------------|------------|
| generate a person jumping | a person jumping | 0.854 |
| show me someone running | a person running | 0.683 |
| show me a young man do a squat | a person doing a squat | 0.696 |
| can you do a dancing? | a person dancing | 0.688 |
| uh, like someone kind of doing a happy dance | a person doing a happy dance | 0.704 |
| like, you know, someone stretching their arms | a person stretching their arms | 0.792 |
| **Overall Average** | | **0.736** |

### Coordinate Alignment (Table 4)

| r_x | r_z | θ_tilt | f_y | Visual outcome |
|-----|-----|--------|-----|----------------|
| 0° | 90° | 90° | 0 | Avatar lying on side, facing sideways |
| +90° | 0° | 0° | −1 | Avatar standing, facing **away** from camera |
| 0° | +180° | 90° | 0 | Avatar lying on side, facing sideways |
| **+90°** | **+180°** | **0°** | **+1** | **Avatar standing, facing camera ✓** |

### Pipeline Latency (Figure 5, RTX 3080 Ti)

| Stage | Mean Time | % of Total |
|-------|-----------|------------|
| 1. MDM Motion Generation | 13.6 s | 3.1% |
| 2. SMPL Parameter Extraction | 49.6 s | 11.2% |
| 3. Coordinate Rotation | 0.15 s | <1% |
| 4. HUGS Rendering | 369.6 s | **83.2%** |
| 5. Video & PLY Extraction | 11.2 s | 2.5% |
| **Total** | **≈444 s (≈7.4 min)** | |

### Storage and Streaming Under Subsampling (Table 5)

Per-frame `.ply` ≈ 101.7 MB; 120-frame sequence at 20 fps:

| Strategy | k | Frames | Total Size | Render Time | Bandwidth |
|----------|---|--------|------------|-------------|-----------|
| Baseline | 1 | 120 | 12.2 GB | 570.5 s | 16.3 Gbps |
| Subsampling | 2 | 60 | 6.1 GB | 215.0 s | 8.1 Gbps |
| **Subsampling** | **4** | **30** | **3.0 GB** | **124.3 s** | **~4.0 Gbps** |

At k=4: **4× reduction** in storage and render time with no loss in motion coverage (client reconstructs intermediate poses via linear interpolation).

---

## Repository Structure

```
modular-3dgs-avatar-pipeline/
├── main.py                          # HUGS training/rendering entry point
├── export_hugs_to_ply.py            # Export HUGS model to .ply
├── benchmark_gaussians.py           # Gaussian count vs render-time analysis
├── compose_multiple_humans.py       # Multi-human scene composition
│
├── scripts/
│   ├── run_text2hugs.py             # End-to-end pipeline orchestrator
│   ├── rotate_hugs_motion_v2.py     # Standalone coordinate rotation utility
│   ├── speech_io.py                 # Whisper STT + HiggsAudio TTS + LLM normalization
│   ├── evaluate.py                  # Quantitative evaluation metrics
│   ├── evaluate_llm_normalization.py# LLM normalization similarity evaluation
│   ├── plot_llm_normalization.py    # Plot Table 3 results
│   ├── plot_pipeline_latency_v2.py  # Plot Figure 5 latency breakdown
│   ├── plot_subsampling.py          # Plot Table 5 subsampling cost
│   ├── benchmark_root_drift.py      # Root translation drift benchmark
│   ├── conda_setup.sh               # Create the hugs conda environment
│   └── prepare_data_models.sh       # Download NeuMan dataset + HUGS checkpoints
│
├── cfg_files/
│   ├── release/neuman/              # Standard HUGS rendering configs
│   └── ablation/neuman/             # Ablation study configs
│
├── data/
│   ├── smpl/                        # SMPL body model files (manual download required)
│   │   ├── SMPL_NEUTRAL.pkl
│   │   ├── SMPL_MALE.pkl
│   │   └── SMPL_FEMALE.pkl
│   ├── neuman/dataset/              # NeuMan dataset (downloaded by prepare_data_models.sh)
│   └── custom_motions/              # Example pre-computed motion NPZ files
│
├── output/
│   └── pretrained_models/           # HUGS checkpoints (downloaded by prepare_data_models.sh)
│       └── <scene>/human_final.pth  + scene_final.pth
│
└── hugs/                            # HUGS model code (from apple/ml-hugs)
    ├── cfg/                         # Config and constants (SMPL_PATH, NEUMAN_PATH)
    ├── datasets/                    # NeuMan dataset loader
    ├── models/                      # HUGS model, SMPL layer, triplane MLP
    ├── renderer/                    # Gaussian renderer
    ├── trainer/                     # Training loop
    └── utils/                       # Rotation, image, graphics utilities
```

---

## Troubleshooting

**`AssertionError: Path data/smpl/SMPL_NEUTRAL.pkl does not exist`**
→ Complete Step 5. SMPL files must be downloaded manually from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) (free registration required).

**`ModuleNotFoundError: No module named 'ollama'`**
→ Run `conda activate hugs && pip install ollama` (Step 4, item 3).

**`❌ MDM repository not found: ~/motion-diffusion-model`**
→ Either create the symlink from Step 1 (`ln -s /path/to/motion-diffusion-model ~/motion-diffusion-model`) or always pass `--mdm_repo /path/to/motion-diffusion-model`.

**`FileNotFoundError: hugs_smpl_original.npz not produced`**
→ Check `mdm_out/extract_smpl.log`. Verify `sample/extract_smpl_params.py` exists inside the MDM repo.

**`❌ --speech-input requires openai-whisper`**
→ Run `conda activate hugs && pip install openai-whisper sounddevice scipy` (Step 7).

**Avatar is horizontal / sideways in output video**
→ The coordinate rotation was not applied. Ensure `--center` is passed and the rotation script ran without errors. Check `rotated_npz/rotate.log`.

---

## Related Work

| Model | Role in Pipeline | Reference |
|-------|-----------------|-----------|
| [HUGS](https://github.com/apple/ml-hugs) | 3DGS avatar rendering | Kocabas et al., CVPR 2024 |
| [MDM](https://github.com/GuyTevet/motion-diffusion-model) | Text-to-motion generation | Tevet et al., arXiv 2022 |
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech-to-text | Zezario et al., ICME 2024 |
| [HiggsAudio v2](https://github.com/boson-ai/higgs-audio) | Text-to-speech readback | Boson AI |
| [Ollama / llama3.2](https://ollama.com) | LLM prompt normalization | Touvron et al., 2023 |
| [SMPL](https://smpl.is.tue.mpg.de/) | Parametric human body model | Loper et al., TOG 2015 |
| [GaussianAvatar](https://huliangxiao.github.io/GaussianAvatar) | 3DGS human avatar | Hu et al., CVPR 2024 |
| [SplattingAvatar](https://initialneil.github.io/SplattingAvatar) | 3DGS human avatar | Shao et al., CVPR 2024 |
| [MotionDiffuse](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html) | Text-to-motion | Zhang et al., TPAMI 2024 |
| [EMAGE](https://pantomatrix.github.io/EMAGE/) | Co-speech gesture generation | Liu et al., CVPR 2024 |

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{leesaksakul2026modular,
  title     = {Modular Neural Pipelines for Text-to-3DGS Avatar Generation:
               System Implementation and Preliminary Optimization},
  author    = {Leesaksakul, Napat and Wonganawat, Nahathai and
               Wutinarongtrakul, Sirawee and Kaewchuay, Buachompoo and
               Hsu, Cheng-Hsin},
  booktitle = {Proceedings of the 1st International Workshop on
               Immersive Communication (ImmerCom '26)},
  year      = {2026},
  address   = {Austin, Texas, USA},
  month     = {October}
}
```

Please also cite the underlying models:

```bibtex
@inproceedings{kocabas2024hugs,
  title     = {{HUGS}: Human Gaussian Splats},
  author    = {Kocabas, Muhammed and Chang, Jen-Hao Rick and Gabriel, James
               and Tuzel, Oncel and Ranjan, Anurag},
  booktitle = {CVPR},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.17910}
}

@article{tevet2022human,
  title   = {Human Motion Diffusion Model},
  author  = {Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan
             and Cohen-Or, Daniel and Bermano, Amit Hacohen},
  journal = {arXiv preprint arXiv:2209.14916},
  year    = {2022}
}
```

---

## Licenses

| Component | License |
|-----------|---------|
| Pipeline scripts (`scripts/`) | MIT — see [LICENSE_PIPELINE](LICENSE_PIPELINE) |
| HUGS model code (`hugs/`) | Apple Sample Code License — see [LICENSE](LICENSE) |
| Gaussian-splatting kernel | Inria / MPII — **research use only** |
| SMPL body model | MPI non-commercial research license |
| Third-party dependencies | See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) |

> The HUGS model code and Gaussian-splatting kernel are licensed for **non-commercial research use only**.
