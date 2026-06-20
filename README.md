# NTHU Avatar — Text/Speech-to-Animation Pipeline

> **[PAPER TITLE — fill in before making repo public]**
> *[AUTHORS] · [Workshop/Conference] 2026*

Generate a rendered 3D human avatar video from a **text prompt or spoken voice**, using **MDM** (Motion Diffusion Model) for motion generation and **HUGS** (Human Gaussian Splats) for photorealistic rendering. This work extends the HUGS pipeline with a fully automated orchestration layer and an optional speech interface (STT via Whisper + TTS via HiggsAudio v2), enabling hands-free avatar animation from natural language commands.

```
Speech / Text  →  MDM  →  SMPL Motion  →  HUGS Rendering  →  result.mp4
```

Based on [HUGS: Human Gaussian Splats](https://arxiv.org/abs/2311.17910) (CVPR 2024) and [MDM: Human Motion Diffusion Model](https://arxiv.org/abs/2209.14916).

---

## System Requirements

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3080 Ti (tested) · ≥ 10 GB VRAM recommended |
| CUDA | 12.2 (driver) · 11.8 (toolkit) |
| OS | Ubuntu 20.04 / 22.04 |
| Conda | Two environments: `hugs` · `mdm` |
| RAM | ≥ 32 GB recommended |
| Storage | ≥ 50 GB (datasets + pretrained models; per-frame PLY ≈ 101 MB/frame, 1 sequence ≈ 12.2 GB) |

---

## Requirements

- [Conda](https://docs.conda.io/en/latest/)
- Two separate Conda environments: `hugs` and `mdm`
- The [MDM repo](https://github.com/GuyTevet/motion-diffusion-model) cloned separately

---

## Setup

### 1. Clone both repos

```bash
git clone <this-repo> ml-hugs-NTHUavatar
git clone https://github.com/GuyTevet/motion-diffusion-model
```

### 2. Set up the HUGS environment

```bash
cd ml-hugs-NTHUavatar
source scripts/conda_setup.sh
```

### 3. Set up the MDM environment

```bash
cd motion-diffusion-model
conda env create -f environment.yml
conda activate mdm
pip install -e .
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
```

Download the MDM 50-step checkpoint from [Google Drive](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view) and place it at:

```
motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt
```

### 4. Download HUGS data and pretrained models

```bash
source scripts/prepare_data_models.sh
```

This downloads:
- SMPL body model → `data/smpl/`
- NeuMan dataset → `data/neuman/`
- Pretrained HUGS checkpoints → `output/pretrained_models/`

---

## Usage

All commands below should be run with the `hugs` conda environment active.

### Text prompt

```bash
conda activate hugs
python scripts/run_text2hugs.py \
  --prompt "a person does a latin dance" \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center
```

### Speech input — microphone

```bash
python scripts/run_text2hugs.py \
  --speech-input \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center
```

### Speech input — browser (for remote/AnyDesk/VNC sessions)

```bash
python scripts/run_text2hugs.py \
  --browser-input \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center
# Opens recording page at http://localhost:9876
```

### Speech input — pre-recorded audio file

```bash
python scripts/run_text2hugs.py \
  --audio-file recording.wav \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center
```

### With TTS readback (HiggsAudio v2)

```bash
python scripts/run_text2hugs.py \
  --speech-input \
  --speech-output \
  --mdm_repo /path/to/motion-diffusion-model \
  --mdm_py /path/to/envs/mdm/bin/python \
  --out_root ./output_text2hugs \
  --center
```

### Dry run (test configuration without executing)

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./test \
  --dry_run
```

---

## Arguments

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | — | Text description of the motion |
| `--out_root` | *(required)* | Root output directory |
| `--scene` | `bike` | Background scene (see table below) |
| `--center` | `False` | Zero-center root translation |
| `--tz` | `1.0` | Depth offset — controls how far in front of camera |
| `--tx` | `0.0` | Horizontal offset |
| `--ty` | `0.0` | Vertical offset |
| `--seed` | `10` | MDM random seed |
| `--steps` | `50` | Number of MDM diffusion steps |
| `--dry_run` | `False` | Print commands without executing |

### Paths

| Argument | Default | Description |
|----------|---------|-------------|
| `--mdm_repo` | `~/motion-diffusion-model` | Path to MDM repository |
| `--mdm_py` | `~/anaconda3/envs/mdm/bin/python` | Python executable for the MDM environment |
| `--hugs_repo` | *(this repo)* | Path to this HUGS repository |
| `--hugs_py` | *(current Python)* | Python executable for the HUGS environment |

### Speech I/O

| Argument | Default | Description |
|----------|---------|-------------|
| `--speech-input` | `False` | Record from local mic (Whisper STT) |
| `--browser-input` | `False` | Record from browser page (for remote sessions) |
| `--audio-file` | — | Path to a pre-recorded audio file to transcribe |
| `--speech-output` | `False` | Read prompt and status aloud (HiggsAudio v2 TTS) |
| `--tts-save-wav` | — | Save TTS audio to this WAV file |
| `--whisper-model` | `base` | Whisper model size: `tiny/base/small/medium/large` |
| `--refine-prompt` | `False` | Use Ollama LLM to clean up the transcribed prompt |
| `--ollama-model` | `llama3.2` | Ollama model for prompt refinement |
| `--record-duration` | `8.0` | Mic recording length in seconds |

---

## Available Scenes

| Scene | Description |
|-------|-------------|
| `bike` | Outdoor — person near a bicycle |
| `citron` | Indoor scene |
| `jogging` | Outdoor jogging path |
| `lab` | Lab environment |
| `parkinglot` | Outdoor parking lot |
| `seattle` | Outdoor urban scene |

Pretrained checkpoints are in `output/pretrained_models/<scene>/`.

---

## Pipeline Stages

```
Speech / Text Prompt
       │
       ▼  [Whisper STT — optional]
  Text Prompt
       │
       ▼  [Ollama LLM normalization — optional]
  Normalized Prompt ("a person ...")
       │
       ▼  [1] MDM Motion Generation
  results.npy
       │
       ▼  [2] SMPL Extraction  (MDM's extract_smpl_params.py)
  hugs_smpl_original.npz
       │
       ▼  [3] Coordinate Rotation  (RX=+90°, RZ=+180°)
  hugs_smpl_upright.npz
       │
       ▼  [4] HUGS Rendering  (3D Gaussian Splatting)
  anim_*.mp4
       │
       ▼  [5] Copy to output
  final/result.mp4
       │
       ▼  [HiggsAudio TTS readback — optional]
```

---

## Output Structure

```
output_text2hugs/<timestamp>_<prompt>/
├── mdm_out/
│   ├── mdm.log
│   └── extract_smpl.log
├── smpl_npz/
│   └── hugs_smpl_original.npz    # Raw SMPL from MDM
├── rotated_npz/
│   └── hugs_smpl_upright.npz     # HUGS coordinate system
├── hugs_logs/
│   └── hugs.log
├── final/
│   └── result.mp4
├── benchmark_timing.json          # Per-stage timing
└── run_record.json                # Full run metadata
```

---

## Pipeline Latency

Measured on RTX 3080 Ti, averaged across multiple prompts:

| Stage | Mean Time |
|-------|-----------|
| 1. MDM Motion Generation | 13.6 s |
| 2. SMPL Parameter Extraction | 49.6 s |
| 3. Coordinate Rotation | 0.15 s |
| 4. HUGS Rendering | 369.6 s |
| 5. Video & PLY Extraction | 11.2 s |
| **Total** | **≈ 7.4 min** |

---

## Related Repos

| Repo | Purpose |
|------|---------|
| [apple/ml-hugs](https://github.com/apple/ml-hugs) | HUGS — Human Gaussian Splats (base model) |
| [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model) | MDM — motion generation from text |
| [openai/whisper](https://github.com/openai/whisper) | Whisper — speech-to-text |
| [bosonai/higgs-audio](https://github.com/bosonai/higgs-audio) | HiggsAudio v2 — text-to-speech |
| [SMPL Body Model](https://smpl.is.tue.mpg.de/) | SMPL — parametric human body model |

---

## Licenses

- **Pipeline scripts** (`scripts/`): MIT — see `LICENSE_PIPELINE`
- **HUGS model code**: Apple Sample Code License — see `LICENSE`
- **Third-party dependencies**: see `THIRD_PARTY_LICENSES.md`

---

## Speech I/O Dependencies (optional)

```bash
# STT — Whisper speech recognition
pip install openai-whisper sounddevice scipy

# TTS — HiggsAudio v2
pip install -e higgs-audio/
# Model weights are downloaded automatically from HuggingFace on first use

# Prompt refinement via local LLM
# 1. Install Ollama: https://ollama.ai
# 2. Pull a model:
ollama pull llama3.2
```

---

## Citation

```bibtex
@inproceedings{kocabas2024hugs,
  title={{HUGS}: Human Gaussian Splatting},
  author={Kocabas, Muhammed and Chang, Jen-Hao Rick and Gabriel, James and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={CVPR},
  year={2024},
  url={https://arxiv.org/abs/2311.17910}
}

@inproceedings{tevet2022human,
  title={Human Motion Diffusion Model},
  author={Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan and Cohen-Or, Daniel and Bermano, Amit Hacohen},
  booktitle={ICLR},
  year={2023},
  url={https://arxiv.org/abs/2209.14916}
}
```

## License

This project is released under the [LICENSE](LICENSE) terms from the original HUGS repository.
