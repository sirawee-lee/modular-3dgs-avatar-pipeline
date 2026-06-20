# Third-Party Licenses

This repository builds on top of several open-source and research-licensed
projects. You must comply with each project's license when using this code.

---

## HUGS — Human Gaussian Splats

- **Source:** https://github.com/apple/ml-hugs
- **License:** Apple Sample Code License (see `LICENSE`)
- **Terms:** Research / non-commercial use. Redistribution requires retaining
  Apple's copyright notice. You may NOT relicense HUGS code under MIT or any
  other open license.

---

## MDM — Human Motion Diffusion Model

- **Source:** https://github.com/GuyTevet/motion-diffusion-model
- **License:** MIT
- **Terms:** Free for research and commercial use with attribution.
- **Note:** MDM is NOT included in this repo. Clone it separately and point
  `--mdm_repo` at it.

---

## SMPL Body Model

- **Source:** https://smpl.is.tue.mpg.de/
- **License:** MPI Perceiving Systems Research License (non-commercial, research only)
- **Terms:** You must register at the SMPL website and download the model files
  yourself. Redistribution of SMPL weights (`.pkl` files) is **prohibited**.
  This is why `data/smpl/` is in `.gitignore`.

---

## AMASS / MPI MoSH Motion Data

- **Source:** https://amass.is.tue.mpg.de/
- **License:** Non-commercial research license
- **Terms:** Must be downloaded directly from the AMASS website after accepting
  their terms. Do NOT redistribute.

---

## NeuMan Dataset

- **Source:** https://github.com/apple/ml-neuman
- **License:** Apple Sample Code License
- **Terms:** Research / non-commercial use. Downloaded by `scripts/prepare_data_models.sh`.

---

## HiggsAudio v2 (TTS)

- **Source:** https://github.com/bosonai/higgs-audio (submodule at `higgs-audio/`)
- **License:** Apache License 2.0
- **Terms:** Free for research and commercial use with attribution.
- **Model weights:** Downloaded automatically from HuggingFace (`bosonai/higgs-audio-v2-*`)
  on first use. Subject to HuggingFace model card terms.

---

## Whisper (STT)

- **Source:** https://github.com/openai/whisper
- **License:** MIT
- **Terms:** Free for research and commercial use with attribution.

---

## diff-gaussian-rasterization

- **Source:** https://github.com/graphdeco-inria/diff-gaussian-rasterization
  (submodule at `submodules/diff-gaussian-rasterization/`)
- **License:** Inria / MPI-IS Research License
- **Terms:** Non-commercial research use only.

---

## simple-knn

- **Source:** https://github.com/camaro1200/simple-knn
  (submodule at `submodules/simple-knn/`)
- **License:** MIT
