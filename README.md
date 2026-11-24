# INFO-H420 - Project on Responsible AI

**Deadline: 10/12/2026**

**Group on UV: P.6**

Members:
- BELKHIRI Rida
- BENMASSAOUD El Mamoune
- DAHRI Mohamed
- EL HARROUTI Imad
- PEREIRA MARTINS Filipe
- SLIM Sami

## Setup

1. Download `adult.data` from the [UCI Adult dataset page](https://archive.ics.uci.edu/dataset/2/adult) and put it in a `data/` folder at the root of the repository (the folder is ignored by git).
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Git Bash on Windows
   # source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```
3. Optional, for Q2 only: install [Node.js](https://nodejs.org) and the Impact Framework with `npm install -g @grnsft/if`.
4. Run a notebook with `jupyter notebook` or `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`.

## Structure

| File | Content |
|---|---|
| `common.py` | Shared pipeline: loading, preprocessing, age binarization, train/validation/test split, classifier, evaluation. Every notebook imports it with `import common as c`. |
| `q1_classification.ipynb` | Q1: preprocessing choices, model selection, the classifier |
| `q2_sustainability.ipynb` | Q2: SCI measurement (CodeCarbon) and carbon-aware simulation (Impact Framework) |
| `sci_manifest.yml`, `sci_output.yaml`, `q2_sci_scenarios.png` | Q2: Impact Framework manifest, its output and the scenario chart |

## Conventions

- Always use `c.load_data()` and `c.split_data(df)` so that all classifiers share the same data and split (seed 42, 60/20/20).
- Protected attributes `age` and `sex` are binary: 1 = privileged group (age >= 37, male).
- Q2 measurements depend on the hardware: the reported values come from one machine (Intel Core i7-12650H). Do not rerun and commit `q2_sustainability.ipynb` from another computer.

## Progress

| Question | Status |
|---|---|
| Q1 Classification | Done |
| Q2 Sustainability | Done |
| Q3 Fairness | To do |
| Q4 Privacy | To do |
| Q5 Privacy and Fairness | To do |
| Q6 Explainability | To do |
| Q7 Explainability and LLMs | To do |
| Q8 Free exploration | To do |