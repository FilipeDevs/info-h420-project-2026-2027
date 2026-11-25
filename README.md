# INFO-H420 - Project on Responsible AI

**Deadline: 10/12/2026**

**Group on UV: ??**

Members:
- BELKHIRI Rida
- BENMASSAOUD El Mamoune
- DAHRI Mohamed
- EL HARROUTI Imad
- PEREIRA MARTINS Filipe
- SLIM Sami 

## Project layout

- [docs/Project.pdf](docs/Project.pdf) contains the assignment.
- [notebooks/q1_classification.ipynb](notebooks/q1_classification.ipynb) implements Q1.
- [src/info_h420/common.py](src/info_h420/common.py) contains the shared data and model pipeline.
- `data/` holds the local Adult dataset.

## Setup with uv on Windows

Run these commands from the project root:

```powershell
uv venv --python 3.14
uv pip install -r requirements.txt -e .
```

Download `adult.data` from UCI as described in the assignment and place it in
`data/adult.data`. Create the `data/` folder if needed. Then open the notebook:

```powershell
.\.venv\Scripts\jupyter.exe lab notebooks/q1_classification.ipynb
```
