# Running LVNet on VIMA observation.

## Preparation

1. Install VIMA Bench
```
git clone https://github.com/vimalabs/VimaBench && cd VimaBench
pip install -e .
```
Also need to install other dependecy listed on `requirements.txt` if not done yet.


2. Download CLIPPY checkpoint.
```
wget https://github.com/kahnchana/clippy/releases/download/v1.0/clippy_5k.pt
```

3. Include GPT API at `config/config.py`


4. Change conda environment to run the pipeline on at `vima.py` variable `CONDA_ENV_NAME`

## Run
```
python vima.py
```