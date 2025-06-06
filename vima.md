# Running LVNet on VIMA observation.

Install VIMA Bench
```
git clone https://github.com/vimalabs/VimaBench && cd VimaBench
pip install -e .
```
Install other dependecy listed on `requirements.txt`.


Download CLIPPY checkpoint.
```
wget https://github.com/kahnchana/clippy/releases/download/v1.0/clippy_5k.pt
```

Include GPT API at `config/config.py`
