# DLTPKT

Release-oriented main experiment code for **DLTPKT(DLTPKT: Interpretable Knowledge Tracing for Dynamic Learning Transfer Process)** on the processed
**Statics** dataset.

## Repository Structure

```text
DLTPKT-Master/
|-- checkpoints/
|   `-- DLTPKT_statics_best.pth  # legacy reference checkpoint
|-- data/statics/
|   |-- attribute/
|   |-- encode/
|   |-- graph/ques_skill.csv
|   `-- train_test/
|       |-- train_all_feature.txt
|       `-- test_all_feature.txt
|-- dltpkt/
|   |-- data.py
|   |-- metrics.py
|   `-- model.py
|-- docs/architecture.md
|-- evaluate.py
|-- requirements.txt
`-- train.py
```

## Environment

Python 3.9 or later is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build for GPU training. The training script
defaults to CUDA; pass `--device cpu` when running without a GPU.

## Train

```bash
python train.py
```

Useful options:

```bash
python train.py --device cuda --epochs 200 --batch-size 64
python train.py --device cpu --epochs 1 --batch-size 8
```

Training writes the best compatible checkpoint and a one-row result table to
`outputs/`.

## Evaluate

After training:

```bash
python evaluate.py
```

To evaluate another compatible checkpoint:

```bash
python evaluate.py --checkpoint path/to/DLTPKT_statics_best.pth
```

The file under `checkpoints/` is retained as a legacy artifact. It predates the
current structural-role projection and mastery-readout modules and is not
compatible with the current model.
