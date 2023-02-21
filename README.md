# Multiplex Graph Fusion methods
This is the official repository for the paper: `Representation learning in multiplex graphs:
Where and how to fuse information?` (see: [https://arxiv.org/abs/2402.17906](https://arxiv.org/abs/2402.17906))

We utilize DVC for pipeline management and data versioning:
```bash
# Setup venv
python3 -m venv .venv
source .venv/bin/activate

# On a machine with CPU only
pip install -r requirements-cpu.txt 

# On a machine with GPU
pip install -r requirements-gpu.txt


# Check pipeline status
dvc status

# Pull precomputed models and embeddings
dvc pull

# Reproduce a specific stage
dvc repro -s -f <name-of-stage>@<stage-variant>
```

You can find full metrics for all the downstream tasks in file: `data/metrics_summary.txt`
and the results from the paper in file: `data/paper_metrics_summary.txt`.

Hyperparameter search scopes are defined in the `experiments/configs` directory
in `hps.yaml` files (placed in directories for specific models). The best chosen
hyperparameters for each model are given in `embed.yaml` files.