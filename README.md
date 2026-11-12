# hydrobench (protocol v2)

Code, data splits and reproduction instructions for:

> Forero, L., de Mello Jr., H.D., Garcia Lima, A.G., Talavera, A., Lazo Lazo, J.G., Pacheco, M.A.C. *Context length shapes the comparison between foundation and domain models for monthly inflow forecasting: a benchmark on 158 Brazilian reservoirs with held-out basins and years.* Submitted to the Journal of Hydrology.

The folder `energy/` holds the code of a companion study on the energy value of inflow forecasts (Energy Informatics manuscript).

The package runs every experiment of the paper (E1 to E10, plus the high- and low-flow diagnostics E11a), computes all metrics and statistical tests, and writes the tables and figures.

## Contents

| Path | What it is |
|---|---|
| `hydrobench/` | Python package: data loading and quality control, dependency groups and folds, causal windowing, models (neural, pooled, Chronos), metrics, statistics, report |
| `notebooks/protocol_v2_colab.ipynb` | Colab notebook that runs the whole protocol cell by cell (one cell per experiment) and writes the report |
| `rodar.py` | Same protocol from the command line (resumable) |
| `splits/folds_and_groups.csv` | Fold, dependency group and regime of each of the 158 basins, record start, and the highest correlation of each test basin with any basin in its training pool |
| `tests/` | Synthetic fixtures and a quick end-to-end test |
| `make_notebook.py` | Regenerates the notebook from the package |

## Data

Daily natural inflow for 161 hydropower sites, 1931 to 2024, is published by the Brazilian National Grid Operator (ONS). Monthly climate indices (ONI, SOI, PDO, AMO, NAO, AAO) are from NOAA. The fold assignments, including the dependency groups rebuilt from data before 2005, are in `splits/`; the derived monthly series and every result table of the paper are in `results/` (fixed-window and strict-grouping controls in `results/tables/controls_E9_E10/`, high and low flows in `results/tables/extremes_E11a/`). Raw predictions of every run (several hundred megabytes) are available from the corresponding author on request. Exact library versions and checkpoint commits are in `results/tables/environment_versions.csv` and `requirements-lock.txt`.

## Reproducing the results

Requirements: Python 3.10 or later, and a GPU with at least 16 GB for the foundation-model experiments (the results in the paper were produced on Google Colab GPUs).

```bash
pip install torch            # with CUDA, from pytorch.org
pip install -r requirements.txt
python rodar.py --dados path/to/daily_inflow.xlsx --indices path/to/indices --saida results_v2 --relatorio
```

Each run is saved under `results_v2/preds` as soon as it finishes, so an interrupted run resumes where it stopped. `--so-relatorio` rebuilds tables and figures from saved predictions only.

Fixed choices (all in `hydrobench/config.py`): five seeds (0 to 4) for every trained model; 3,000 AdamW steps, batch 256, learning rate 1e-3; context of 24 months for the main benchmark and 60, 120, 240 and 360 months for the equal-context experiments (E7, E8); training targets up to December 2004 and test targets from January 2005 to December 2024; dependency groups by average-linkage clustering of daily log-flow anomalies at r = 0.85; five folds stratified by regime.

Quick check on synthetic data (a few minutes, CPU):

```bash
cd tests && python -c "import fixtures as F, os; F.make_daily_xlsx('Vazoes_teste.xlsx'); os.makedirs('indices', exist_ok=True); F.make_indices('indices'); F.make_m4('m4.csv')" && python teste_rapido.py E1
```

## Energy study (Energy Informatics manuscript)

The folder `energy/` holds one Colab notebook per stage of the energy study, each self-contained (the package is embedded) and resumable, with the script that generates it. All inputs are downloaded from ONS open data (https://dados.ons.org.br) and the Central Bank of Brazil (IPCA series 433).

| Notebook | Stage | Paper |
|---|---|---|
| `energia_fase0` to `energia_fase2` | Matching of basins to plants, productivity check, inflow to natural energy inflow | Methods, "From inflow to energy" |
| `energia_fase3` to `energia_fase5` | Energy accuracy, price-weighted exposure, recalibrated and debiased forecasts | Table 2, Figures 2 and 3 |
| `energia_fase6` | Closed-loop decision experiment with causal information, hindsight optimum, inference, frontier, nested rule, sensitivity | Tables 1, 4 to 6, Figures 4 to 6 |
| `energia_fase7` | First version of the resampled histories (transported forecasts only); superseded by phase 8 | not used in the current version |
| `energia_fase8` | Adequacy criteria and calibrations, imputed capacity, native Chronos-2 quantiles with copula aggregation and calibration, quantile policies, PAR(p) and Chronos-2 re-run on 100 resampled histories | Tables 3, 7 to 10, Figures 7 and 8, Additional file 1 |

Settings used in the paper: checkpoint `amazon/chronos-2` (zero-shot, context up to 512 months, 21 quantiles); PAR(p) with order by BIC up to 6, refitted every December; causal bias correction over 60 target months (at least 12); demand factors 1.196 (thermal), 1.019 (storage) and 1.18 (compromise); deficit cost R$ 5,000/MWh at December 2024 prices; thermal availability 0.85. Seeds: block bootstrap seed 0 (12-month blocks, 2,000 replicates); resampled histories use `numpy.random.default_rng([0, h])` for history h with 3-year blocks; copula draws seed 0 (2,000 draws for the observed record, 1,000 per history), Gaussian copula on standardised log monthly inflow anomalies of 1931 to 2004 with eigenvalue floor 1e-6. The complete list is Table S4 of Additional file 1 of the manuscript.

## Licence

MIT. See `LICENSE`. Please cite the paper (`CITATION.cff`).

## Controls added in revision (E9, E10)

- `hydrobench/controls.py`: E9 varies the context from 24 to 360 months while holding the training, validation and test windows, the minibatch sequence and the 3,000 updates fixed; E10 repeats the benchmark with single-linkage dependency groups (r >= 0.85, 52 groups).
- `notebooks/controles_E9_E10.ipynb` (built by `make_controls_notebook.py`) runs both controls on a GPU in Colab and writes the tables `E9_*.csv` and `E10_*.csv`.
