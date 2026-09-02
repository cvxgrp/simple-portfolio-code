# Simple Dynamic Stock/Bond/Gold Portfolios

This repository contains the code used to reproduce the results in [*Simple Dynamic
Stock/Bond/Gold Portfolios*](https://stanford.edu/~boyd/papers/stock_bond_gold_portfolios.html).

## Reproduce the results

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone this repository and enter it:

   ```bash
   git clone https://github.com/cvxgrp/simple-portfolio-code.git
   cd simple-portfolio-code
   ```

3. Install the dependencies:

   ```bash
   uv sync
   ```

4. Create `keys/fred_api.key` containing your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).
5. Run the scripts in order:

   ```bash
   uv run python scripts/1_download_data.py
   uv run python scripts/2_download_evaluation_data.py
   uv run python scripts/3_download_distributions.py
   uv run python scripts/4_generate_alphas.py
   uv run python scripts/5_run_portfolios.py
   uv run python scripts/6_make_results.py
   uv run python scripts/7_hyperparameter_sensitivity.py
   uv run python scripts/8_walkforward_sensitivity.py
   uv run python scripts/9_statistical_inference.py
   uv run python scripts/10_lagged_information.py
   uv run python scripts/11_risk_based_benchmarks.py
   uv run python scripts/12_covariance_sensitivity.py
   uv run python scripts/13_black_litterman.py
   uv run python scripts/14_cost_sensitivity.py
   ```

Most scripts should finish in less than a minute, although
`12_covariance_sensitivity.py` may take a couple of minutes. Running the full
sequence should take roughly 10 minutes or less in total.

If you encounter an error while reproducing the results, please open a GitHub
issue.

Generated tables are written to `output/tables/` and figures to `output/plots/`.
