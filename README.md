# CapstoneProject_AIinHealthcare

This project analyzes gene mutation data for leukemia classification using machine learning models (Random Forest, XGBoost, TabNet).

## Quick Start

1. **Prepare Data**
   - Place your `.maf.gz` files in the `data/` directory.
   - Ensure `disease_ids.csv` is present in the root directory.

2. **Preprocess Data**
   - Run the preprocessing notebook or script to generate the feature matrix:

     ```sh
     python src/preprocess.py
     ```

   - This creates `combined_mutation_matrix.csv`.

3. **Run Models**
   - For Random Forest/XGBoost: use `model_tests.ipynb` or run scripts in `src/`.
   - For TabNet: run `tabnet.py` (requires PyTorch and pytorch-tabnet).
   - For final evaluation and predictions: use `final_model.ipynb` for the complete workflow and final results.

4. **Feature Selection**
   - Use `src/feature_selection.py` to select top genes and evaluate model performance.

5. **Results & Visualizations**
   - Output plots and results are saved in the `visualizations/` folder or directly included in the notebook.

## File Overview

- `data/` — Raw mutation files
- `disease_ids.csv` — Disease labels
- `src/` — Source code (preprocessing, feature selection, models)
- `tabnet.py` — TabNet model script
- `model_tests.ipynb` — Model comparison notebook
- `preprocess.ipynb` — Data processing notebook
- `final_model.ipynb` — Final model pipeline and results

## Notes

- For TabNet, install extra dependencies:

  ```sh
  pip install pytorch-tabnet torch
  ```

- For XGBoost, install:

  ```sh
  pip install xgboost
  ```

## Citation

If you use this code, please cite appropriately.