# Home Credit Default Risk — Credit Risk Prediction with ML

A project for predicting whether a loan applicant will default, built as a seminar paper for the **Data Science 2** course at the Faculty of Mathematics, University of Belgrade.

The project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (307k applications, 7 relational tables) and progresses through four phases: basic preprocessing → advanced feature engineering → clustering & boosting → hyperparameter tuning. Nine classifiers are evaluated, with **LightGBM** achieving the best result (ROC-AUC **0.787**).

## Project Structure

| Notebook | Description |
|---|---|
| `simple_solution.ipynb` | preprocessing, 7 classifiers on the base table (122 features) |
| `advanced_solution.ipynb` | Feature engineering from all 7 tables (~300 features), class balancing |
| `clustering_and_boosting.ipynb` | K-Means clustering, XGBoost & LightGBM |
| `hyperparameter_tuning.ipynb` | Two-phase RandomizedSearchCV for LightGBM |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/marko1684/ip2.git
cd ip2
```

### 2. Download the data

Download the dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSV files into the `raw_data/` directory.

### 3. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the notebooks

Open the notebooks in order (`simple` → `advanced` → `clustering/boosting` → `hyperparameter tuning`) using Jupyter or VS Code:

```bash
jupyter notebook
```
