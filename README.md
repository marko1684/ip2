# Home Credit Default Risk — Credit Risk Prediction with ML

A machine learning project for predicting whether a loan applicant will default, built as a seminar paper for the **Data Mining 2** course at the Faculty of Mathematics, University of Belgrade.

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
git clone https://github.com/<your-username>/home-credit-default-risk.git
cd home-credit-default-risk
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

Open the notebooks in order (`01` → `04`) using Jupyter or VS Code:

```bash
jupyter notebook
```

## Key Results

| Model | ROC-AUC | Recall |
|---|---|---|
| LightGBM (optimized) | **0.787** | 0.690 |
| LightGBM (base) | 0.784 | 0.678 |
| XGBoost | 0.782 | 0.646 |
| Gradient Boosting | 0.780 | 0.044 |

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn
