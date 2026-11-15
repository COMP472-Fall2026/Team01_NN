# COMP472 Project — Team 01

Image recognition and sentiment classification for **COMP472, Fall 2026**.
The project compares four classifiers on the supplied numerical features and uses
k-Means to investigate how the discovered groups correspond to the known classes.

- [Kaggle competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia)
- [Quan's Kaggle notebook](https://www.kaggle.com/code/quandmai267/quanmai-knn-clustering-ipynb) — verified execution: **Version 13**

## Team and responsibilities

| Member | Responsibility |
|---|---|
| **Tarek Barake** | Decision Tree and Random Forest implementations and experiments |
| **Quan Mai** | k-Nearest Neighbors and k-Means clustering implementations and experiments |
| **Youssef Jedidi** | Feed-forward Neural Network implementation and experiments |
| **All members** | Compare validation results, select a model for each dataset, and prepare the report and presentation |

Each owner evaluates their classifiers on **both datasets**, records the settings
and results, and generates test predictions if their model is selected. Decision
Tree, Random Forest, and k-NN use implementations from scratch; PyTorch layers and
optimizers are permitted for the Neural Network.

## Repository contents

```text
Team01_NN/
├── QuanMai_KNN_Clustering.ipynb  # k-NN and k-Means on both datasets
├── README.md                   # Project overview, results, and reproduction
└── .gitignore                  # Local environments, credentials, data, and outputs
```

This checkout currently contains Quan's notebook. The tree/forest and Neural
Network notebooks can be added by their owners. Their results are not reported
here until the corresponding experiments are available and verified.

## Datasets

| Dataset | Labeled examples | Hidden test examples | Features | Labels |
|---|---:|---:|---:|---|
| Images | 20,000 | 5,000 | 500 | Class IDs `0–49` |
| Sentiment | 12,000 | 5,000 | 768 | `0`: negative, `1`: neutral, `2`: positive |

Download the competition files from Kaggle:

```text
image_train.csv
image_test.csv
sentiment_train.csv
sentiment_test.csv
sample_submission.csv
```

The data consists of pre-extracted numerical features. Competition datasets and
account credentials are kept outside version control.

## Run the notebook on Kaggle

1. Join the competition and accept its rules to access the data.
2. Import [QuanMai_KNN_Clustering.ipynb](QuanMai_KNN_Clustering.ipynb) into a Kaggle notebook, or open Quan's linked notebook if you have access.
3. Attach the competition data. The first code cell expects the files under:

   ```text
   /kaggle/input/competitions/comp-472-project-fall-2026-concordia/
   ```

4. Use a CPU session and run all cells from top to bottom. The notebook limits numerical operations to four CPU threads.
5. Inspect the inline tables and figures. Download the two test-prediction CSVs from:

   ```text
   /kaggle/working/part_b/image_knn_candidate_predictions.csv
   /kaggle/working/part_b/sentiment_knn_candidate_predictions.csv
   ```

Each CSV contains **5,000 rows** with columns `id,label`. Experiment tables and
figures remain inline; the notebook does not export experiment CSVs or image files.

The Kaggle Python environment supplies NumPy, pandas, Matplotlib, scikit-learn,
IPython, and threadpoolctl. The first cell prints package versions for reproduction.
For local execution, use a separate copy with `DATA_DIR` and `OUTPUT_DIR` set to
local folders containing the same competition files.

## Shared validation convention

Use the original CSV row order and a **stratified 80/20 split with seed 42**,
separately for each dataset:

```python
train_idx, val_idx = train_test_split(
    range(len(train_df)),
    test_size=0.20,
    random_state=42,
    stratify=train_df['label'],
)
```

| Dataset | Training portion | Validation portion |
|---|---:|---:|
| Images | 16,000 | 4,000 |
| Sentiment | 9,600 | 2,400 |

Matching the source files, row order, seed, stratification, and splitting code
gives teammates the same validation rows. An 80/20 ratio alone does not ensure this.
Fit learned preprocessing on the training portion and apply that transformation
to validation examples. After selecting settings, refit preprocessing on all
labeled examples when producing hidden-test predictions.

## k-NN implementation and results

The NumPy implementation calculates distances in batches, selects the nearest
training examples, and combines their votes. It compares:

| Setting | Values tested |
|---|---|
| Image k | `1, 3, 5, 10, 20` |
| Sentiment k | `1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000` |
| Distance | Euclidean, cosine |
| Voting | Uniform, inverse-distance weighted |
| Preprocessing | Raw, standardization, L2 normalization, square-root + L2, PCA whitening |

Square-root preprocessing uses `sign(x) * sqrt(abs(x))`, preserving negative
sentiment-feature values. It is ordinary square root for nonnegative image
features. PCA whitening retains **256 components** for both datasets.
Scikit-learn supplies splitting, PCA, and evaluation utilities; k-NN and k-Means
are implemented in the notebook.

The search tests **100 image configurations** and **240 sentiment configurations**:

- Images: 5 k values × 2 distances × 5 preprocessing methods × 2 voting choices.
- Sentiment: 12 k values × 2 distances × 5 preprocessing methods × 2 voting choices.

Best tested configurations on the fixed **20% validation sets**:

| Dataset | Preprocessing | k | Distance | Voting | Validation accuracy |
|---|---|---:|---|---|---:|
| Images | Square-root + L2 | 1 | Euclidean | Uniform | **57.175%** |
| Sentiment | PCA whitening, 256 components | 4,000 | Cosine | Distance-weighted | **65.500%** |

The equally weighted mean validation accuracy is **61.3375%**. These are results
used for tuning; hidden-test accuracy is determined separately by Kaggle.

## k-Means clustering

Both datasets are clustered using their training portions, **L2 normalization**,
and **Euclidean distance**. Each configuration tries **three reproducible random
starts**, allows up to **200 iterations** per start, and selects the lowest
inertia. True labels enter only after fitting to interpret the clusters.

| Dataset | Tested cluster counts | Reference count | Reference purity | Reference ARI |
|---|---|---:|---:|---:|
| Images | 10, 25, 50, 75 | 50 | 10.99% | 0.0249 |
| Sentiment | 2, 3, 5, 10 | 3 | 38.50% | 0.0142 |

The inline displays compare cluster composition, class distributions, frequently
co-clustered class pairs, and the effect of changing k. Reference displays use the
known class counts; the highest tested ARI occurs at **k = 75 for images** and
**k = 10 for sentiment**. The notebook includes the complete comparison.

Purity describes class dominance inside clusters. Adjusted Rand Index (ARI)
measures agreement with true labels after correcting for chance: 1 means perfect
agreement and values around 0 indicate chance-level agreement. Both datasets show
weak overall class agreement under this clustering setup, with some coherent
image-class subsets. Clustering is exploratory analysis and does not generate
the competition predictions.

## Model selection and competition submission

The team selects the strongest validated classifier **separately for images and
sentiment**. The two winning classifiers may come from different members.

1. Each selected owner fixes their settings, uses all labeled examples, and exports predictions paired with their original test IDs.
2. Combine the selected image and sentiment predictions into **one 10,000-row CSV** with columns `id,label`.
3. Align rows by `id` against `sample_submission.csv`; each sample ID must appear exactly once. Image labels must be integers `0–49`, and sentiment labels must be integers `0–2`.
4. Upload the combined CSV to Kaggle after the team selects the models.

Quan's notebook contains a `make_submission` helper. `BUILD_SUBMISSION = False`
by default, so normal execution writes only the two per-task prediction files.
If k-NN is selected for both tasks, enable the flag to create `submission.csv`.
If a teammate wins a task, use that teammate's `id,label` table for that task.

Kaggle weights the two classification tasks equally:

```text
Overall score = (image accuracy + sentiment accuracy) / 2
```

## Report and presentation

The team report and presentation should explain all four classifiers, compare
their validation accuracies on both datasets, discuss hyperparameters and errors,
and interpret the clustering experiments. Each member documents their assigned
work and should understand the complete team submission.

Use only the supplied features and labeled training data. External labeled data,
new pretrained feature encoders, and hidden test labels are excluded. Course
deadlines and the academic code/report submission procedure are specified on
Moodle; Kaggle receives the prediction CSV for scoring.
