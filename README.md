# COMP472 Project — Team 01

Image recognition and sentiment classification for **COMP472, Fall 2026**. The project compares k-Nearest Neighbors (k-NN), Decision Tree, Random Forest, and a feed-forward Neural Network on the supplied numerical features. k-Means provides an additional analysis of how feature-based groups correspond to true classes.

## 1. Team and responsibilities

| Member | Assigned work | Status |
|---|---|---|
| Quan Mai | k-NN classification and k-Means clustering | Implemented and documented below |
| Tarek Barake | Decision Tree and Random Forest | Implemented and documented in Section 8 |
| Youssef Jedidi | Neural networks (image CNN, sentiment MLP) and combined prediction notebook | Implemented and documented in Section 9 |
| All members | Compare validation results, select a model for each task, combine predictions, and prepare the report/presentation | Complete after all classifiers are evaluated |


## 2. Repository contents and file purposes

### Source and documentation

| File or directory | Purpose |
|---|---|
| [README.md](README.md) | File inventory, setup, preprocessing, training/evaluation, prediction, and team integration instructions. |
| [quanmai-partb.ipynb](quanmai-partb.ipynb) | Data loading/splitting, k-NN implementation, preprocessing comparisons, the added image setup, model selection, prediction export, and k-Means on both datasets. Includes saved Kaggle Version 3 results. |
| [COMP472-FFNN-CNN.ipynb](COMP472-FFNN-CNN.ipynb) | Image CNN and sentiment MLP: preprocessing, training, validation curves, and neural-network submission export. Includes saved Kaggle GPU outputs. |
| [COMP472-Group1-Prediction-Only.ipynb](COMP472-Group1-Prediction-Only.ipynb) | Final combined pipeline (image k-NN + sentiment MLP): validation check, full-data refit, and competition `submission.csv` export. |


## 3. Quan

### 3.1. Required data

Create a new notebook under "Code" section in Kaggle Competition to directly access the data provided by the Competition.

| Task | Labeled rows | Hidden test rows | Feature columns | Labels |
|---|---:|---:|---|---|
| Images | 20,000 | 5,000 | `feature_0` through `feature_499` | Integers `0–49` |
| Sentiment | 12,000 | 5,000 | `feature_0` through `feature_767` | `0`: negative, `1`: neutral, `2`: positive |

Training CSVs contain `id`, feature columns, and `label`; test CSVs contain `id` and feature columns. The loader selects features by their explicit names, converts them to numeric arrays, and keeps IDs as strings. IDs and labels are excluded from the feature matrix. Keep the original CSV row order.

### 3.2. Execute on Kaggle

1. Open [quanmai-partb](quanmai-partb.ipynb)
2. Attach the competition data. In the first code cell, confirm:

   ```python
   DATA_DIR = Path('/kaggle/input/competitions/comp-472-project-fall-2026-concordia')
   OUTPUT_DIR = Path('/kaggle/working/part_b')
   ```

   If Kaggle attaches the data elsewhere, set `DATA_DIR` to the directory containing all five CSVs.
3. Select a **CPU** session. Numerical operations are limited to four CPU threads; a GPU and internet access are not required for execution.
4. Restart the session and run all cells from top to bottom, or use **Save Version → Save & Run All** for a completed run with saved outputs.

[COMP472-FFNN-CNN.ipynb](COMP472-FFNN-CNN.ipynb) and [COMP472-Group1-Prediction-Only.ipynb](COMP472-Group1-Prediction-Only.ipynb) use the same competition data path (`data_dir` in Cell 1) and write their prediction CSVs to `/kaggle/working`. Run all cells top to bottom. Use a **CPU** session for the prediction-only notebook; a **GPU** session is recommended for the CNN notebook (it automatically uses CUDA when available and falls back to CPU otherwise).

## 4. Data preprocessing: execution order and transformations

### 4.1. Load and split the data

Run **Section 1: Data and the shared validation split** first. It imports libraries, sets `SEED = 42`, loads both datasets, and creates the shared split:

```python
train_idx, val_idx = train_test_split(
    range(len(train)),
    test_size=0.2,
    random_state=42,
    stratify=train['label'],
)
```

| Task | Training portion | Validation portion | Separate hidden test set |
|---|---:|---:|---:|
| Images | 16,000 | 4,000 | 5,000 |
| Sentiment | 9,600 | 2,400 | 5,000 |

All teammates should use the same source files, row order, stratification, seed, and split code so validation scores are directly comparable.

### 4.2. Define and apply preprocessing

Run **Section 2: k-NN from scratch**, then the **first code cell in Section 3**. These define `normalize`, `preprocess`, `root_covariance_features`, and `comparison_preprocess`. The following Section 3 search cell applies them automatically for every tested configuration. The following preprocessing techniques are implemented inside the notebook

| Method | Transformation | Applied to |
|---|---|---|
| Raw (`raw`) | Keep the original feature values. | Both tasks |
| Standardization (`standard`) | Subtract each feature's training mean and divide by its training standard deviation; replace a zero standard deviation with 1. | Both tasks |
| L2 normalization (`l2`) | Divide each nonzero example by its Euclidean length; zero vectors stay zero. | Both tasks |
| Square-root + L2 | Apply `sign(x) * sqrt(abs(x))`, then normalize each row. Negative sentiment-feature signs are preserved. | Both tasks |
| PCA whitening (256) | Fit PCA on training rows, retain 256 components, and whiten the retained directions. Apply the fitted transformation to queries. | Both tasks |
| Root ratios + covariance | Apply the image-specific sequence below. | Images only |

For **Root ratios + covariance**, the notebook:

1. Divides each image's feature magnitudes by its smallest nonzero magnitude, rounds the ratios, and takes square roots.
2. Subtracts the training mean, divides by `max(training_std, 1e-6) ** 0.2`, and L2-normalizes each row.
3. Learns covariance directions from the transformed training rows, weights each direction by `(eigenvalue + 0.1 * mean_eigenvalue) ** 0.2`, and L2-normalizes again. Both powers remain fixed at **0.2** in the current notebook.

## 5. Quan's k-NN: training, evaluation, and prediction

### 5.1. Fit and compare configurations

After the preprocessing definitions, run the remaining cells in **Section 3**. For every query, the implementation calculates distances in batches, retrieves neighbors, and combines their class votes. the notebook run kNN using a combination of different settings of preprocessing, number of k, distance calculation, and voting methods on both datasets.

| Search | k values | Distance | Voting | Preprocessing | Configurations |
|---|---|---|---|---|---:|
| Original image grid | `1, 3, 5, 10, 20` | Euclidean, cosine | Uniform, inverse-distance | Original five methods | `5 × 2 × 2 × 5 = 100` |
| Additional image sweep | `1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000` | Locally scaled cosine | Distance power (18) | Root ratios + covariance | `12 × 1 × 1 × 1 = 12` |
| Sentiment grid | `1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000` | Euclidean, cosine | Uniform, inverse-distance | Original five methods | `12 × 2 × 2 × 5 = 240` |

Totals: **112 image configurations**, **240 sentiment configurations**, and **352 configurations overall**. 


- **Locally scaled cosine:** multiply cosine distance by a scale for the candidate training image. Scales start from the average distance to its 10 nearest other training images and are refined five times. This neighborhood of 10 is separate from classification k. For normalized vectors, the score is `max(1 - query @ training_image, 1e-6) * scale`.
- **Distance power (18):** each selected neighbor receives `(closest_distance / its_distance) ** 18`. The closest neighbor has weight 1; farther neighbors rapidly lose influence. Weights are summed by class.

Table 1 reports the best configuration for each preprocessing method; Figure 1 shows the effect of k.

### 5.2. Evaluate and select the model

Run **Section 4: Best setup for each dataset** after the search. Use the results already calculated in Section 3 to select the overall winner for each dataset. Table 2: reports the winning preprocessing, k, distance, voting, and validation accuracy.
Figure 2: compares each setup's best accuracy and labels the winner for each dataset. 

and the error summary report the results. These scores come from the fixed 20% validation sets used for tuning

Verified results from the completed run:

| Task | Preprocessing | k | Distance | Voting | Validation accuracy |
|---|---|---:|---|---|---:|
| Images | Root ratios + covariance | 100 | Locally scaled cosine | Distance power (18) | **67.975%** |
| Sentiment | PCA whitening (256) | 4,000 | Cosine | Inverse-distance | **65.500%** |

The equally weighted mean is **66.7375%**

## 5. k-Means clustering: fit and interpret

Run **Section 5: k-Means clustering from scratch: images and sentiment** after the preceding sections. It uses the same training portions, **L2 normalization**, and **Euclidean distance**, independently of the selected classification preprocessing.

For each cluster count, the implementation initializes centers from random examples, alternates assignment and mean updates, and tries **three starts** with seed 42. Each start allows **200 iterations**, stopping when center movement is at most `1e-4 * max(norm(centers), 1)`. An empty cluster receives a random example. The start with the smallest **inertia** (sum of squared distances to assigned centers) is retained.

| Task | Tested cluster counts | Reference count | Reference purity | Reference ARI |
|---|---|---:|---:|---:|
| Images | `10, 25, 50, 75` | 50 | **10.99%** | **0.0249** |
| Sentiment | `2, 3, 5, 10` | 3 | **38.50%** | **0.0142** |

Tables 3–7 and Figures 3–5 show cluster sizes/composition, random starts, class distributions, pairwise class mixing, and changes with the number of clusters. Purity measures the proportion belonging to each cluster's largest true class, aggregated across clusters. Adjusted Rand Index (ARI) measures agreement with true labels after adjustment for chance. Labels are used **after fitting** for interpretation, never for initialization or cluster assignment.

Cluster IDs are arbitrary and need not correspond to class IDs. Results show weak overall class agreement under this setup, although some image groups are more coherent. Clustering is exploratory and does not generate competition predictions. Tables and plots remain inline; this section writes no CSVs or image files.

## 7. Team comparison and final application

Complete this comparison using each member's results on the shared validation rows. `TBD` means the notebook/results have not yet been provided.

| Classifier | Owner | Image validation accuracy | Sentiment validation accuracy | Implementation/results |
|---|---|---:|---:|---|
| k-NN | Quan | 67.975% | 65.500% | Current notebook, Kaggle Version 3 |
| Decision Tree | Tarek | 12.500% | 60.000% | [Comp 472 v9.ipynb](https://github.com/COMP472-Fall2026/Team01_NN/blob/main/Comp%20472%20v9.ipynb) |
| Random Forest | Tarek | 29.075% | 66.542% | [Comp 472 v9.ipynb](https://github.com/COMP472-Fall2026/Team01_NN/blob/main/Comp%20472%20v9.ipynb) |
| CNN (images) / MLP (sentiment) | Youssef | 46.300% | 66.542% | [COMP472-FFNN-CNN.ipynb](COMP472-FFNN-CNN.ipynb), Kaggle Version 3 GPU run |

For our final competition submission, [COMP472-Group1-Prediction-Only.ipynb](COMP472-Group1-Prediction-Only.ipynb) pairs Quan's image k-NN (**67.975%**) with Youssef's sentiment MLP (**66.375%** in the saved local CPU outputs; **66.250%** on Kaggle CPU, giving a **67.1125%** mean and a **0.684** public leaderboard score). Minor differences in sentiment MLP validation accuracy across runs (66.250%–66.542%) come from CPU vs. GPU and platform/library differences.

## 8. Tarek — Decision Tree and Random Forest

**Owner:** Tarek Barake

**Notebook:** [Comp 472 v9.ipynb](https://github.com/COMP472-Fall2026/Team01_NN/blob/main/Comp%20472%20v9.ipynb)

| Item | Details |
|---|---|
| Dependencies | NumPy, pandas, Matplotlib, joblib (optional), scikit-learn (for comparison tests) |
| Data and preprocessing | Kaggle image and sentiment CSV files; 64-bin quantile transformation fitted on training data |
| Validation split | Stratified 80/20 split, seed 42. Custom implementation; validation rows may differ from teammates' splits |
| Running instructions | Run notebook cells in order, with `QUICK_RUN = False` for full experiments |

### Decision Tree

- **Implementation:** CART Decision Tree implemented from scratch using NumPy. Supports Gini and entropy impurity, maximum depth, minimum split size, and minimum leaf size.
- **Training:** Hyperparameter tuning in Section 6, including depth, leaf size, split size, impurity criterion, and number of bins.
- **Evaluation:** Image accuracy **12.500%**; sentiment accuracy **50.917%**. Detailed results and overfitting analysis in Sections 9–11.
- **Application:** Not selected for the final Kaggle submission.
- **Retraining:** Run the notebook from the beginning; no saved model checkpoint.

### Random Forest

- **Implementation:** Ensemble of custom Decision Trees with random feature selection, optional bootstrap sampling, and probability averaging.
- **Training:** Hyperparameter tuning in Section 8, including feature selection, tree depth, leaf size, split size, criterion, and bootstrap. Final forests contain **200 trees** each.
- **Evaluation:** Image accuracy **29.075%**; sentiment accuracy **60.000%**. Comparison with Decision Tree in Section 9.
- **Application:** Not selected for the final Kaggle submission, as other team models achieved higher validation accuracies.
- **Retraining:** Run the notebook from the beginning; no saved model checkpoint.

**Summary:** Random Forest outperformed Decision Tree on both tasks. Full implementation, hyperparameter experiments, evaluation metrics, and visualizations are available in the linked notebook.

## 9. Youssef — Neural networks and combined prediction

- **[COMP472-FFNN-CNN.ipynb](COMP472-FFNN-CNN.ipynb):** Trains and evaluates the image CNN and sentiment MLP, plots learning curves, and exports neural-only predictions.
- **[COMP472-Group1-Prediction-Only.ipynb](COMP472-Group1-Prediction-Only.ipynb):** Runs our selected combination (image k-NN + sentiment MLP) for validation and generates the final competition submission.

### 9.1. Dependencies and preprocessing

Both notebooks use `numpy`, `pandas`, `torch`, and `scikit-learn` (plus `matplotlib` in `COMP472-FFNN-CNN.ipynb`), all pre-installed on Kaggle and tested locally on Python 3.11 (PyTorch 2.10, scikit-learn 1.8).

- **Split:** Stratified 80/20 split with `SEED = 42` (16,000/4,000 for images; 9,600/2,400 for sentiment), matching the shared split in Section 4.1 (`id` and `label` excluded from features; sorting by `id` preserves the same split).
- **Image preprocessing:** In `COMP472-FFNN-CNN.ipynb` (Cell 3), `StandardScaler` is fitted on training rows only. In `COMP472-Group1-Prediction-Only.ipynb` (Cell 2), the image k-NN uses the same root-ratio + covariance pipeline as Section 4.2, fitted on training rows for validation and on all labeled rows for test prediction.
- **Sentiment preprocessing:** Both notebooks pass the 768 supplied features directly to the MLP without extra scaling.

### 9.2. Architecture and training

| Setting | Image CNN | Sentiment MLP |
|---|---|---|
| Input → output | 500 features → 50 classes | 768 features → 3 classes |
| Architecture | BatchNorm1d → Dropout(0.3) → Linear(500, 4096) → reshape to 16×16×16 learned feature maps → two 3×3 Conv2d blocks (32 channels, second stride 2, BatchNorm2d) → Flatten → Linear(2048, 50) | Linear 768 → 300 → 64 → 3 |
| Activations | GELU after projection and each conv block | ReLU after both hidden layers |
| Regularization | Dropout 0.3 (input), 0.1 / 0.2 (2D after reshape and between conv blocks), 0.6 (before classifier); BatchNorm | Dropout 0.2 after second hidden layer |
| Loss / optimizer | CrossEntropyLoss / AdamW (`lr=3e-4`, `weight_decay=0.05`) | CrossEntropyLoss / AdamW (`lr=1e-3`, `weight_decay=0.01`) |
| Schedule / epochs | CosineAnnealingLR (`T_max=60`), 59 epochs | CosineAnnealingLR (`T_max=6`), 5 epochs |
| Batch size / seed | 128 / 42 | 64 / 42 |

### 9.3. How to run, evaluate, and apply

No saved model checkpoints are required; both notebooks retrain from scratch and run inference end to end.

**Neural comparison (`COMP472-FFNN-CNN.ipynb`, 7 cells):**
1. **Cell 1:** Load CSVs from `data_dir` and build the 80/20 splits.
2. **Cells 2–3:** Define the CNN/MLP models, scaling, and PyTorch training loop.
3. **Cells 4–5:** Train and evaluate the sentiment MLP (**66.542%** val) and image CNN (**46.300%** val).
4. **Cell 6:** Print the combined mean and plot training/validation learning curves. (On images, training accuracy with dropout disabled approaches 100% while validation stays near 46.3%, showing clear overfitting; on sentiment, training accuracy is tracked during training batches with dropout enabled.)
5. **Cell 7:** Refit both models on 100% of labeled data and write `/kaggle/working/comp472_cnn_submission.csv` (10,000 `id,label` rows aligned to `sample_submission.csv`).

**Final combined submission (`COMP472-Group1-Prediction-Only.ipynb`, 5 cells):**
1. **Cell 1:** Load image and sentiment train/test CSVs from `data_dir`.
2. **Cells 2–3:** Define the image k-NN and sentiment MLP training/inference functions.
3. **Cell 4:** Run the 80/20 validation check for both models (**67.975%** image, **66.375%** local CPU sentiment).
4. **Cell 5:** Refit on all 20,000 labeled image rows and 12,000 sentiment rows, verify ID alignment, and write `/kaggle/working/submission.csv` (10,000 `id,label` rows).
