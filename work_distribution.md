# COMP472 Project — Fall 2026
## Work distribution and team workflow

**Team:** Quan Mai, Youssef Jedidi, Tarek Barake   
**Working approach (team choice):** Each member develops and runs their own Google Colab notebook, saves it to a shared private GitHub repository, and reports their results. The team manually selects the strongest validated model for each dataset. Kaggle receives **one CSV containing predictions for both hidden test datasets**.  

---

## 1. Project overview

| Task | Labeled training samples | Hidden test samples | Features | Labels |
|---|---:|---:|---:|---|
| Image recognition | 20,000 | 5,000 | 500 | Integers `0–49` (50 classes) |
| Sentiment analysis | 12,000 | 5,000 | 768 | `0` negative, `1` neutral, `2` positive |

Each of the four required classifiers must be **implemented and evaluated on both datasets**:

- **k-Nearest Neighbors (k-NN):** Implement the core algorithm from scratch.
- **Decision Tree:** Implement the core algorithm from scratch.
- **Random Forest:** Implement the core algorithm from scratch, potentially reusing the team's Decision Tree implementation.
- **Feed-forward Neural Network:** Build and train the model ourselves; PyTorch layers and optimizers are permitted.

The team must also conduct **at least one clustering experiment** using one supplied dataset. The clustering result is for analysis/presentation, **not** the Kaggle score.

**Evaluation:** Kaggle uses classification accuracy, weighting image and sentiment equally:

`Overall score = (Image accuracy + Sentiment accuracy) / 2`

---

## 2. Ownership and work division

**`Owner` = responsible for implementing, experimenting, documenting, and delivering that work. `X` = not assigned.**

| Responsibility | Tarek | Quan | Youssef |
|---|:---:|:---:|:---:|
| Decision Tree implementation + experiments | **Owner** | X | X |
| Random Forest implementation + experiments | **Owner** | X | X |
| k-NN implementation + experiments | X | **Owner** | X |
| Clustering experiment (suggested: k-Means) | X | **Owner** | X |
| Neural Network implementation + experiments | X | X | **Owner** |
| Final notebook and reproducible trained model for assigned algorithm(s) | **Owner** | **Owner** | **Owner** |
| Validate assigned classifier(s) on **image and sentiment** datasets | **Owner** | **Owner** | **Owner** |
| Record accuracy, hyperparameters, observations, and error analysis for assigned model(s) | **Owner** | **Owner** | **Owner** |
| Compare validation accuracies and choose the best model **for each task** | **Joint** | **Joint** | **Joint** |
| Generate test predictions if own model is selected | **Owner if selected** | **Owner if selected** | **Owner if selected** |
| Prepare own report section and presentation material | **Owner** | **Owner** | **Owner** |

### Member A — Decision Tree + Random Forest

**Responsibilities**

- [ ] Implement Decision Tree from scratch: splitting, stopping rules, nodes, and predictions.
- [ ] Investigate tree depth, minimum split/leaf sample sizes, and Gini vs. entropy where feasible.
- [ ] Implement Random Forest from scratch using the Decision Tree code: bootstrapping, feature subsampling, and majority voting.
- [ ] Investigate number of trees, maximum depth, and features considered per split.
- [ ] Run **both algorithms on both datasets** using the agreed validation convention.
- [ ] Record training and validation accuracy, selected parameters, and noteworthy errors.
- [ ] Explain whether the forest generalizes better than a single tree and why.
- [ ] If one of these models wins a task, generate test predictions with their correct sample IDs.

**Member A deliverable:** `Member_A_Trees.ipynb`, containing **both complete implementations, trained-model workflow, experiments on both datasets, results, analysis, and reproducible test-prediction code**. Save final model checkpoints only if they are useful; a notebook that can retrain the model is also acceptable as a reproducibility artifact.

### Member B — k-NN + clustering

**Responsibilities**

- [ ] Implement k-NN from scratch: distances, nearest-neighbor selection, and voting.
- [ ] Investigate values of `k` (e.g., 1, 3, 5, 10, 20), distance metrics, scaling, and weighted/unweighted voting.
- [ ] Run k-NN on **both datasets** using the agreed validation convention.
- [ ] Conduct at least **one clustering experiment**. Suggested: k-Means on the sentiment embeddings.
- [ ] Interpret cluster membership, cluster sizes, and whether clusters correspond to known classes (use labels only **after** clustering for interpretation).
- [ ] Record validation accuracy, selected hyperparameters, error patterns, and clustering findings.
- [ ] If k-NN wins a task, generate test predictions with their correct sample IDs.

**Member B deliverable:** `Member_B_KNN_Clustering.ipynb`, containing **k-NN implementation and experiments for both datasets, clustering experiment, results, analysis, and reproducible test-prediction code**.


### Member C — Neural Network

**Responsibilities**

- [ ] Build and train a feed-forward Neural Network using PyTorch.
- [ ] For images, use **500 inputs / 50 outputs**; for sentiment, **768 inputs / 3 outputs**.
- [ ] Investigate some of: hidden layer count/width, activation, learning rate, optimizer, batch size, epochs, dropout, and weight decay.
- [ ] Run the Neural Network on **both datasets** using the agreed validation convention.
- [ ] Record validation accuracy, selected hyperparameters, and training/validation curves.
- [ ] Explain overfitting/underfitting and the impact of architecture or training choices.
- [ ] If the Neural Network wins a task, generate test predictions with their correct sample IDs.

**Member C deliverable:** `Member_C_NeuralNetwork.ipynb`, containing **model definition, training, experiments for both datasets, results, plots, analysis, and reproducible test-prediction code**. A saved PyTorch `state_dict` is optional if the notebook already explains how to retrain and obtain predictions.

---

## 3. Repository structure 

```text
COMP472-Project/
├── notebooks/
│   ├── Member_A_Trees.ipynb
│   ├── Member_B_KNN_Clustering.ipynb
│   └── Member_C_NeuralNetwork.ipynb
├── work_distribution.md
├── README.md                 # How to load the Kaggle data and run each notebook
├── requirements.txt          # Optional: package versions
└── .gitignore
```

- Every member writes the **algorithm and experimentation code inside their own notebook**.
- Each member should commit their own notebook to the shared private GitHub repository. Avoid overwriting each other's work.
- GitHub stores your work; **Kaggle only scores the prediction CSV**. Moodle/the instructor determines how the teaching team receives the reproducible code and report.

**Recommended notebook outline for each member:**

1. Imports, environment details, seeds.
2. Load provided data and identify feature columns, `id`, and training labels.
3. Make the agreed train/validation split.
4. Implement the assigned model(s).
5. Train, tune hyperparameters, and compare validation accuracy.
6. Show best settings, relevant plots/errors, and explanations.
7. Include a clearly marked **final prediction section** to generate test predictions **if the model is selected**.

---

## 4. Why use the same training/validation split?

**This is a team recommendation, not an extra course requirement.** It helps make model accuracy comparisons fair.

Suppose Member A evaluates on one set of 4,000 image examples and Member C evaluates on a different set of 4,000. One subset might be easier than the other. Their accuracy numbers would not be as directly comparable.

**Recommended low-effort convention:** Everyone loads the original Kaggle training CSVs in the same order and uses a **stratified 80/20 split with `random_state=42`**. This gives the same validation rows when the input row order and splitting code match.

| Task | Training portion | Validation portion | Kaggle test (separate) |
|---|---:|---:|---:|
| Image | 16,000 | 4,000 | 5,000 |
| Sentiment | 9,600 | 2,400 | 5,000 |

For example, with `train_test_split` from scikit-learn (allowed for data splitting):

```python
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    range(len(train_df)),
    test_size=0.20,
    random_state=42,
    stratify=train_df["label"],
)
```

Run this **separately** for `image_train.csv` and `sentiment_train.csv`. Everyone should use the same source CSV row ordering, parameters, and split code. **No separate split files or coordinator are needed**; the team just agrees on this convention.

Members can apply **different model-appropriate preprocessing**, such as normalization for k-NN or the Neural Network, but must fit any preprocessing on the training portion only and apply it to validation/test data. Never use hidden Kaggle test labels or the public leaderboard in place of local validation.

---

## 5. Simple workflow: independent notebooks → model selection → Kaggle

1. **Set up the team.** Each student joins the Kaggle competition and the same three-person Kaggle team (`COMP472_GroupXX` matching the assigned Moodle group). Download `image_train.csv`, `image_test.csv`, `sentiment_train.csv`, `sentiment_test.csv`, and `sample_submission.csv`.
2. **Work independently.** A implements/tunes the two tree classifiers, B implements/tunes k-NN and performs clustering, C implements/tunes the Neural Network. **Each member's final deliverable is their own complete notebook and reproducible model(s).** All required classifiers must be evaluated on **both tasks**.
3. **Report validation performance.** Each member shares their best **image validation accuracy** and **sentiment validation accuracy** per required classifier, together with chosen hyperparameters. Record the results in a single team comparison table (even if it is simply pasted into the report).
4. **Choose the best model manually, separately for each task.** Compare the validation results and select one model for images and one for sentiment. They can come from **different members**. Don't use the public Kaggle leaderboard as the primary way to select models.
5. **Generate test predictions in the selected model's existing notebook.** The relevant owner uses their chosen model and preprocessing to predict the **5,000 test rows** for that task. After hyperparameters are fixed, the owner may retrain on the entire labeled training set before producing test predictions.
6. **Create one submission CSV.** If the **same member wins both tasks**, that member can produce all 10,000 predictions and create `submission.csv` directly in their own notebook. If **two different members win**, each exports `id,label` predictions for their selected task, and **either selected model owner** combines those two small prediction tables into one `submission.csv` in an existing notebook. **A fourth notebook is not necessary.**
7. **Submit to Kaggle.** Upload the single `submission.csv` containing predictions for **both test datasets**. Keep the model code, experiment results, final prediction cells, and reproducibility instructions in the team repo for the academic deliverables.

### Manual selection example (hypothetical numbers)

| Model | Owner | Image validation accuracy | Sentiment validation accuracy |
|---|---|---:|---:|
| Decision Tree | A | 64% | 70% |
| Random Forest | A | 78% | **84%** |
| k-NN | B | 75% | 80% |
| Neural Network | C | **85%** | 82% |

**Decision:** Use **Member C's Neural Network** for image test predictions and **Member A's Random Forest** for sentiment test predictions. C and A each generate their 5,000 test predictions in their own notebooks. Then A **or** C combines both sets of `id,label` rows to make the single 10,000-row Kaggle submission.

### Submission CSV requirements

- [ ] Exactly two columns: `id,label`.
- [ ] Exactly **10,000 predictions**: 5,000 images + 5,000 sentiment samples.
- [ ] IDs match `sample_submission.csv` **exactly once each**; don't assume the sample file uses the same row order as the test files.
- [ ] Predictions are paired with their own test `id` before combining. Align against `sample_submission.csv` by `id`.
- [ ] Image labels are **integers 0–49**; sentiment labels are **integers 0–2**.
- [ ] No missing predictions, extra IDs, duplicate IDs, or accidental index column.
- [ ] Only test predictions are included (not training/validation predictions).

**Minimal CSV assembly cell** (put this inside **one existing member notebook** after both selected models' predictions are available):

```python
import pandas as pd

# Two tables supplied by the winning model owner(s):
# image_results columns: id, label (5,000 rows)
# sentiment_results columns: id, label (5,000 rows)
all_predictions = pd.concat([image_results, sentiment_results], ignore_index=True)

sample = pd.read_csv("sample_submission.csv")
submission = sample[["id"]].merge(
    all_predictions,
    on="id",
    how="left",
    validate="one_to_one",
)

assert len(submission) == 10_000
assert submission["label"].notna().all()
assert set(submission["id"]) == set(sample["id"])

submission["label"] = submission["label"].astype(int)
image_mask = submission["id"].str.startswith("img_")
sentiment_mask = submission["id"].str.startswith("text_")
assert image_mask.sum() == 5_000 and sentiment_mask.sum() == 5_000
assert submission.loc[image_mask, "label"].between(0, 49).all()
assert submission.loc[sentiment_mask, "label"].between(0, 2).all()

submission.to_csv("submission.csv", index=False)
```

The code above **only combines and validates predictions**. The actual model training and prediction must remain in the selected owner's notebook(s).

---

## 6. What each member must document in their notebook

Each notebook should be ready to serve as that member's final reproducible code deliverable:

- [ ] Complete implementation of assigned algorithm(s), complying with the from-scratch/PyTorch rules.
- [ ] Results for **both** image and sentiment datasets (except clustering, which needs only one).
- [ ] Clear validation method and the same split convention as teammates.
- [ ] Tested hyperparameters, chosen settings, and **validation accuracies**.
- [ ] Brief interpretation of results, including relevant errors and underfitting/overfitting.
- [ ] Training and prediction steps; saved model checkpoint if helpful, or sufficient code to retrain.
- [ ] Reproducible seeds, package versions, data-loading instructions, and cells that run top-to-bottom.
- [ ] A final test-prediction section that produces `id,label` if the team chooses that member's model.
- [ ] Notebook saved to GitHub.

**No teammate review is assigned.** Each member is accountable for the correctness and reproducibility of their notebook. **However, the professor states that any member may be questioned about any part of the team's project**, so all three still need a working understanding of the complete submission.

---

## 7. Report, presentation, and other course requirements

**Kaggle receives only the CSV for scoring.** The academic project also requires a **short report through Moodle**, a presentation involving **all three members**, and reproducible code handed in as instructed by the teaching team.

The report/presentation should cover:

- All four classification algorithms and a **4 × 2 comparison table** of validation accuracy.
- How algorithms were implemented and which hyperparameters were explored.
- Validation method, preprocessing choices, and differences between the two tasks.
- Training behavior, overfitting/underfitting, and a short error analysis (e.g., confusion matrices or difficult classes).
- At least one clustering experiment and what the clusters reveal.
- Which model was selected for image and which for sentiment, and **why**.
- Important observations, relevant computational costs, division of work, and outside resources/assistance.
- Enough information to reproduce the final experiments (seeds, versions, training procedure, hyperparameters, and data setup).

**Writing division:** A documents trees/forest experiments, B documents k-NN/clustering, C documents Neural Network experiments. Combine those contributions into the required team report and slides. There is no separate code-review assignment.

**Coding/data restrictions:** Use only the supplied labeled training data and pre-extracted numerical features; no external labeled data, new pretrained feature encoders/classifiers, or hidden test labels. Libraries may be used for standard data manipulation, train/validation splitting, metrics, and visualization, but **not as complete replacements for required from-scratch supervised algorithms**.

**Important:** Check Moodle/instructor announcements for the actual deadlines, detailed report instructions, and the required method for submitting the code. The public Kaggle score is interim feedback; the **private leaderboard** determines the competition performance grade.
