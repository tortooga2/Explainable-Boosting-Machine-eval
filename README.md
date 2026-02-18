# Report 3: Explainable Boosting Machine

## Overview of Model

The Explainable Boosting Machine (EBM) is inherently interpretable without sacrificing much accuracy, achieving this by modeling the relationship between each feature and the prediction independently. It is a modern variant of the Generalized Additive Model (GAM) that targets the variables as a sum of functions, one for each feature. One of its major advantages is its useful diagnostic plots (feature effects plus interactions) displayed as graphs. Its ability to learn one feature at a time allows us to analyze plots that show the exact impact (the shape of the relationship) that a feature's value has on the model's output, making it easy to diagnose the model's behavior. The added focus on additive effects (interactions) can also show us the interactions between features. This means that while the model is very capable, the accuracy of it is mainly a measure of how deeply the model extracted information from the features itself. It is this difference from other black box models that most drove our decision-making around treating our data. As you will see, this model is only as good as the dataset.

---

## Regression

### Dataset: 515k Hotel Reviews from Europe

> [https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

The dataset chosen was 515k Hotel Reviews — a hotel review/hospitality satisfaction dataset where each row is an individual user review for a specific hotel, containing both categorical and continuous features.

**Categorical features:** Hotel Name, Hotel Address, Date, Review Nationality, and Tags.

**Continuous features:** Review Score, Average Hotel Score, Total Number of Reviews, Days Since User Last Reviewed, Total Negative Word Count in Review, Total Positive Word Count in Review, Reviewer Score, Total Number of Reviews the Reviewer Has Given, and Hotels Longitude and Latitude.

The dataset also includes the actual review text, however given the nature of the model, encoding this review was possible but it defeated the purpose — the features required for text vectorization would be incomprehensible to the model user. Of these features, we thought that a valuable use of the model would be trying to predict a user's final review score based on the other features.

### Data Processing

We had attempted to vectorize the text and tags, however found it removed from the interpretability of the final model. It is also important to note that this dataset has already undergone pretty extensive processing by the original creator. To preserve interpretability, methods like PCA were not used. The only major change made was converting dates from text to individual integers for year, month, and day — which gave us some surprising insights later.

The dataset was split 80/20 for training and testing respectively. A 5-fold cross validation on `X_train` and `Y_train` was performed on the training data. The test set is held out entirely during the training and tuning phases, ensuring that model performance is calculated using unseen data.

**Hyperparameters tuned:**
- `max_bins`: 32, 64, 128
- `interactions`: 1, 2, 5, 10
- `learning_rate`: 0.02, 0.01 *(kept constrained by the nature of the model)*
- `max_rounds`: 50, 100

We found that `max_rounds` represented the largest impact to training time.

### Cross Validation Results

The most noteworthy result from cross-validation was the behavior of `max_bins`. The general assumption was that increasing max bins — like increasing the number of estimators in a decision tree-based model — would result in better prediction. However, that assumption proved very wrong here. Since user review scores range from 1–10 at 0.5 intervals, and the number of bins represents the number of discretized steps we break down our features into, using 128 bins is overkill and a complete misalignment of this hyperparameter with our dataset. This similarly follows for many of our categorical features.

### Results of Final Model

| Metric | Score |
|--------|-------|
| Mean Absolute Error (Test Set) | 0.9074 |
| R² Score (Test Set) | 0.4481 |

From the graph and the final R² score, the model did not perform particularly well. The standard deviation for predictions across the entire range of scores averaged roughly ±1. Especially concerning was the low end of the graph — the model mainly predicted values near the mean of the dataset, only starting to align with actual data near the high end. This is known as **Regression to the Mean**, resulting from a lack of data at the low end, where the model defaults to estimating the mean to best minimize error.

As scores increase toward 8 and above, the model begins to meet the true predictions, though it consistently underestimates. This transition near 8 makes sense within the context of our dataset: since these are user reviews, people are less likely to give a score above 7 when they have conflicted or negative views. This is apparent on platforms like IMDB, where most films cluster within ±0.5 of 7, with scores of 8 or higher reserved for notably good titles. This is best described as **Central Tendency Bias**, as well as a known phenomenon called **Goodhart's Law** — *when a measure becomes a target, it ceases to be a good measure.* However, while predictive accuracy is generally low, the main value of this model is that it is **interpretable**.

### Interpreting the Results

One of the provided EBM feature graphs shows the contribution to the overall prediction by review month. The x-axis spans January (1) through December (12). The beginning of the year shows a positive bias on predicted score, which weakens as the year progresses, turning negative in June, reaching an all-time low in November before slightly recovering in December.

The data distribution (shown in orange) shows no dramatic bias toward specific months, so what explains the negative bias in November? A few factors likely contribute:

- **Seasonal depression** — Many people experience lower mood when weather deteriorates.
- **Travel disruptions** — A majority of flight cancellations and delays occur during winter months.
- **End-of-year stress** — Workers rush to complete projects as the fiscal year closes.
- **Holiday travel** — Peak travel season brings crowded hotels, family obligations, and the added stress of traveling with children.

This graph illustrates a key strength of the EBM: it reveals interesting correlations between features and predictions, and invites interpretation. The month-score relationship shows that external circumstances surrounding a reviewer's stay can meaningfully affect their assessment.

Another notable interaction graph plots **negative word count vs. positive word count**. The expected linear correlation holds — more positive words and fewer negative words predicts a higher score. However, at the extremes this breaks down: when either sentiment becomes overwhelming, the model's predictive ability actually degrades (indicated by red areas in the heatmap). While this may partly reflect data sparsity at the extremes, it is still a noteworthy finding.

As demonstrated, the main benefit of the EBM is that it enables the creation of new knowledge and testable hypotheses by examining the relevance of each feature to a given prediction. While the ability to predict new review scores is limited, the actual analytical value of the model remains high.

---

## Classification

### Dataset: Corpus of Social Touch (CoST)

> [https://data.4tu.nl/articles/dataset/Corpus_of_Social_Touch_CoST_/12696869/1](https://data.4tu.nl/articles/dataset/Corpus_of_Social_Touch_CoST_/12696869/1)

For classification, we used the Corpus of Social Touch dataset — a collection of 7,805 gesture sequences covering 14 different social touch gestures:

> *grab, hit, massage, pat, pinch, poke, press, rub, scratch, slap, squeeze, stroke, tap, tickle*

Each gesture was performed in three variations — **gentle, normal, and rough** — on a pressure sensor grid wrapped around a mannequin arm, across 31 subjects. Since the dataset is temporal, each gesture consists of multiple frames made up of 64 sensor channels.

### Data Processing

- Split at the **sequence level** (full gesture instances across frames)
- Random state set to **42** for reproducibility
- **80/20 train/test split** with stratification to preserve class proportions
- Applied **Robust Scaler** (center by median, scale by IQR) to reduce sensitivity to outliers common in sensor data
- Hyperparameters tuned using **StratifiedKFold CV (3 folds)** on training data only

**Hyperparameters:**
- `interactions`: 0 *(also tested with 1, but kept 0 for speed and interpretability; accuracy difference was small)*
- `max_rounds`: 50, 100, 500
- `learning_rate`: 0.05 *(low to reduce divergence without excessively slowing training)*
- `max_bins`: 32 *(moderate to reduce noise sensitivity and training time)*
- Outer bags for stability; inner bags kept small to limit runtime

### Feature Engineering

Since EBM expects tabular inputs rather than raw time series, each gesture sequence was converted into a fixed-length feature vector. For each of the 64 channels, the following features were computed:

| Feature | Description |
|---------|-------------|
| Mean, Std, Max | Overall pressure level, variability, and peak force |
| Mean Absolute Difference (1st diff) | Channel-wise proxy for rate of pressure change |
| Std of 2nd Difference | "Jerkiness" — smooth vs. abrupt gesture dynamics |
| First-half vs. Second-half Mean | Temporal asymmetry (ramp-up vs. release gestures) |
| Time Index of Max Pressure (normalized) | When the peak occurred within the gesture |
| Zero-Crossing Rate | Oscillatory/alternating patterns |
| Channel-wise Energy | Total accumulated pressure magnitude |

All features were stacked horizontally to form **X ∈ ℝ^{R×640}**, preserving interpretability since each feature has a clear physical meaning.

### Results

**Confusion Matrix findings:** Low-force gestures (e.g., squeeze, tickle, scratch) were frequently confused with one another, while high-force localized gestures (e.g., poke) had high correctly predicted rates. This aligns intuitively with how similar gentle gestures feel in terms of raw pressure signatures.

**ROC Curve:** A one-vs-rest approach was used across all 14 gesture classes. Selected AUC scores:

| Gesture | AUC |
|---------|-----|
| Massage | 0.99 |
| Poke | 0.98 |
| Slap | 0.98 |
| Grab | 0.97 |
| Hit | 0.97 |
| Stroke | 0.97 |
| Pinch | 0.97 |
| Press | 0.97 |
| **Micro-average** | **0.97** |
| Pat | 0.95 |
| Scratch | 0.95 |
| Squeeze | 0.95 |
| Tickle | 0.95 |
| Tap | 0.94 |
| Rub | 0.93 |

**Top 20 Features:** The most important features were `MeanAbsDiff` channels (e.g., Ch36, Ch37, Ch52), representing sensor locations and movement dynamics most critical for distinguishing between gestures. The EBM measures importance by how much each feature's value shifts the model's prediction confidence.

### Conclusion

While the EBM worked well on the regression dataset, it did not achieve the classification accuracy hoped for on the CoST dataset. Key challenges included:

- **Computational cost** — The matrix shape of 7,805 × 640 made training and cross-validation expensive.
- **Information compression** — Temporal statistics (mean, peak, energy) compress sequences, meaning two gestures can share similar aggregate profiles while differing in fine-grained temporal structure.
- **No interactions** — With `interactions = 0`, the model cannot represent complex spatiotemporal dependencies that distinguish subtle gestures from one another.

This explains why the model remains interpretable and tractable, yet underfits nuanced temporal gesture distinctions. The EBM's strength — its transparency — comes at the cost of expressive power for datasets where complex feature interactions across time are critical for accurate classification.
