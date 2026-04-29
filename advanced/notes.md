# Advanced Analysis: Autoencoder-Based Latent State Discovery

## Goal

The goal of the advanced part of the project was to investigate whether raw physiological time-series data could be used to discover latent emotional or physiological states in an unsupervised way.

The intended pipeline was:

```text
Raw HR / EDA / TEMP signals
→ preprocessing and windowing
→ Conv1D autoencoder
→ latent representation
→ K-Means clustering
→ evaluation against phase, cohort, individual, and questionnaire variables
```

The motivation was to use the autoencoder as a self-supervised representation learning method. Instead of manually defining features, the model learns a compressed latent representation of the physiological signals. These latent vectors are then clustered to investigate whether meaningful hidden states emerge.

---

## Dataset Used

For this advanced part, we used the **raw time-series dataset**, not the simplified preprocessed feature file.

The signals used were:

```text
HR, EDA, TEMP
```

The raw dataset also contains BVP, but BVP was excluded because it is noisier and sampled at a different frequency.

The experiment contains three main phase types:

```text
phase1 = pre-puzzle / resting phase
phase2 = puzzle / competition / stress phase
phase3 = post-puzzle / recovery phase
```

Each participant completed multiple rounds, and after each phase they filled out self-reported questionnaire values related to frustration, positive emotions, and negative emotions.

---

## Final Folder Structure

The final structure separates reusable processed data from model-specific outputs.

```text
project_root/
├── data/
│   ├── raw/
│   │   └── data/
│   │       └── dataset/
│   └── processed/
│       └── autoencoder/
│           └── autoencoder_windows.npz
│
└── advanced/
    ├── v1_autoencoding.py
    ├── outputs/
    │   └── conv_ae_32/
    │       ├── figures/
    │       ├── eval/
    │       ├── kmeans_silhouette_scores.csv
    │       ├── metrics.json
    │       ├── phase_latents.npy
    │       ├── phase_metadata_with_clusters.csv
    │       └── training_loss.csv
    │
    └── utils/
        ├── conv_autoencoder.py
        ├── data_processing.py
        └── evaluate/
            └── evaluate_clusters.py
```

The rule is:

```text
data/processed/
= reusable processed input data

advanced/outputs/
= results from model experiments
```

This organization avoids mixing reusable processed data with experiment-specific results.

---

## Code Organization

### `data_processing.py`

This file contains only data extraction and preprocessing functions.

It performs:

```text
raw dataset traversal
→ signal loading
→ resampling to 1 Hz
→ questionnaire metadata attachment
→ 60-second window creation
→ global signal standardization
→ conversion to Conv1D tensor format
```

It does **not** train models, run clustering, or save experiment results.

The main reusable output is:

```text
data/processed/autoencoder/autoencoder_windows.npz
```

This file contains the processed windows and metadata needed by the autoencoder models.

---

### `conv_autoencoder.py`

This file defines only the Conv1D autoencoder model.

The model input has shape:

```text
batch_size × channels × time
```

For our data:

```text
batch_size × 3 × 60
```

where the three channels are:

```text
HR, EDA, TEMP
```

The encoder compresses each 60-second physiological window into a latent vector.

We tested several latent dimensions:

```text
32, 16, 8, 4, 2
```

---

### `v1_autoencoding.py`

This is the main experiment script.

It performs:

```text
load or create processed data
→ train Conv1D autoencoder
→ extract window-level latent vectors
→ average window latents per phase
→ run K-Means on phase-level latent vectors
→ save useful experiment results
```

The model is trained using reconstruction loss:

```text
MSE(original window, reconstructed window)
```

Clustering is performed on the original latent space, not on PCA or t-SNE projections.

The most important saved outputs are:

```text
advanced/outputs/conv_ae_<latent_dim>/phase_latents.npy
advanced/outputs/conv_ae_<latent_dim>/phase_metadata_with_clusters.csv
advanced/outputs/conv_ae_<latent_dim>/training_loss.csv
advanced/outputs/conv_ae_<latent_dim>/kmeans_silhouette_scores.csv
advanced/outputs/conv_ae_<latent_dim>/metrics.json
```

---

### `evaluate_clusters.py`

This script evaluates the clusters produced by any model, as long as the model saves:

```text
phase_metadata_with_clusters.csv
```

The evaluator compares cluster labels against:

```text
Phase
Cohort
Round
Individual
```

It computes:

```text
ARI = Adjusted Rand Index
NMI = Normalized Mutual Information
```

It also computes questionnaire profiles by cluster, when numeric questionnaire variables are present.

Evaluation results are saved in:

```text
advanced/outputs/<experiment_name>/eval/
```

The evaluation files are:

```text
cluster_alignment_metrics.csv
questionnaire_cluster_profiles.csv
summary.txt
```

---

## Methodology

### 1. Preprocessing

Each raw signal was resampled to 1 Hz.

The signals used were:

```text
HR, EDA, TEMP
```

The data was divided into overlapping windows:

```text
window size = 60 seconds
step size   = 30 seconds
```

Therefore, each input sample had shape:

```text
60 × 3
```

Rows where all signals were missing were removed, and windows containing remaining missing values were skipped.

The signal channels were standardized globally using `StandardScaler`.

---

### 2. Autoencoder Training

A Conv1D autoencoder was trained to reconstruct each 60-second physiological window.

The learned representation was extracted from the encoder.

For each window, the model produced a latent vector of size:

```text
latent_dim
```

The tested latent dimensions were:

```text
latent_dim = 32
latent_dim = 16
latent_dim = 8
latent_dim = 4
latent_dim = 2
```

---

### 3. Phase-Level Aggregation

The autoencoder produces one latent vector per 60-second window.

However, the experimental labels and questionnaire values are phase-level.

Therefore, window-level latent vectors were averaged for each:

```text
Cohort + Individual + Round + Phase
```

This produced one latent vector per participant-round-phase.

The final phase-level dataset contained:

```text
312 phase-level samples
```

---

### 4. Clustering

K-Means clustering was applied to the phase-level latent vectors.

The number of clusters was selected using silhouette score over a range of candidate `k` values.

For the main model with `latent_dim = 32`, the best result was:

```text
best_k = 2
best_silhouette = 0.3449
```

This indicates moderate geometric structure in the learned latent space.

However, silhouette score only measures cluster separation in latent space. It does not prove that the clusters correspond to emotional states.

---

## Main Results: Conv1D Autoencoder with Latent Dimension 32

The main model used:

```text
model: Conv1DAutoencoder
latent_dim: 32
window_size: 60
step_size: 30
epochs: 30
best_k: 2
best_silhouette: 0.3449
final_training_loss: 0.00656
```

The final reconstruction loss was low, suggesting that the model learned to reconstruct the physiological windows well.

However, cluster evaluation showed that the resulting clusters did not correspond to experimental phase.

---

## Alignment Metrics

```text
variable       ARI       NMI
Phase       -0.0021    0.0013
Cohort       0.2198    0.3507
Round       -0.0018    0.0023
Individual   0.0048    0.0310
```

---

## Phase Interpretation

The phase alignment metrics were almost zero:

```text
Phase ARI ≈ 0
Phase NMI ≈ 0
```

This means that the clusters do not separate:

```text
pre-puzzle / rest
puzzle / stress
post-puzzle / recovery
```

The phase-cluster table confirms this:

```text
Cluster   0   1
Phase          
phase1   26  78
phase2   27  77
phase3   31  73
```

The distribution across clusters is almost the same for all three phases.

Therefore, the current model does not provide evidence that the learned clusters correspond to latent emotional states.

---

## Cohort Effect

The strongest alignment was with Cohort:

```text
Cohort ARI = 0.2198
Cohort NMI = 0.3507
```

The cohort-cluster table showed strong cohort imbalance:

```text
Cluster   0   1
Cohort         
D1_1      2  94
D1_2     64   8
D1_3      0  48
D1_4      2  46
D1_5      0  24
D1_6     16   8
```

This suggests that the learned representation captures cohort or acquisition-session effects more strongly than emotional phase effects.

Possible explanations include:

```text
different acquisition sessions
sensor placement differences
recording conditions
cohort-level physiological baselines
session-specific noise
```

---

## Latent Dimension Comparison

We tested several latent dimensions to see whether a smaller latent space would reduce cohort effects and improve phase alignment.

### Results

```text
latent_dim = 32
Phase NMI      = 0.0013
Cohort NMI     = 0.3507
Individual NMI = 0.0310

latent_dim = 16
Phase NMI      = 0.0002
Cohort NMI     = 0.3694
Individual NMI = 0.0222

latent_dim = 8
Phase NMI      = 0.0030
Cohort NMI     = 0.2796
Individual NMI = 0.1246

latent_dim = 4
Phase NMI      = 0.0011
Cohort NMI     = 0.3726
Individual NMI = 0.0224

latent_dim = 2
Phase NMI      = 0.0025
Cohort NMI     = 0.2952
Individual NMI = 0.1746
```

---

## Interpretation of Latent Dimension Results

Across all latent dimensions, Phase alignment remained approximately zero.

This means that reducing the latent dimension did not make the autoencoder discover phase-related emotional states.

Cohort alignment remained much higher than Phase alignment for every tested latent dimension.

The `latent_dim = 8` and `latent_dim = 2` models reduced Cohort NMI somewhat, but Individual NMI increased. This suggests that smaller latent spaces may shift some structure from cohort-level effects toward individual-level baseline effects, but they still do not recover experimental phase structure.

Overall:

```text
Changing latent dimension alone did not solve the problem.
The latent space remained dominated by cohort/session or baseline effects.
```

---

## Meaning of the Evaluation Metrics

### Adjusted Rand Index

The Adjusted Rand Index, or ARI, measures agreement between two clusterings or labelings.

In this project, it compares:

```text
K-Means cluster labels
vs.
Phase / Cohort / Round / Individual labels
```

Interpretation:

```text
ARI = 1
perfect agreement

ARI ≈ 0
agreement no better than random

ARI < 0
worse than random, usually interpreted as no meaningful agreement
```

Therefore, Phase ARI values around zero indicate that the clusters do not match the experimental phases.

---

### Normalized Mutual Information

Normalized Mutual Information, or NMI, measures how much information one labeling gives about another.

Interpretation:

```text
NMI = 1
perfect correspondence

NMI = 0
no shared information
```

In our results, Phase NMI was near zero for all latent dimensions, meaning that knowing the cluster label gives almost no information about the phase.

Cohort NMI was much higher, meaning that cluster labels are more informative about acquisition cohort than experimental phase.

---

## Current Conclusion

The Conv1D autoencoder successfully learned a compact representation of raw HR, EDA, and TEMP windows.

However, the learned clusters do not correspond to the experimental phases.

The main finding is:

```text
The autoencoder finds structure in the latent space,
but this structure is mainly cohort-related,
not phase-related.
```

Therefore, the current model should be interpreted as a useful representation-learning baseline, but not as evidence that latent emotional states were discovered.

A careful conclusion is:

> The Conv1D autoencoder learned a low-dimensional representation of physiological time-series windows and produced moderately separated K-Means clusters. However, cluster evaluation showed almost no alignment with experimental phase, while cohort alignment was substantially stronger. This suggests that the learned representation is dominated by cohort or acquisition-session effects rather than latent emotional states. Reducing the latent dimension from 32 to 2 did not substantially improve phase alignment. These results highlight the difficulty of unsupervised emotional-state discovery from noisy wearable biosignals and support the need for stronger baseline normalization and comparison against simpler feature-based methods.

---

## Next Steps

Based on the current results, changing the latent dimension is not enough.

The next methodological improvements should focus on reducing baseline and cohort effects.

### 1. Subject-Level or Cohort-Level Normalization

The next experiment should normalize HR, EDA, and TEMP within each individual or within each cohort before training.

Possible options:

```text
global normalization
individual-level normalization
individual-round normalization
cohort-level normalization
```

The goal is to test whether removing baseline differences reduces cohort alignment and improves phase or questionnaire alignment.

---

### 2. Compare Against PCA

A PCA baseline should be added.

This would test whether the nonlinear autoencoder actually improves over a simpler linear representation.

The comparison should use the same evaluation metrics:

```text
silhouette score
Phase ARI/NMI
Cohort ARI/NMI
Individual ARI/NMI
questionnaire profiles
```

---

### 3. Analyze Questionnaire Profiles

The file:

```text
questionnaire_cluster_profiles.csv
```

should be inspected to determine whether clusters differ in self-reported variables such as:

```text
frustration
nervousness
upset
hostility
alertness
difficulty
```

If clusters do not align with phase but show differences in questionnaire scores, they may still capture some emotion-related variation.

---

### 4. Use the Simplified Feature Dataset

The assignment expects the simplified extracted feature file as the minimum dataset. Therefore, the raw time-series autoencoder should be presented as an advanced extension.

A complete project should also include an analysis based on the simplified feature dataset, such as:

```text
simplified HR/EDA/TEMP features
→ PCA or VAE
→ clustering
→ evaluation against phase and questionnaires
```

This would make the analysis better aligned with the core project requirements while keeping the raw-signal autoencoder as an advanced comparison.

---

## Summary

We implemented a complete unsupervised representation-learning pipeline using raw physiological time-series data.

The pipeline:

```text
loads raw HR/EDA/TEMP signals
resamples them to 1 Hz
creates 60-second overlapping windows
trains a Conv1D autoencoder
extracts latent representations
aggregates them per phase
clusters them using K-Means
evaluates clusters against metadata and questionnaires
```

The main result is that the autoencoder learns structure, but this structure is not aligned with experimental phase.

Instead, the clusters are much more strongly associated with cohort.

This suggests that cohort/session or baseline effects dominate the learned representation, and that further preprocessing, normalization, and baseline comparisons are necessary before claiming discovery of latent emotional states.

## Normalization Experiments

Because the initial autoencoder results showed strong cohort alignment, we tested additional normalization strategies.

The goal was to determine whether removing baseline or cohort-specific effects would reveal phase-related structure.

The tested normalization strategies were:

```text
cohort normalization
individual normalization
individual-round normalization
```

All normalized experiments used the same Conv1D autoencoder setup with:

```text
latent_dim = 32
window_size = 60
step_size = 30
```

---

## Cohort Normalization Results

```text
variable       ARI       NMI
Phase       -0.0011    0.0056
Cohort      -0.0040    0.0071
Round        0.0119    0.0215
Individual   0.0582    0.1343
```

### Interpretation

Cohort normalization substantially reduced the cohort effect.

Compared with the original global-normalization result:

```text
Cohort NMI: 0.3507 → 0.0071
```

This means that cohort normalization successfully removed most of the cohort/session structure from the latent space.

However, Phase NMI remained very low:

```text
Phase NMI: 0.0013 → 0.0056
```

Although this is slightly higher than before, it is still essentially close to zero.

Therefore, cohort normalization removed the main cohort confound, but it did not reveal clear phase-related emotional states.

After cohort normalization, Individual NMI increased:

```text
Individual NMI: 0.0310 → 0.1343
```

This suggests that once cohort differences are removed, the remaining structure may be more related to individual-level physiological differences.

---

## Individual Normalization Results

```text
variable       ARI       NMI
Phase       -0.0038    0.0001
Cohort       0.2549    0.4031
Round       -0.0034    0.0011
Individual   0.0012    0.0095
```

### Interpretation

Individual normalization did not improve phase alignment.

Phase NMI remained almost zero:

```text
Phase NMI = 0.0001
```

Cohort NMI actually increased compared with the original global-normalization result:

```text
Cohort NMI: 0.3507 → 0.4031
```

This suggests that normalizing within individuals removed individual baseline differences, but did not remove cohort-level effects. In fact, it made cohort-level structure even more dominant in the learned latent space.

---

## Individual-Round Normalization Results

```text
variable       ARI       NMI
Phase       -0.0031    0.0007
Cohort       0.2811    0.4256
Round       -0.0038    0.0004
Individual   0.0031    0.0107
```

### Interpretation

Individual-round normalization also did not improve phase alignment.

Phase NMI remained essentially zero:

```text
Phase NMI = 0.0007
```

Cohort NMI increased even further:

```text
Cohort NMI: 0.3507 → 0.4256
```

This was the strongest cohort alignment among the tested normalization methods.

This suggests that normalizing within each individual and round removes local subject/round variation, but the remaining latent structure is still strongly cohort-related.

---

## Summary of Normalization Results

```text
Normalization        Phase NMI   Cohort NMI   Individual NMI   Interpretation
Global               0.0013      0.3507       0.0310           cohort-driven
Cohort               0.0056      0.0071       0.1343           cohort removed, no phase structure
Individual           0.0001      0.4031       0.0095           cohort effect stronger
Individual-round     0.0007      0.4256       0.0107           cohort effect strongest
```

The most important result is that **cohort normalization successfully reduced the cohort confound**, but **phase alignment still remained very weak**.

This suggests that the lack of phase structure is not only due to cohort effects dominating the latent space. Even after removing cohort structure, the autoencoder still does not recover clear rest/puzzle/recovery states.

---

## Updated Conclusion

The Conv1D autoencoder successfully learned a compact representation of raw HR, EDA, and TEMP windows.

However, the learned clusters do not correspond to the experimental phases.

The main finding is:

```text
The autoencoder finds structure in the latent space,
but this structure is mainly cohort-related or baseline-related,
not phase-related.
```

Reducing the latent dimension did not solve the issue.

Cohort normalization successfully removed the cohort confound, but it still did not produce phase-aligned clusters.

Therefore, the current model should be interpreted as a useful representation-learning baseline, but not as evidence that latent emotional states were discovered.

A careful conclusion is:

> The Conv1D autoencoder learned a low-dimensional representation of physiological time-series windows and produced moderately separated K-Means clusters. However, cluster evaluation showed almost no alignment with experimental phase, while cohort alignment was substantially stronger. Reducing the latent dimension from 32 to 2 did not substantially improve phase alignment. Cohort normalization successfully reduced cohort alignment, but did not reveal meaningful phase-related structure. These results highlight the difficulty of unsupervised emotional-state discovery from noisy wearable biosignals and suggest that additional feature engineering, alternative clustering methods, or simplified feature-based analyses are needed before claiming discovery of latent emotional states.

---

## Next Steps

Based on the current results, changing the latent dimension and normalization strategy is not enough.

The next methodological improvements should focus on testing whether the issue is caused by the representation, the clustering algorithm, or the raw data itself.

### 1. Try Alternative Clustering Algorithms

So far, clustering was done using K-Means.

K-Means assumes roughly spherical clusters and may not work well if the latent space has elongated or overlapping structures.

The next clustering methods to try are:

```text
Gaussian Mixture Models
Agglomerative clustering
DBSCAN or HDBSCAN
```

The most useful next experiment is:

```text
cohort-normalized ConvAE latent space
→ GMM clustering
→ same evaluation metrics
```

The key question is:

```text
Does a more flexible clustering method improve Phase NMI or questionnaire interpretability?
```

If GMM still produces Phase NMI close to zero, then the issue is probably not only K-Means.

---

### 2. Compare Against PCA

A PCA baseline should be added.

This would test whether the nonlinear autoencoder actually improves over a simpler linear representation.

The comparison should use the same evaluation metrics:

```text
silhouette score
Phase ARI/NMI
Cohort ARI/NMI
Individual ARI/NMI
questionnaire profiles
```

If PCA gives similar or better results, then the autoencoder is not adding much for this task.

---

### 3. Analyze Questionnaire Profiles

The file:

```text
questionnaire_cluster_profiles.csv
```

should be inspected to determine whether clusters differ in self-reported variables such as:

```text
frustration
nervousness
upset
hostility
alertness
difficulty
```

If clusters do not align with phase but show differences in questionnaire scores, they may still capture some emotion-related variation.

If neither phase nor questionnaire profiles differ by cluster, then the clusters are not emotionally interpretable.

---

### 4. Use the Simplified Feature Dataset

The assignment expects the simplified extracted feature file as the minimum dataset. Therefore, the raw time-series autoencoder should be presented as an advanced extension.

A complete project should also include an analysis based on the simplified feature dataset, such as:

```text
simplified HR/EDA/TEMP features
→ PCA or VAE
→ clustering
→ evaluation against phase and questionnaires
```

This would make the analysis better aligned with the core project requirements while keeping the raw-signal autoencoder as an advanced comparison.

---

## Short Summary

We extended the initial autoencoder analysis by testing whether normalization could reduce cohort/session effects.

The original globally normalized model produced clusters that were much more aligned with cohort than with phase.

Cohort normalization successfully reduced Cohort NMI from `0.3507` to `0.0071`, showing that the main cohort confound could be removed. However, Phase NMI remained very low, increasing only from `0.0013` to `0.0056`.

Individual and individual-round normalization did not help. In fact, both increased Cohort NMI, suggesting that those normalization strategies made cohort-level structure more dominant.

Overall, normalization confirmed that the raw time-series autoencoder does not currently recover phase-aligned latent emotional states. The next steps should be to test alternative clustering algorithms, compare against PCA, inspect questionnaire profiles, and run a baseline analysis using the simplified feature dataset.