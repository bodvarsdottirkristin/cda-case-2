# Advanced Analysis: Autoencoder-Based Latent State Discovery

## Goal

The goal of the advanced part was to test whether raw physiological time-series data could reveal latent emotional or physiological states through unsupervised learning.

The intended pipeline was:

```text
Raw HR / EDA / TEMP signals
→ preprocessing and windowing
→ representation learning or dimensionality reduction
→ clustering
→ evaluation against phase, role, cohort, individual, and questionnaire variables
```

The main representation model was a Conv1D autoencoder. We later added PCA as a linear baseline and tested alternative clustering methods to check whether the results were specific to K-Means.

---

## Dataset and Signals

For this advanced analysis, we used the **raw time-series dataset**, not the simplified preprocessed feature file.

The signals used were:

```text
HR, EDA, TEMP
```

BVP was excluded because it is noisier and sampled at a different frequency.

The experiment contains three main phases:

```text
phase1 = pre-puzzle / resting phase
phase2 = puzzle / competition / stress phase
phase3 = post-puzzle / recovery phase
```

Participants completed multiple rounds and filled out questionnaire values after each phase, including frustration, positive emotions, and negative emotions. The response files also contain role/context variables such as `Puzzler` and `parent`.

---

## Code Organization

The project was reorganized to separate reusable data, model definitions, experiment scripts, and evaluation scripts.

```text
data/processed/
= reusable processed input data

advanced/outputs/
= model-specific experiment results
```

Main files:

```text
advanced/utils/data_processing.py
advanced/utils/data_processing_norm.py
advanced/utils/conv_autoencoder.py
advanced/utils/PCA.py
advanced/v1_autoencoding.py
advanced/v2_autoencoding.py
advanced/v3_PCA.py
advanced/v4_gmm.py
advanced/v5_agglomerative.py
advanced/utils/evaluate/evaluate_clusters.py
advanced/utils/evaluate/check_questionnaire_profiles.py
```

### Data-processing scripts

`data_processing.py` handles the original raw-data pipeline:

```text
raw dataset traversal
→ signal loading
→ resampling to 1 Hz
→ questionnaire metadata attachment
→ 60-second window creation
→ global signal standardization
→ conversion to Conv1D format
```

The reusable processed file is:

```text
data/processed/autoencoder/autoencoder_windows.npz
```

`data_processing_norm.py` creates normalized processed datasets to test whether baseline or cohort effects were dominating the learned latent space. We tested:

```text
cohort normalization
individual normalization
individual-round normalization
```

### Model and experiment scripts

`conv_autoencoder.py` defines the Conv1D autoencoder. The input shape is:

```text
batch_size × 3 × 60
```

where the three channels are `HR`, `EDA`, and `TEMP`.

`v1_autoencoding.py` runs the globally normalized ConvAE experiment. `v2_autoencoding.py` runs the same model using normalized processed data. `v3_PCA.py` runs a PCA baseline. `v4_gmm.py` applies GMM clustering to an existing phase-level representation. `v5_agglomerative.py` applies Agglomerative clustering with fixed interpretable values of `k`.

### Evaluation scripts

`evaluate_clusters.py` evaluates cluster alignment with:

```text
Phase
Puzzler
parent
Cohort
Round
Individual
```

using:

```text
ARI = Adjusted Rand Index
NMI = Normalized Mutual Information
```

`check_questionnaire_profiles.py` checks whether clusters differ in questionnaire variables such as:

```text
frustrated, nervous, upset, ashamed, afraid, active, attentive, determined, difficulty
```

Role/context variables such as `Puzzler` and `parent` are not treated as emotion scores.

---

## Methodology

### 1. Preprocessing

Each raw signal was resampled to 1 Hz. Windows were created with:

```text
window size = 60 seconds
step size   = 30 seconds
```

Each input sample had shape:

```text
60 × 3
```

Rows where all signals were missing were removed, and windows containing missing values were skipped.

The original pipeline used global standardization. Later, cohort, individual, and individual-round normalization were tested.

### 2. Autoencoder training

A Conv1D autoencoder was trained to reconstruct each 60-second HR/EDA/TEMP window using reconstruction MSE.

We tested several latent dimensions:

```text
32, 16, 8, 4, 2
```

The autoencoder produced one latent vector per window.

### 3. Phase-level aggregation

Because phase labels and questionnaire values are phase-level, window-level latent vectors were averaged for each:

```text
Cohort + Individual + Round + Phase
```

This produced:

```text
312 phase-level samples
```

Clustering was then applied to these phase-level representations.

---

## Main ConvAE Result: Latent Dimension 32

The main ConvAE model used:

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

The low reconstruction loss suggests that the autoencoder learned to reconstruct the physiological windows well. However, the clusters did not align with experimental phase.

```text
variable       ARI       NMI
Phase       -0.0021    0.0013
Cohort       0.2198    0.3507
Round       -0.0018    0.0023
Individual   0.0048    0.0310
```

The Phase NMI was near zero, meaning that clusters did not separate rest, puzzle/stress, and recovery phases. In contrast, Cohort NMI was much higher, suggesting that the learned representation mainly captured cohort/session effects.

---

## Latent Dimension Comparison

We tested whether a smaller latent space would reduce cohort effects or improve phase alignment.

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

Across all latent dimensions, Phase NMI stayed close to zero. Reducing the latent dimension did not make the autoencoder discover phase-related emotional states.

---

## Normalization Experiments

Because the original model was strongly cohort-driven, we tested cohort, individual, and individual-round normalization.

All normalized experiments used:

```text
latent_dim = 32
window_size = 60
step_size = 30
```

### Results

```text
Normalization        Phase NMI   Cohort NMI   Individual NMI   Interpretation
Global               0.0013      0.3507       0.0310           cohort-driven
Cohort               0.0056      0.0071       0.1343           cohort removed, no phase structure
Individual           0.0001      0.4031       0.0095           cohort effect stronger
Individual-round     0.0007      0.4256       0.0107           cohort effect strongest
```

Cohort normalization was the most useful because it reduced Cohort NMI from `0.3507` to `0.0071`. However, Phase NMI remained near zero, so removing the cohort confound did not reveal clear phase-related emotional states.

Individual and individual-round normalization did not help. In fact, they increased cohort alignment, suggesting that removing individual baselines may have exposed or amplified cohort/session-level structure.

---

## PCA Baseline

We added PCA as a simple linear baseline to compare against the nonlinear ConvAE.

The PCA pipeline used the same processed windows:

```text
N windows × 60 seconds × 3 signals
```

Each window was flattened into:

```text
60 × 3 = 180 features
```

Then PCA was applied, PCA scores were averaged per phase, and K-Means was applied to the phase-level PCA representation.

Standard PCA was used instead of Sparse PCA because the goal was to create a simple baseline, not to interpret individual PCA components.

### PCA with 32 components

```text
variable       ARI       NMI
Phase       -0.0051    0.0022
Cohort       0.1779    0.2849
Round        0.0046    0.0323
Individual   0.0360    0.1191
```

PCA also showed almost no phase alignment. Cohort and individual alignment were higher, meaning that the lack of phase-related clustering was not specific to the ConvAE.

```text
Model       Phase NMI   Cohort NMI   Individual NMI   Interpretation
ConvAE 32   0.0013      0.3507       0.0310           mostly cohort-driven
PCA 32      0.0022      0.2849       0.1191           cohort/individual-driven
```

Neither method discovered phase-related clusters.

---

## Questionnaire Profile Evaluation

To test whether clusters had any emotional interpretation, we added:

```text
advanced/utils/evaluate/check_questionnaire_profiles.py
```

This script computes per-cluster means, standard deviations, mean differences, and standardized differences for questionnaire variables.

The key value is:

```text
standardized_difference
```

which measures how large the cluster difference is relative to the global standard deviation.

### Original ConvAE 32

Largest questionnaire differences:

```text
variable      mean difference   standardized difference
active        0.5131            0.4372
difficulty    1.1670            0.3826
frustrated    0.6823            0.3435
inspired      0.3502            0.3407
determined    0.3541            0.3207
attentive     0.3083            0.2768
```

The original ConvAE clusters showed moderate questionnaire differences, but because this model was strongly cohort-aligned, these differences may be confounded by cohort/session effects.

### Cohort-normalized ConvAE 32

Largest differences included:

```text
variable      mean difference   standardized difference
parent        0.6071            1.2079
nervous       0.3548            0.6147
ashamed       0.2757            0.5022
attentive     0.5177            0.4648
determined    0.4364            0.3952
Puzzler       0.1469            0.2932
difficulty    0.8276            0.2713
upset         0.1255            0.2031
```

`parent` and `Puzzler` are role/context variables, not emotion scores. After excluding them, the strongest emotion-related differences were `nervous`, `ashamed`, `attentive`, `determined`, `difficulty`, and `upset`.

Because cohort alignment was low after cohort normalization, these differences are less likely to be explained by cohort alone. However, Phase NMI remained near zero, so the clusters still cannot be confidently interpreted as latent emotional states.

---

## Alternative Clustering Methods

After K-Means, we tested alternative clustering methods on the **cohort-normalized ConvAE latent space**:

```text
Gaussian Mixture Model selected using BIC
Agglomerative clustering with fixed k values
```

The goal was to check whether the poor phase alignment was caused by K-Means being too restrictive.

### GMM-BIC

GMM can model softer and more elliptical clusters than K-Means. The number of components was selected using BIC because GMM is a probabilistic model and BIC penalizes unnecessary complexity.

```text
variable       ARI       NMI
Phase       -0.0027    0.0035
Puzzler     -0.0026    0.0040
parent       0.0847    0.1401
Cohort       0.0036    0.0147
Round        0.0069    0.0158
Individual   0.0421    0.1079
```

GMM did not improve phase or Puzzler alignment. Cohort alignment remained low, confirming that cohort normalization was still effective. The strongest remaining alignments were with `parent` and `Individual`.

Largest emotion-related differences:

```text
variable      standardized difference
nervous       0.5400
ashamed       0.4186
attentive     0.3857
determined    0.2655
inspired      0.2171
```

GMM did not reveal phase-based emotional states, but it showed weak role-related and affect-related variation.

### Agglomerative clustering

Agglomerative clustering is a form of hierarchical clustering. It starts with each sample as its own cluster and progressively merges the most similar clusters until the chosen number of clusters remains.

We did not select `k` using silhouette. Instead, we tested fixed, interpretable values:

```text
k = 2, 3, 4
```

The motivation was:

```text
k = 2 → possible low/high arousal separation
k = 3 → matches rest / puzzle-stress / recovery
k = 4 → allows one additional role-related or affect-related state
```

The results for `k = 2` and `k = 3` were weak. The most interesting result was obtained with `k = 4`.

```text
variable       ARI       NMI
Phase       -0.0025    0.0049
Puzzler      0.0009    0.0070
parent       0.2277    0.2297
Cohort      -0.0055    0.0187
Round        0.0063    0.0222
Individual   0.0571    0.1330
```

Agglomerative clustering still did not recover phase-based states. However, `k = 4` produced the strongest `parent` alignment and moderate individual alignment.

Largest emotion-related differences:

```text
variable      standardized difference
ashamed       0.7286
nervous       0.6840
determined    0.5395
upset         0.3890
afraid        0.3800
attentive     0.3642
hostile       0.3544
```

This suggests that the cohort-normalized latent space may contain weak role-related and affect-related structure, especially related to `parent`, `ashamed`, and `nervous`, but not clear rest/stress/recovery states.

### Clustering-method comparison

```text
Method              Phase NMI   Puzzler NMI   parent NMI   Cohort NMI   Individual NMI
K-Means              0.0056      0.0070        0.1001       0.0071       0.1343
GMM-BIC              0.0035      0.0040        0.1401       0.0147       0.1079
Agglomerative k=4    0.0049      0.0070        0.2297       0.0187       0.1330
```

None of the clustering methods recovered phase-aligned emotional states. Agglomerative `k = 4` produced the strongest role-related structure and the clearest questionnaire differences.

---

## Final Interpretation

The Conv1D autoencoder learned a compact representation of raw HR, EDA, and TEMP windows, and clustering found some structure in the latent space. However, the learned clusters did not align with experimental phase.

The main finding is:

```text
The learned clusters are not clear rest / stress / recovery states.
```

Instead:

```text
The original model mostly captured cohort/session structure.
Cohort normalization removed the cohort confound.
Even after cohort normalization, phase alignment remained near zero.
Alternative clustering methods did not recover phase states.
Some weak role-related and affect-related variation appeared, especially for parent role, nervousness, and shame.
```

A careful conclusion is:

> The raw physiological time-series data contains weak affect-related and role-related variation, but the unsupervised clusters do not clearly correspond to latent emotional states. The learned representations are more strongly influenced by cohort/session, individual, and role-related effects than by experimental phase.

---

## Next Steps

The most useful next steps are:

```text
1. Run a baseline analysis on the simplified feature dataset.
2. Compare engineered features against the raw ConvAE and PCA representations.
3. Test whether simplified features produce stronger phase, Puzzler, parent, or questionnaire alignment.
4. Keep the raw ConvAE analysis as the advanced extension.
```

The simplified feature dataset may be especially important because it already contains engineered physiological features, which may be more suitable for phase-level clustering than raw 60-second signal windows.
