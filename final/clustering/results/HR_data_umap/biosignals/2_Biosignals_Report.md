# Biosignals Analysis Report: HR_data_umap.csv

## Model: K-Means (K=2)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0914` | NMI: `0.0867` | p-value: `1.3380e-10` | Cramer's V: `0.3818`

|   K-Means_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|----------------------:|---------:|---------:|---------:|
|                     0 |       66 |       19 |       52 |
|                     1 |       38 |       85 |       52 |

**Target: Cohort**  
- ARI: `-0.0051` | NMI: `0.0006` | p-value: `9.9345e-01` | Cramer's V: `0.0385`

|   K-Means_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|----------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                     0 |     44 |     30 |     22 |     21 |     10 |     10 |
|                     1 |     52 |     42 |     26 |     27 |     14 |     14 |

**Target: Round**  
- ARI: `0.0581` | NMI: `0.0659` | p-value: `1.1120e-08` | Cramer's V: `0.3577`

|   K-Means_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|----------------------:|----------:|----------:|----------:|----------:|
|                     0 |        13 |        40 |        51 |        33 |
|                     1 |        65 |        38 |        27 |        45 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature           |       p-val |   Cohens_d |
|----------:|:------------------|------------:|-----------:|
|         0 | EDA_TD_P_Skew     | 2.1856e-12  |   0.849402 |
|         0 | EDA_TD_P_Kurtosis | 1.31791e-11 |   0.820243 |
|         0 | EDA_TD_P_Peaks    | 3.52141e-12 |   0.789616 |
|         0 | HR_TD_AUC         | 1.39713e-07 |   0.600134 |
|         0 | TEMP_TD_AUC       | 2.47129e-06 |   0.542173 |
|         1 | EDA_TD_P_Skew     | 2.1856e-12  |   0.849402 |
|         1 | EDA_TD_P_Kurtosis | 1.31791e-11 |   0.820243 |
|         1 | EDA_TD_P_Peaks    | 3.52141e-12 |   0.789616 |
|         1 | HR_TD_AUC         | 1.39713e-07 |   0.600134 |
|         1 | TEMP_TD_AUC       | 2.47129e-06 |   0.542173 |

## Model: K-Medoids (K=2)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0728` | NMI: `0.0676` | p-value: `1.3683e-08` | Cramer's V: `0.3407`

|   K-Medoids_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------------:|---------:|---------:|---------:|
|                       0 |       38 |       80 |       50 |
|                       1 |       66 |       24 |       54 |

**Target: Cohort**  
- ARI: `-0.0044` | NMI: `0.0020` | p-value: `9.1191e-01` | Cramer's V: `0.0696`

|   K-Medoids_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                       0 |     47 |     40 |     26 |     27 |     14 |     14 |
|                       1 |     49 |     32 |     22 |     21 |     10 |     10 |

**Target: Round**  
- ARI: `0.0530` | NMI: `0.0592` | p-value: `6.7465e-08` | Cramer's V: `0.3407`

|   K-Medoids_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------------:|----------:|----------:|----------:|----------:|
|                       0 |        63 |        36 |        27 |        42 |
|                       1 |        15 |        42 |        51 |        36 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature           |       p-val |   Cohens_d |
|----------:|:------------------|------------:|-----------:|
|         0 | EDA_TD_P_Skew     | 2.10009e-12 |   0.830934 |
|         0 | EDA_TD_P_Kurtosis | 1.4969e-11  |   0.793755 |
|         0 | EDA_TD_P_Peaks    | 2.61132e-11 |   0.745795 |
|         0 | HR_TD_AUC         | 2.57796e-07 |   0.57559  |
|         0 | EDA_TD_P_ReT      | 1.65248e-05 |   0.506422 |
|         1 | EDA_TD_P_Skew     | 2.10009e-12 |   0.830934 |
|         1 | EDA_TD_P_Kurtosis | 1.4969e-11  |   0.793755 |
|         1 | EDA_TD_P_Peaks    | 2.61132e-11 |   0.745795 |
|         1 | HR_TD_AUC         | 2.57796e-07 |   0.57559  |
|         1 | EDA_TD_P_ReT      | 1.65248e-05 |   0.506422 |

## Model: GMM (K=2)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0552` | NMI: `0.0558` | p-value: `4.8572e-07` | Cramer's V: `0.3053`

|   GMM_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       54 |       20 |       52 |
|                 1 |       50 |       84 |       52 |

**Target: Cohort**  
- ARI: `-0.0060` | NMI: `0.0015` | p-value: `9.5416e-01` | Cramer's V: `0.0594`

|   GMM_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     41 |     28 |     21 |     19 |      8 |      9 |
|                 1 |     55 |     44 |     27 |     29 |     16 |     15 |

**Target: Round**  
- ARI: `0.0649` | NMI: `0.0780` | p-value: `6.7762e-10` | Cramer's V: `0.3825`

|   GMM_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |         9 |        38 |        49 |        30 |
|                 1 |        69 |        40 |        29 |        48 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature           |       p-val |   Cohens_d |
|----------:|:------------------|------------:|-----------:|
|         0 | EDA_TD_P_Skew     | 6.06754e-13 |   0.862027 |
|         0 | EDA_TD_P_Peaks    | 1.18051e-12 |   0.809602 |
|         0 | EDA_TD_P_Kurtosis | 1.28352e-11 |   0.79842  |
|         0 | HR_TD_AUC         | 8.33477e-08 |   0.607037 |
|         0 | TEMP_TD_AUC       | 1.25443e-06 |   0.581333 |
|         1 | EDA_TD_P_Skew     | 6.06754e-13 |   0.862027 |
|         1 | EDA_TD_P_Peaks    | 1.18051e-12 |   0.809602 |
|         1 | EDA_TD_P_Kurtosis | 1.28352e-11 |   0.79842  |
|         1 | HR_TD_AUC         | 8.33477e-08 |   0.607037 |
|         1 | TEMP_TD_AUC       | 1.25443e-06 |   0.581333 |

