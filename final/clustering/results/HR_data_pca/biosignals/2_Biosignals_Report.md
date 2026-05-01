# Biosignals Analysis Report: HR_data_pca.csv

## Model: K-Means (K=3)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0476` | NMI: `0.0530` | p-value: `1.0649e-06` | Cramer's V: `0.2308`

|   K-Means_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|----------------------:|---------:|---------:|---------:|
|                     0 |       63 |       32 |       59 |
|                     1 |       30 |       47 |       17 |
|                     2 |       11 |       25 |       28 |

**Target: Cohort**  
- ARI: `-0.0015` | NMI: `0.0065` | p-value: `8.5803e-01` | Cramer's V: `0.0936`

|   K-Means_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|----------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                     0 |     50 |     30 |     27 |     24 |     10 |     13 |
|                     1 |     30 |     23 |     11 |     15 |      9 |      6 |
|                     2 |     16 |     19 |     10 |      9 |      5 |      5 |

**Target: Round**  
- ARI: `0.0296` | NMI: `0.0393` | p-value: `1.0338e-04` | Cramer's V: `0.2110`

|   K-Means_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|----------------------:|----------:|----------:|----------:|----------:|
|                     0 |        21 |        39 |        48 |        46 |
|                     1 |        34 |        24 |        22 |        14 |
|                     2 |        23 |        15 |         8 |        18 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_P_Peaks     | 2.61791e-11 |   0.783332 |
|         0 | HR_TD_std          | 8.02814e-09 |   0.770449 |
|         0 | HR_TD_Max          | 2.35983e-10 |   0.764312 |
|         0 | HR_TD_AUC          | 1.62476e-09 |   0.700481 |
|         0 | EDA_TD_P_Skew      | 6.56232e-09 |   0.648467 |
|         1 | HR_TD_std          | 5.21127e-20 |   1.51369  |
|         1 | HR_TD_Max          | 1.38298e-22 |   1.4619   |
|         1 | HR_TD_Slope_min    | 1.12129e-15 |   1.05669  |
|         1 | HR_TD_AUC          | 1.79109e-11 |   0.879398 |
|         1 | HR_TD_Slope_max    | 1.65974e-17 |   0.787842 |
|         2 | EDA_TD_P_Slope_min | 4.49901e-11 |   0.817999 |
|         2 | EDA_TD_T_Slope_max | 3.92607e-11 |   0.812653 |
|         2 | EDA_TD_P_Slope_max | 1.15444e-10 |   0.779086 |
|         2 | EDA_TD_T_Slope_min | 1.04926e-09 |   0.775463 |
|         2 | EDA_TD_P_std       | 7.18718e-11 |   0.760706 |

## Model: K-Medoids (K=3)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0723` | NMI: `0.0824` | p-value: `6.9336e-10` | Cramer's V: `0.2792`

|   K-Medoids_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------------:|---------:|---------:|---------:|
|                       0 |       47 |       63 |       27 |
|                       1 |       11 |       30 |       32 |
|                       2 |       46 |       11 |       45 |

**Target: Cohort**  
- ARI: `0.0004` | NMI: `0.0067` | p-value: `8.5211e-01` | Cramer's V: `0.0942`

|   K-Medoids_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                       0 |     46 |     30 |     23 |     21 |     10 |      7 |
|                       1 |     19 |     19 |     13 |      9 |      6 |      7 |
|                       2 |     31 |     23 |     12 |     18 |      8 |     10 |

**Target: Round**  
- ARI: `0.0155` | NMI: `0.0294` | p-value: `1.6054e-03` | Cramer's V: `0.1849`

|   K-Medoids_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------------:|----------:|----------:|----------:|----------:|
|                       0 |        37 |        35 |        33 |        32 |
|                       1 |        28 |        16 |         9 |        20 |
|                       2 |        13 |        27 |        36 |        26 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | HR_TD_Max          | 1.16928e-21 |   1.15737  |
|         0 | HR_TD_std          | 2.85298e-15 |   1.0598   |
|         0 | HR_TD_Slope_min    | 4.02854e-11 |   0.772726 |
|         0 | HR_TD_AUC          | 1.68851e-10 |   0.730017 |
|         0 | HR_TD_Mean         | 7.85924e-09 |   0.645156 |
|         1 | EDA_TD_P_Slope_min | 2.94701e-11 |   0.727028 |
|         1 | EDA_TD_P_Slope_max | 1.19332e-10 |   0.688497 |
|         1 | EDA_TD_T_Slope_max | 2.11128e-11 |   0.679757 |
|         1 | EDA_TD_P_std       | 6.6814e-11  |   0.646896 |
|         1 | EDA_TD_T_Slope_min | 6.94119e-10 |   0.642156 |
|         2 | EDA_TD_P_Peaks     | 4.45466e-12 |   0.879852 |
|         2 | HR_TD_AUC          | 1.14541e-11 |   0.819414 |
|         2 | HR_TD_Max          | 3.49587e-12 |   0.808525 |
|         2 | EDA_TD_P_Skew      | 5.94389e-09 |   0.721497 |
|         2 | EDA_TD_P_Kurtosis  | 6.08826e-09 |   0.705809 |

## Model: GMM (K=2)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0339` | NMI: `0.0460` | p-value: `9.4077e-06` | Cramer's V: `0.2724`

|   GMM_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       91 |       60 |       75 |
|                 1 |       13 |       44 |       29 |

**Target: Cohort**  
- ARI: `0.0009` | NMI: `0.0021` | p-value: `9.1373e-01` | Cramer's V: `0.0692`

|   GMM_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     73 |     49 |     34 |     35 |     17 |     18 |
|                 1 |     23 |     23 |     14 |     13 |      7 |      6 |

**Target: Round**  
- ARI: `0.0265` | NMI: `0.0405` | p-value: `1.0767e-05` | Cramer's V: `0.2873`

|   GMM_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |        40 |        59 |        67 |        60 |
|                 1 |        38 |        19 |        11 |        18 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_P_Peaks     | 4.41676e-07 |   0.688835 |
|         0 | EDA_TD_T_Slope_max | 4.41001e-12 |   0.677866 |
|         0 | EDA_TD_P_Skew      | 6.03448e-08 |   0.660402 |
|         0 | EDA_TD_P_Kurtosis  | 4.98875e-09 |   0.635257 |
|         0 | EDA_TD_P_Slope_min | 2.53119e-10 |   0.626738 |
|         1 | EDA_TD_P_Peaks     | 4.41676e-07 |   0.688835 |
|         1 | EDA_TD_T_Slope_max | 4.41001e-12 |   0.677866 |
|         1 | EDA_TD_P_Skew      | 6.03448e-08 |   0.660402 |
|         1 | EDA_TD_P_Kurtosis  | 4.98875e-09 |   0.635257 |
|         1 | EDA_TD_P_Slope_min | 2.53119e-10 |   0.626738 |

