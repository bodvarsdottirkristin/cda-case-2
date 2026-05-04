# Analysis Report for: `HR_data_spca.csv`

## Model: K-Means (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0320` | NMI: `0.0391` | p-value: `4.0539e-05` | Cramer's V: `0.2546`

|   K-Means_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       17 |       47 |       33 |
|                 1 |       87 |       57 |       71 |

**Target: Cohort**  
ARI: `0.0043` | NMI: `0.0062` | p-value: `4.7531e-01` | Cramer's V: `0.1206`

|   K-Means_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     25 |     29 |     14 |     13 |      8 |      8 |
|                 1 |     71 |     43 |     34 |     35 |     16 |     16 |

**Target: Round**  
ARI: `0.0190` | NMI: `0.0278` | p-value: `4.9339e-04` | Cramer's V: `0.2386`

|   K-Means_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |        38 |        25 |        16 |        18 |
|                 1 |        40 |        53 |        62 |        60 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_T_Slope_max | 1.26507e-16 |   0.855793 |
|         0 | EDA_TD_P_Slope_min | 2.58665e-15 |   0.809603 |
|         0 | EDA_TD_T_Slope_min | 2.04599e-14 |   0.795639 |
|         0 | EDA_TD_P_std       | 9.21887e-16 |   0.793739 |
|         0 | EDA_TD_P_Max       | 5.76201e-15 |   0.791547 |
|         0 | EDA_TD_P_Slope_max | 1.32602e-14 |   0.778353 |
|         0 | EDA_TD_T_std       | 3.04515e-11 |   0.653006 |
|         0 | EDA_TD_P_Kurtosis  | 1.48982e-08 |   0.651346 |
|         0 | EDA_TD_P_Skew      | 2.28371e-07 |   0.641627 |
|         0 | EDA_TD_P_Peaks     | 1.94972e-06 |   0.556637 |
|         1 | EDA_TD_T_Slope_max | 1.26507e-16 |   0.855793 |
|         1 | EDA_TD_P_Slope_min | 2.58665e-15 |   0.809603 |
|         1 | EDA_TD_T_Slope_min | 2.04599e-14 |   0.795639 |
|         1 | EDA_TD_P_std       | 9.21887e-16 |   0.793739 |
|         1 | EDA_TD_P_Max       | 5.76201e-15 |   0.791547 |
|         1 | EDA_TD_P_Slope_max | 1.32602e-14 |   0.778353 |
|         1 | EDA_TD_T_std       | 3.04515e-11 |   0.653006 |
|         1 | EDA_TD_P_Kurtosis  | 1.48982e-08 |   0.651346 |
|         1 | EDA_TD_P_Skew      | 2.28371e-07 |   0.641627 |
|         1 | EDA_TD_P_Peaks     | 1.94972e-06 |   0.556637 |

---

## Model: K-Medoids (Optimal K=3)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0079` | NMI: `0.0221` | p-value: `7.8430e-03` | Cramer's V: `0.1489`

|   K-Medoids_Cluster |   phase1 |   phase2 |   phase3 |
|--------------------:|---------:|---------:|---------:|
|                   0 |       51 |       53 |       55 |
|                   1 |       47 |       29 |       35 |
|                   2 |        6 |       22 |       14 |

**Target: Cohort**  
ARI: `-0.0011` | NMI: `0.0049` | p-value: `9.4156e-01` | Cramer's V: `0.0813`

|   K-Medoids_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                   0 |     48 |     41 |     22 |     23 |     14 |     11 |
|                   1 |     37 |     22 |     17 |     18 |      8 |      9 |
|                   2 |     11 |      9 |      9 |      7 |      2 |      4 |

**Target: Round**  
ARI: `0.0280` | NMI: `0.0526` | p-value: `1.7702e-07` | Cramer's V: `0.2597`

|   K-Medoids_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------------:|----------:|----------:|----------:|----------:|
|                   0 |        33 |        52 |        39 |        35 |
|                   1 |        20 |        20 |        36 |        35 |
|                   2 |        25 |         6 |         3 |         8 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature             |       p-val |   Cohens_d |
|----------:|:--------------------|------------:|-----------:|
|         0 | HR_TD_Max           | 1.81539e-06 |   0.516124 |
|         0 | HR_TD_std           | 5.32102e-05 |   0.507904 |
|         0 | HR_TD_Slope_min     | 1.55701e-05 |   0.468364 |
|         0 | TEMP_TD_Kurtosis    | 1.20741e-05 |   0.445716 |
|         0 | HR_TD_Mean          | 0.00121063  |   0.363144 |
|         0 | EDA_TD_T_Slope_max  | 0.0464777   |   0.307979 |
|         0 | EDA_TD_P_Max        | 0.0190889   |   0.300536 |
|         0 | TEMP_TD_Min         | 0.00704473  |   0.299179 |
|         0 | TEMP_TD_Mean        | 0.00583685  |   0.29408  |
|         0 | TEMP_TD_Median      | 0.00571355  |   0.293558 |
|         1 | EDA_TD_T_Slope_max  | 4.67712e-09 |   0.608615 |
|         1 | EDA_TD_P_Max        | 1.14634e-08 |   0.59361  |
|         1 | EDA_TD_P_std        | 1.27659e-08 |   0.59342  |
|         1 | EDA_TD_P_Slope_max  | 1.47807e-07 |   0.557019 |
|         1 | EDA_TD_P_Slope_min  | 3.16845e-08 |   0.536744 |
|         1 | EDA_TD_T_std        | 9.85877e-06 |   0.525747 |
|         1 | EDA_TD_T_Slope_min  | 1.81593e-07 |   0.497651 |
|         1 | TEMP_TD_Kurtosis    | 8.35731e-05 |   0.476724 |
|         1 | EDA_TD_T_Skew       | 0.000228837 |   0.469779 |
|         1 | HR_TD_Max           | 6.44113e-06 |   0.448037 |
|         2 | EDA_TD_P_Skew       | 2.51563e-05 |   0.655751 |
|         2 | EDA_TD_P_Kurtosis   | 3.64293e-06 |   0.650462 |
|         2 | EDA_TD_P_Slope_min  | 7.18744e-07 |   0.550907 |
|         2 | EDA_TD_T_Slope      | 2.11865e-05 |   0.537374 |
|         2 | EDA_TD_P_std        | 1.00818e-06 |   0.534824 |
|         2 | EDA_TD_P_Peaks      | 0.00122088  |   0.525792 |
|         2 | EDA_TD_T_std        | 1.88794e-05 |   0.516043 |
|         2 | EDA_TD_T_Slope_mean | 8.10533e-05 |   0.506469 |
|         2 | EDA_TD_T_Slope_max  | 1.1576e-07  |   0.503714 |
|         2 | EDA_TD_P_Max        | 4.83229e-06 |   0.492485 |

---

## Model: GMM (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0351` | NMI: `0.0355` | p-value: `6.6148e-05` | Cramer's V: `0.2484`

|   GMM_Cluster |   phase1 |   phase2 |   phase3 |
|--------------:|---------:|---------:|---------:|
|             0 |       26 |       57 |       42 |
|             1 |       78 |       47 |       62 |

**Target: Cohort**  
ARI: `-0.0035` | NMI: `0.0044` | p-value: `6.6336e-01` | Cramer's V: `0.1019`

|   GMM_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|             0 |     37 |     34 |     17 |     21 |      8 |      8 |
|             1 |     59 |     38 |     31 |     27 |     16 |     16 |

**Target: Round**  
ARI: `0.0044` | NMI: `0.0093` | p-value: `1.1592e-01` | Cramer's V: `0.1377`

|   GMM_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------:|----------:|----------:|----------:|----------:|
|             0 |        38 |        34 |        29 |        24 |
|             1 |        40 |        44 |        49 |        54 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_T_Skew      | 3.89563e-06 |   0.590334 |
|         0 | EDA_TD_T_Slope_min | 2.12233e-08 |   0.586609 |
|         0 | EDA_TD_T_Slope_max | 2.07013e-10 |   0.581193 |
|         0 | EDA_TD_P_Peaks     | 8.64225e-06 |   0.523697 |
|         0 | EDA_TD_T_Kurtosis  | 0.00107703  |   0.51914  |
|         0 | EDA_TD_P_Slope_max | 2.27602e-07 |   0.514164 |
|         0 | EDA_TD_P_ReT       | 3.4092e-06  |   0.50787  |
|         0 | EDA_TD_P_Max       | 8.63794e-08 |   0.507845 |
|         0 | EDA_TD_P_std       | 2.3707e-08  |   0.481771 |
|         0 | EDA_TD_P_Skew      | 6.84374e-05 |   0.468583 |
|         1 | EDA_TD_T_Skew      | 3.89563e-06 |   0.590334 |
|         1 | EDA_TD_T_Slope_min | 2.12233e-08 |   0.586609 |
|         1 | EDA_TD_T_Slope_max | 2.07013e-10 |   0.581193 |
|         1 | EDA_TD_P_Peaks     | 8.64225e-06 |   0.523697 |
|         1 | EDA_TD_T_Kurtosis  | 0.00107703  |   0.51914  |
|         1 | EDA_TD_P_Slope_max | 2.27602e-07 |   0.514164 |
|         1 | EDA_TD_P_ReT       | 3.4092e-06  |   0.50787  |
|         1 | EDA_TD_P_Max       | 8.63794e-08 |   0.507845 |
|         1 | EDA_TD_P_std       | 2.3707e-08  |   0.481771 |
|         1 | EDA_TD_P_Skew      | 6.84374e-05 |   0.468583 |

---

