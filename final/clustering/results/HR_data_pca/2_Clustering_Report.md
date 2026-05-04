# Analysis Report for: `HR_data_pca.csv`

## Model: K-Means (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0335` | NMI: `0.0384` | p-value: `4.7409e-05` | Cramer's V: `0.2526`

|   K-Means_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       85 |       55 |       65 |
|                 1 |       19 |       49 |       39 |

**Target: Cohort**  
ARI: `-0.0000` | NMI: `0.0026` | p-value: `8.6613e-01` | Cramer's V: `0.0775`

|   K-Means_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     66 |     44 |     32 |     33 |     14 |     16 |
|                 1 |     30 |     28 |     16 |     15 |     10 |      8 |

**Target: Round**  
ARI: `0.0414` | NMI: `0.0508` | p-value: `3.3350e-07` | Cramer's V: `0.3249`

|   K-Means_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |        31 |        54 |        62 |        58 |
|                 1 |        47 |        24 |        16 |        20 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_T_Slope_max | 1.75706e-14 |   0.703237 |
|         0 | EDA_TD_P_Slope_min | 1.0918e-13  |   0.680914 |
|         0 | EDA_TD_P_Kurtosis  | 8.11048e-09 |   0.661779 |
|         0 | EDA_TD_P_std       | 7.68734e-14 |   0.657823 |
|         0 | EDA_TD_P_Skew      | 8.37057e-08 |   0.6493   |
|         0 | EDA_TD_P_Slope_max | 1.15475e-12 |   0.64731  |
|         0 | EDA_TD_P_Max       | 3.7571e-13  |   0.645312 |
|         0 | EDA_TD_T_Slope_min | 1.69166e-12 |   0.635629 |
|         0 | EDA_TD_P_Peaks     | 1.86818e-07 |   0.599441 |
|         0 | EDA_TD_T_std       | 3.93388e-10 |   0.556952 |
|         1 | EDA_TD_T_Slope_max | 1.75706e-14 |   0.703237 |
|         1 | EDA_TD_P_Slope_min | 1.0918e-13  |   0.680914 |
|         1 | EDA_TD_P_Kurtosis  | 8.11048e-09 |   0.661779 |
|         1 | EDA_TD_P_std       | 7.68734e-14 |   0.657823 |
|         1 | EDA_TD_P_Skew      | 8.37057e-08 |   0.6493   |
|         1 | EDA_TD_P_Slope_max | 1.15475e-12 |   0.64731  |
|         1 | EDA_TD_P_Max       | 3.7571e-13  |   0.645312 |
|         1 | EDA_TD_T_Slope_min | 1.69166e-12 |   0.635629 |
|         1 | EDA_TD_P_Peaks     | 1.86818e-07 |   0.599441 |
|         1 | EDA_TD_T_std       | 3.93388e-10 |   0.556952 |

---

## Model: K-Medoids (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0466` | NMI: `0.0559` | p-value: `2.3631e-07` | Cramer's V: `0.3127`

|   K-Medoids_Cluster |   phase1 |   phase2 |   phase3 |
|--------------------:|---------:|---------:|---------:|
|                   0 |       86 |       54 |       84 |
|                   1 |       18 |       50 |       20 |

**Target: Cohort**  
ARI: `-0.0086` | NMI: `0.0043` | p-value: `7.0598e-01` | Cramer's V: `0.0974`

|   K-Medoids_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                   0 |     67 |     49 |     38 |     33 |     19 |     18 |
|                   1 |     29 |     23 |     10 |     15 |      5 |      6 |

**Target: Round**  
ARI: `0.0264` | NMI: `0.0387` | p-value: `1.4036e-05` | Cramer's V: `0.2842`

|   K-Medoids_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------------:|----------:|----------:|----------:|----------:|
|                   0 |        39 |        62 |        64 |        59 |
|                   1 |        39 |        16 |        14 |        19 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature             |       p-val |   Cohens_d |
|----------:|:--------------------|------------:|-----------:|
|         0 | EDA_TD_P_Peaks      | 8.09785e-15 |   1.04324  |
|         0 | EDA_TD_P_Skew       | 3.307e-13   |   0.921221 |
|         0 | EDA_TD_P_Kurtosis   | 3.93944e-14 |   0.851859 |
|         0 | EDA_TD_P_ReT        | 1.31193e-10 |   0.688773 |
|         0 | EDA_TD_T_Slope      | 3.97452e-08 |   0.577724 |
|         0 | EDA_TD_T_Slope_mean | 1.88568e-07 |   0.555215 |
|         0 | HR_TD_Skew          | 7.25788e-05 |   0.514194 |
|         0 | TEMP_TD_AUC         | 0.00526076  |   0.490831 |
|         0 | EDA_TD_T_std        | 0.000992382 |   0.485322 |
|         0 | HR_TD_Slope_min     | 0.000148172 |   0.48359  |
|         1 | EDA_TD_P_Peaks      | 8.09785e-15 |   1.04324  |
|         1 | EDA_TD_P_Skew       | 3.307e-13   |   0.921221 |
|         1 | EDA_TD_P_Kurtosis   | 3.93944e-14 |   0.851859 |
|         1 | EDA_TD_P_ReT        | 1.31193e-10 |   0.688773 |
|         1 | EDA_TD_T_Slope      | 3.97452e-08 |   0.577724 |
|         1 | EDA_TD_T_Slope_mean | 1.88568e-07 |   0.555215 |
|         1 | HR_TD_Skew          | 7.25788e-05 |   0.514194 |
|         1 | TEMP_TD_AUC         | 0.00526076  |   0.490831 |
|         1 | EDA_TD_T_std        | 0.000992382 |   0.485322 |
|         1 | HR_TD_Slope_min     | 0.000148172 |   0.48359  |

---

## Model: GMM (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0066` | NMI: `0.0094` | p-value: `7.4068e-02` | Cramer's V: `0.1292`

|   GMM_Cluster |   phase1 |   phase2 |   phase3 |
|--------------:|---------:|---------:|---------:|
|             0 |       35 |       51 |       41 |
|             1 |       69 |       53 |       63 |

**Target: Cohort**  
ARI: `0.0046` | NMI: `0.0057` | p-value: `5.1950e-01` | Cramer's V: `0.1162`

|   GMM_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|             0 |     33 |     32 |     21 |     18 |     13 |     10 |
|             1 |     63 |     40 |     27 |     30 |     11 |     14 |

**Target: Round**  
ARI: `0.0409` | NMI: `0.0470` | p-value: `1.5310e-06` | Cramer's V: `0.3090`

|   GMM_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------:|----------:|----------:|----------:|----------:|
|             0 |        48 |        38 |        19 |        22 |
|             1 |        30 |        40 |        59 |        56 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature             |       p-val |   Cohens_d |
|----------:|:--------------------|------------:|-----------:|
|         0 | EDA_TD_P_Slope_max  | 2.97862e-08 |   0.539699 |
|         0 | EDA_TD_T_Slope_min  | 6.26333e-08 |   0.536713 |
|         0 | EDA_TD_T_std        | 7.17053e-08 |   0.536467 |
|         0 | EDA_TD_T_Slope_max  | 6.25969e-09 |   0.521504 |
|         0 | EDA_TD_P_Max        | 3.86959e-08 |   0.514289 |
|         0 | EDA_TD_P_std        | 1.71543e-08 |   0.513223 |
|         0 | EDA_TD_P_Slope_min  | 2.53584e-08 |   0.49488  |
|         0 | EDA_TD_T_Max        | 0.000139394 |   0.445815 |
|         0 | EDA_TD_P_Slope_mean | 5.8711e-06  |   0.419548 |
|         0 | HR_TD_Slope_mean    | 0.000390986 |   0.39843  |
|         1 | EDA_TD_P_Slope_max  | 2.97862e-08 |   0.539699 |
|         1 | EDA_TD_T_Slope_min  | 6.26333e-08 |   0.536713 |
|         1 | EDA_TD_T_std        | 7.17053e-08 |   0.536467 |
|         1 | EDA_TD_T_Slope_max  | 6.25969e-09 |   0.521504 |
|         1 | EDA_TD_P_Max        | 3.86959e-08 |   0.514289 |
|         1 | EDA_TD_P_std        | 1.71543e-08 |   0.513223 |
|         1 | EDA_TD_P_Slope_min  | 2.53584e-08 |   0.49488  |
|         1 | EDA_TD_T_Max        | 0.000139394 |   0.445815 |
|         1 | EDA_TD_P_Slope_mean | 5.8711e-06  |   0.419548 |
|         1 | HR_TD_Slope_mean    | 0.000390986 |   0.39843  |

---

