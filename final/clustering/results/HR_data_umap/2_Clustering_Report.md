# Analysis Report for: `HR_data_umap.csv`

## Model: K-Means (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0831` | NMI: `0.0774` | p-value: `8.9056e-10` | Cramer's V: `0.3655`

|   K-Means_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       23 |       69 |       43 |
|                 1 |       81 |       35 |       61 |

**Target: Cohort**  
ARI: `-0.0042` | NMI: `0.0021` | p-value: `9.0665e-01` | Cramer's V: `0.0706`

|   K-Means_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     41 |     35 |     20 |     19 |      9 |     11 |
|                 1 |     55 |     37 |     28 |     29 |     15 |     13 |

**Target: Round**  
ARI: `0.0318` | NMI: `0.0364` | p-value: `3.4415e-05` | Cramer's V: `0.2735`

|   K-Means_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |        51 |        33 |        23 |        28 |
|                 1 |        27 |        45 |        55 |        50 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_P_Skew      | 1.60026e-12 |   0.85432  |
|         0 | EDA_TD_P_Kurtosis  | 1.21626e-12 |   0.805269 |
|         0 | EDA_TD_P_Peaks     | 2.9061e-11  |   0.781732 |
|         0 | EDA_TD_P_std       | 3.11711e-10 |   0.631743 |
|         0 | EDA_TD_T_Slope_max | 1.18657e-10 |   0.613234 |
|         0 | EDA_TD_P_Slope_min | 2.46885e-09 |   0.593863 |
|         0 | EDA_TD_P_Max       | 3.23569e-09 |   0.585906 |
|         0 | EDA_TD_P_ReT       | 7.02907e-08 |   0.577047 |
|         0 | EDA_TD_P_Slope_max | 1.78509e-08 |   0.561907 |
|         0 | EDA_TD_T_std       | 5.86031e-07 |   0.561118 |
|         1 | EDA_TD_P_Skew      | 1.60026e-12 |   0.85432  |
|         1 | EDA_TD_P_Kurtosis  | 1.21626e-12 |   0.805269 |
|         1 | EDA_TD_P_Peaks     | 2.9061e-11  |   0.781732 |
|         1 | EDA_TD_P_std       | 3.11711e-10 |   0.631743 |
|         1 | EDA_TD_T_Slope_max | 1.18657e-10 |   0.613234 |
|         1 | EDA_TD_P_Slope_min | 2.46885e-09 |   0.593863 |
|         1 | EDA_TD_P_Max       | 3.23569e-09 |   0.585906 |
|         1 | EDA_TD_P_ReT       | 7.02907e-08 |   0.577047 |
|         1 | EDA_TD_P_Slope_max | 1.78509e-08 |   0.561907 |
|         1 | EDA_TD_T_std       | 5.86031e-07 |   0.561118 |

---

## Model: K-Medoids (Optimal K=2)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0532` | NMI: `0.0543` | p-value: `6.3505e-07` | Cramer's V: `0.3024`

|   K-Medoids_Cluster |   phase1 |   phase2 |   phase3 |
|--------------------:|---------:|---------:|---------:|
|                   0 |       20 |       57 |       44 |
|                   1 |       84 |       47 |       60 |

**Target: Cohort**  
ARI: `-0.0049` | NMI: `0.0005` | p-value: `9.9614e-01` | Cramer's V: `0.0344`

|   K-Medoids_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                   0 |     38 |     28 |     19 |     17 |     10 |      9 |
|                   1 |     58 |     44 |     29 |     31 |     14 |     15 |

**Target: Round**  
ARI: `0.0291` | NMI: `0.0352` | p-value: `5.1217e-05` | Cramer's V: `0.2686`

|   K-Medoids_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------------:|----------:|----------:|----------:|----------:|
|                   0 |        46 |        32 |        19 |        24 |
|                   1 |        32 |        46 |        59 |        54 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_P_Skew      | 2.48097e-09 |   0.709638 |
|         0 | EDA_TD_P_Kurtosis  | 2.01998e-09 |   0.688929 |
|         0 | EDA_TD_P_Peaks     | 2.33213e-08 |   0.641513 |
|         0 | HR_TD_Skew         | 1.90751e-05 |   0.51096  |
|         0 | EDA_TD_P_ReT       | 3.16221e-06 |   0.498497 |
|         0 | EDA_TD_T_Slope_max | 2.24771e-08 |   0.492989 |
|         0 | EDA_TD_P_std       | 8.87655e-08 |   0.445673 |
|         0 | EDA_TD_P_Slope_min | 1.75664e-07 |   0.43859  |
|         0 | EDA_TD_T_std       | 5.84947e-06 |   0.435079 |
|         0 | EDA_TD_P_Max       | 4.28216e-07 |   0.422621 |
|         1 | EDA_TD_P_Skew      | 2.48097e-09 |   0.709638 |
|         1 | EDA_TD_P_Kurtosis  | 2.01998e-09 |   0.688929 |
|         1 | EDA_TD_P_Peaks     | 2.33213e-08 |   0.641513 |
|         1 | HR_TD_Skew         | 1.90751e-05 |   0.51096  |
|         1 | EDA_TD_P_ReT       | 3.16221e-06 |   0.498497 |
|         1 | EDA_TD_T_Slope_max | 2.24771e-08 |   0.492989 |
|         1 | EDA_TD_P_std       | 8.87655e-08 |   0.445673 |
|         1 | EDA_TD_P_Slope_min | 1.75664e-07 |   0.43859  |
|         1 | EDA_TD_T_std       | 5.84947e-06 |   0.435079 |
|         1 | EDA_TD_P_Max       | 4.28216e-07 |   0.422621 |

---

## Model: GMM (Optimal K=5)

### Contingency Tables & Alignment Metrics
**Target: Phase**  
ARI: `0.0478` | NMI: `0.0709` | p-value: `2.0514e-08` | Cramer's V: `0.2874`

|   GMM_Cluster |   phase1 |   phase2 |   phase3 |
|--------------:|---------:|---------:|---------:|
|             0 |       14 |       20 |       20 |
|             1 |       25 |        3 |       30 |
|             2 |        8 |       39 |       22 |
|             3 |       24 |       20 |       12 |
|             4 |       33 |       22 |       20 |

**Target: Cohort**  
ARI: `-0.0007` | NMI: `0.0183` | p-value: `5.2979e-01` | Cramer's V: `0.1230`

|   GMM_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|--------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|             0 |     16 |      9 |     10 |      7 |      4 |      8 |
|             1 |     18 |     14 |      9 |     11 |      2 |      4 |
|             2 |     20 |     21 |      9 |      9 |      5 |      5 |
|             3 |     24 |      8 |     10 |      7 |      5 |      2 |
|             4 |     18 |     20 |     10 |     14 |      8 |      5 |

**Target: Round**  
ARI: `0.0196` | NMI: `0.0377` | p-value: `1.1712e-03` | Cramer's V: `0.1862`

|   GMM_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|--------------:|----------:|----------:|----------:|----------:|
|             0 |        14 |        15 |        13 |        12 |
|             1 |         5 |        14 |        20 |        19 |
|             2 |        30 |        19 |         6 |        14 |
|             3 |        12 |         9 |        19 |        16 |
|             4 |        17 |        21 |        20 |        17 |

### Top 10 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | HR_TD_Skew         | 0.000600544 |   0.575069 |
|         0 | HR_TD_Slope_min    | 6.1476e-05  |   0.539309 |
|         0 | HR_TD_std          | 0.00296     |   0.45963  |
|         0 | TEMP_TD_Kurtosis   | 0.130942    |   0.327795 |
|         0 | TEMP_TD_Slope      | 0.0227449   |   0.317202 |
|         0 | HR_TD_Max          | 0.119506    |   0.317135 |
|         0 | TEMP_TD_Slope_mean | 0.0183696   |   0.316157 |
|         0 | EDA_TD_T_Slope     | 0.0467947   |   0.313208 |
|         0 | HR_TD_Slope_mean   | 0.00781418  |   0.305695 |
|         0 | HR_TD_Slope        | 0.00740096  |   0.303878 |
|         1 | EDA_TD_P_Peaks     | 3.39972e-15 |   1.12961  |
|         1 | EDA_TD_P_Skew      | 2.17202e-11 |   1.06725  |
|         1 | EDA_TD_P_Kurtosis  | 1.7226e-10  |   1.04302  |
|         1 | EDA_TD_P_ReT       | 4.04213e-08 |   0.937703 |
|         1 | HR_TD_AUC          | 3.16106e-10 |   0.890161 |
|         1 | HR_TD_Mean         | 5.82649e-07 |   0.634765 |
|         1 | HR_TD_Max          | 3.574e-06   |   0.576406 |
|         1 | HR_TD_Median       | 7.35712e-06 |   0.570669 |
|         1 | EDA_TD_P_RT        | 0.00212934  |   0.568992 |
|         1 | HR_TD_Slope_max    | 1.14131e-07 |   0.510688 |
|         2 | EDA_TD_P_Peaks     | 2.84653e-08 |   0.882942 |
|         2 | EDA_TD_P_Skew      | 4.59339e-08 |   0.748082 |
|         2 | EDA_TD_T_Slope_max | 4.48759e-10 |   0.723348 |
|         2 | EDA_TD_T_Slope_min | 4.18159e-08 |   0.714455 |
|         2 | EDA_TD_P_Kurtosis  | 5.03781e-09 |   0.70954  |
|         2 | EDA_TD_P_Max       | 1.03081e-07 |   0.675352 |
|         2 | EDA_TD_P_std       | 1.95345e-08 |   0.660224 |
|         2 | EDA_TD_P_Slope_min | 1.35456e-07 |   0.634396 |
|         2 | EDA_TD_P_Slope_max | 6.13161e-07 |   0.601059 |
|         2 | EDA_TD_T_std       | 3.2739e-06  |   0.595852 |
|         3 | EDA_TD_P_Slope_min | 1.68967e-07 |   0.654738 |
|         3 | EDA_TD_P_Max       | 3.11648e-08 |   0.648891 |
|         3 | EDA_TD_P_Slope_max | 1.7975e-07  |   0.639272 |
|         3 | EDA_TD_T_Slope_max | 1.98051e-07 |   0.60904  |
|         3 | EDA_TD_P_std       | 1.36542e-07 |   0.601084 |
|         3 | EDA_TD_T_std       | 1.93959e-06 |   0.552692 |
|         3 | EDA_TD_T_Slope_min | 1.65758e-05 |   0.539208 |
|         3 | EDA_TD_T_Max       | 0.000131191 |   0.49418  |
|         3 | HR_TD_Max          | 0.0158959   |   0.428741 |
|         3 | EDA_TD_T_Mean      | 0.000872498 |   0.422713 |
|         4 | HR_TD_Max          | 3.59601e-22 |   1.71586  |
|         4 | HR_TD_std          | 1.72985e-19 |   1.67331  |
|         4 | HR_TD_Slope_min    | 2.1402e-19  |   1.36745  |
|         4 | HR_TD_Slope_max    | 8.25032e-16 |   0.890788 |
|         4 | HR_TD_Mean         | 9.52555e-08 |   0.820573 |
|         4 | HR_TD_AUC          | 2.66313e-06 |   0.691479 |
|         4 | HR_TD_Skew         | 1.22218e-05 |   0.608174 |
|         4 | HR_TD_Median       | 0.00200643  |   0.542622 |
|         4 | HR_TD_Slope        | 0.105411    |   0.394504 |
|         4 | HR_TD_Slope_mean   | 0.10716     |   0.39228  |

---

