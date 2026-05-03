# Biosignals Analysis Report: HR_data_spca.csv

## Model: K-Means (K=3)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0112` | NMI: `0.0254` | p-value: `4.6185e-03` | Cramer's V: `0.1553`

|   K-Means_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|----------------------:|---------:|---------:|---------:|
|                     0 |       38 |       31 |       22 |
|                     1 |       58 |       48 |       58 |
|                     2 |        8 |       25 |       24 |

**Target: Cohort**  
- ARI: `-0.0023` | NMI: `0.0041` | p-value: `9.7085e-01` | Cramer's V: `0.0737`

|   K-Means_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|----------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                     0 |     31 |     21 |     13 |     12 |      8 |      6 |
|                     1 |     51 |     37 |     25 |     27 |     10 |     14 |
|                     2 |     14 |     14 |     10 |      9 |      6 |      4 |

**Target: Round**  
- ARI: `0.0037` | NMI: `0.0126` | p-value: `1.6379e-01` | Cramer's V: `0.1213`

|   K-Means_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|----------------------:|----------:|----------:|----------:|----------:|
|                     0 |        24 |        26 |        21 |        20 |
|                     1 |        34 |        38 |        49 |        43 |
|                     2 |        20 |        14 |         8 |        15 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | HR_TD_std          | 1.10405e-25 |   1.76384  |
|         0 | HR_TD_Max          | 1.50583e-23 |   1.48193  |
|         0 | HR_TD_Slope_min    | 1.77696e-23 |   1.38602  |
|         0 | HR_TD_Slope_max    | 1.326e-18   |   0.952773 |
|         0 | HR_TD_Mean         | 1.83286e-05 |   0.568039 |
|         1 | HR_TD_std          | 5.56641e-18 |   1.10322  |
|         1 | HR_TD_Max          | 5.84273e-16 |   0.960533 |
|         1 | HR_TD_Slope_min    | 1.14372e-15 |   0.877369 |
|         1 | HR_TD_Slope_max    | 1.3132e-15  |   0.730425 |
|         1 | HR_TD_AUC          | 4.93187e-05 |   0.48354  |
|         2 | EDA_TD_T_Slope_min | 9.67679e-09 |   0.83602  |
|         2 | EDA_TD_T_Slope_max | 1.36237e-10 |   0.835198 |
|         2 | EDA_TD_P_Slope_min | 1.18522e-09 |   0.775468 |
|         2 | EDA_TD_P_Slope_max | 2.67032e-09 |   0.761934 |
|         2 | EDA_TD_P_Max       | 2.34661e-09 |   0.760372 |

## Model: K-Medoids (K=3)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0151` | NMI: `0.0264` | p-value: `2.4199e-03` | Cramer's V: `0.1626`

|   K-Medoids_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------------:|---------:|---------:|---------:|
|                       0 |       12 |       29 |       32 |
|                       1 |       42 |       32 |       22 |
|                       2 |       50 |       43 |       50 |

**Target: Cohort**  
- ARI: `0.0002` | NMI: `0.0084` | p-value: `7.1924e-01` | Cramer's V: `0.1064`

|   K-Medoids_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                       0 |     18 |     19 |     13 |      9 |      8 |      6 |
|                       1 |     32 |     25 |     12 |     13 |      8 |      6 |
|                       2 |     46 |     28 |     23 |     26 |      8 |     12 |

**Target: Round**  
- ARI: `0.0132` | NMI: `0.0218` | p-value: `9.5894e-03` | Cramer's V: `0.1647`

|   K-Medoids_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------------:|----------:|----------:|----------:|----------:|
|                       0 |        29 |        17 |        11 |        16 |
|                       1 |        24 |        28 |        22 |        22 |
|                       2 |        25 |        33 |        45 |        40 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | EDA_TD_P_Slope_min | 4.79301e-12 |   0.778288 |
|         0 | EDA_TD_P_Slope_max | 3.251e-12   |   0.767948 |
|         0 | EDA_TD_T_Slope_max | 1.61474e-12 |   0.741568 |
|         0 | EDA_TD_P_Max       | 6.09203e-12 |   0.736694 |
|         0 | EDA_TD_T_Slope_min | 1.74371e-10 |   0.719604 |
|         1 | HR_TD_std          | 4.32907e-28 |   1.79259  |
|         1 | HR_TD_Max          | 1.84626e-24 |   1.47124  |
|         1 | HR_TD_Slope_min    | 9.24697e-24 |   1.31176  |
|         1 | HR_TD_Slope_max    | 8.52335e-20 |   0.933286 |
|         1 | HR_TD_Mean         | 1.16261e-05 |   0.556778 |
|         2 | HR_TD_std          | 1.7395e-17  |   1.03103  |
|         2 | HR_TD_Max          | 2.14592e-14 |   0.905641 |
|         2 | HR_TD_Slope_min    | 2.75167e-13 |   0.752926 |
|         2 | HR_TD_Slope_max    | 5.63373e-14 |   0.661797 |
|         2 | EDA_TD_P_Slope_max | 7.8939e-07  |   0.489934 |

## Model: GMM (K=3)

### Contingency Tables & Metrics
**Target: Phase**  
- ARI: `0.0074` | NMI: `0.0210` | p-value: `1.2881e-02` | Cramer's V: `0.1426`

|   GMM_Bio_Cluster |   phase1 |   phase2 |   phase3 |
|------------------:|---------:|---------:|---------:|
|                 0 |       43 |       33 |       25 |
|                 1 |       54 |       53 |       57 |
|                 2 |        7 |       18 |       22 |

**Target: Cohort**  
- ARI: `-0.0020` | NMI: `0.0044` | p-value: `9.6185e-01` | Cramer's V: `0.0765`

|   GMM_Bio_Cluster |   D1_1 |   D1_2 |   D1_3 |   D1_4 |   D1_5 |   D1_6 |
|------------------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                 0 |     34 |     25 |     14 |     13 |      9 |      6 |
|                 1 |     50 |     37 |     26 |     27 |     10 |     14 |
|                 2 |     12 |     10 |      8 |      8 |      5 |      4 |

**Target: Round**  
- ARI: `0.0032` | NMI: `0.0118` | p-value: `1.9182e-01` | Cramer's V: `0.1180`

|   GMM_Bio_Cluster |   round_1 |   round_2 |   round_3 |   round_4 |
|------------------:|----------:|----------:|----------:|----------:|
|                 0 |        27 |        29 |        22 |        23 |
|                 1 |        34 |        38 |        49 |        43 |
|                 2 |        17 |        11 |         7 |        12 |

### Top 5 Discriminating Original Features (Mann-Whitney U)
|   Cluster | Feature            |       p-val |   Cohens_d |
|----------:|:-------------------|------------:|-----------:|
|         0 | HR_TD_std          | 1.68041e-27 |   1.7056   |
|         0 | HR_TD_Max          | 2.92753e-25 |   1.45987  |
|         0 | HR_TD_Slope_min    | 3.58857e-24 |   1.28295  |
|         0 | HR_TD_Slope_max    | 8.30372e-19 |   0.890132 |
|         0 | HR_TD_Mean         | 2.05336e-06 |   0.587339 |
|         1 | HR_TD_std          | 7.79738e-22 |   1.22037  |
|         1 | HR_TD_Max          | 4.79026e-19 |   1.07149  |
|         1 | HR_TD_Slope_min    | 3.55525e-18 |   0.921299 |
|         1 | HR_TD_Slope_max    | 2.19466e-18 |   0.778629 |
|         1 | HR_TD_AUC          | 1.09169e-05 |   0.508845 |
|         2 | EDA_TD_P_Slope_min | 1.6771e-08  |   0.815584 |
|         2 | EDA_TD_T_Slope_max | 1.1597e-08  |   0.811657 |
|         2 | EDA_TD_T_Slope_min | 2.00899e-07 |   0.797328 |
|         2 | EDA_TD_P_Slope_max | 1.15636e-07 |   0.765534 |
|         2 | EDA_TD_P_Max       | 1.45507e-07 |   0.690422 |

