import numpy as np

def highly_corr(df, perf=0.95):
    """
    Parameters
    ----------
    df:
        Input DataFrame.
    perf:
        ratio set to 0.95

    Returns
    -------
    list[str]
        Names of features  which present a correlation higher than 95% wrt one other feature; this last one omitted
    """  
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    # (This prevents comparing a feature to itself or comparing the same pair twice)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > perf)]

    return to_drop