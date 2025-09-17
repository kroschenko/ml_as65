from df import df

corr_matrix = df.corr()


targetCorr = corr_matrix["MEDV"].drop("MEDV")

mostCorrelated = targetCorr.abs().idxmax()
print(f"Most corr with MEDV: {mostCorrelated}")
