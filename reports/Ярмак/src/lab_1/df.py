import pandas as pd

fileName = "BostonHousing.csv"

df = pd.read_csv(fileName)

corr_matrix = df.corr()
targetCorr = corr_matrix['MEDV'].drop('MEDV')
mostCorrelated = targetCorr.abs().idxmax()