import seaborn as sns
import matplotlib.pyplot as plt

from df import df


corr_matrix = df.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()
