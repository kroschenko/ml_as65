import seaborn as sns
import matplotlib.pyplot as plt

from df import df, mostCorrelated


sns.scatterplot(x=df[mostCorrelated], y=df["MEDV"])
plt.xlabel(mostCorrelated)
plt.ylabel("MEDV")
plt.show()
