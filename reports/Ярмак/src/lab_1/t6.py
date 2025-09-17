import matplotlib.pyplot as plt
import seaborn as sns

from t5 import df_normalized

sns.histplot(df_normalized["CRIM"], bins=30, kde=True)
plt.xlabel("CRIM (norm)")
plt.ylabel("Frequency")
plt.show()
