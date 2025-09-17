from sklearn.preprocessing import MinMaxScaler
from df import pd, df

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_normalized.describe())
