import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

a = 0.2
bet = 0.3
gam = 0.08
delv = 0.3

x0 = 0.0
x1 = 40.0
step = 0.1
xs = np.arange(x0, x1, step)

m_in = 10
m_h = 4

np.random.seed(42)
torch.manual_seed(42)

lr = 0.01
n_epochs = 2000
train_ratio = 0.6

def gen_y(a, b, c, d, xarr):
    return a * np.cos(b * xarr) + c * np.sin(d * xarr)

def sliding_windows(yarr, win):
    X = []
    Y = []
    for i in range(len(yarr) - win):
        X.append(yarr[i:i+win].copy())
        Y.append(yarr[i+win])
    return np.array(X), np.array(Y).reshape(-1, 1)

class SimpleLSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=4, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.hidden_act = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        last_a = self.hidden_act(last)
        y = self.fc(last_a)
        return y

y_all = gen_y(a, bet, gam, delv, xs)
X_all, Y_all = sliding_windows(y_all, m_in)

train_N = int(len(X_all) * train_ratio)
X_tr, Y_tr = X_all[:train_N], Y_all[:train_N]
X_te, Y_te = X_all[train_N:], Y_all[train_N:]

mu = X_tr.mean(axis=0, keepdims=True)
sigma = X_tr.std(axis=0, keepdims=True) + 1e-8
my = Y_tr.mean(axis=0)
sy = Y_tr.std(axis=0) + 1e-8

Xtr_n = (X_tr - mu) / sigma
Xte_n = (X_te - mu) / sigma
Ytr_n = (Y_tr - my) / sy
Yte_n = (Y_te - my) / sy

Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32).unsqueeze(2)
Ytr_t = torch.tensor(Ytr_n, dtype=torch.float32)
Xte_t = torch.tensor(Xte_n, dtype=torch.float32).unsqueeze(2)
Yte_t = torch.tensor(Yte_n, dtype=torch.float32)

model = SimpleLSTMNet(input_size=1, hidden_size=m_h, out_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_hist = []
for ep in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(Xtr_t)
    loss = criterion(out, Ytr_t)
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.item())

with torch.no_grad():
    pred_tr_n = model(Xtr_t).cpu().numpy()
pred_tr = pred_tr_n * sy + my
dev_tr = pred_tr.flatten() - Y_tr.flatten()

df_train = pd.DataFrame({
    'ref': Y_tr.flatten(),
    'pred': pred_tr.flatten(),
    'diff': dev_tr
})
df_train.to_csv("train_res.csv", index=False)

initial_window = X_te[0].copy()
win = initial_window.copy()
steps = len(Y_te)
forecast = []
win = X_te[0].copy()
model.eval()
with torch.no_grad():
    for i in range(len(Y_te)):
        w_norm = (win.reshape(1, -1) - mu) / sigma
        w_t = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(2)
        pr_n = model(w_t).cpu().numpy()
        pr = (pr_n * sy + my).flatten()[0]  
        forecast.append(pr)
        win = np.concatenate([win[1:], [pr]]) 


df_test = pd.DataFrame({
    'ref': Y_te.flatten(),
    'pred': np.array(forecast).flatten(),
    'diff': np.array(forecast).flatten() - Y_te.flatten()
})
df_test.to_csv("test_res.csv", index=False)

time_tr = xs[m_in : m_in + len(Y_tr)]
plt.figure(figsize=(10,4))
plt.plot(time_tr, Y_tr.flatten())
plt.plot(time_tr, pred_tr.flatten())
plt.show()

plt.figure(figsize=(8,4))
plt.plot(loss_hist)
plt.title("Loss")
plt.show()
