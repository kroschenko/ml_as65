import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a_list = np.linspace(0.05, 0.2, 8)
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
lr = 0.01
n_epochs = 2000
report_every = 200
train_ratio = 0.6

def gen_y(a, b, c, d, xarr):
    return a * np.cos(b * xarr) + c * np.sin(d * xarr)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_der(s):
    return s * (1.0 - s)

def sliding_windows(yarr, win):
    X = []
    Y = []
    for i in range(len(yarr) - win):
        X.append(yarr[i:i+win].copy())
        Y.append(yarr[i+win])
    return np.array(X), np.array(Y).reshape(-1, 1)

class SimpleMLP:
    def __init__(self, nin, nhid, nout=1):
        self.W1 = np.random.randn(nin, nhid) * 0.5
        self.b1 = np.zeros((1, nhid))
        self.W2 = np.random.randn(nhid, nout) * 0.5
        self.b2 = np.zeros((1, nout))

    def forward(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = a1.dot(self.W2) + self.b2
        return z1, a1, z2

    def predict(self, X):
        return self.forward(X)[2]

    def train(self, X, Y, epochs=1000, lr=0.01):
        n = X.shape[0]
        hist = []
        for ep in range(epochs):
            z1, a1, z2 = self.forward(X)
            err = z2 - Y
            loss = np.mean(err**2)
            hist.append(loss)
            dW2 = a1.T.dot(err) * (2.0/n)
            db2 = np.sum(err, axis=0, keepdims=True) * (2.0/n)
            da1 = err.dot(self.W2.T)
            dz1 = da1 * sigmoid_der(a1)
            dW1 = X.T.dot(dz1) * (1.0/n)
            db1 = np.sum(dz1, axis=0, keepdims=True) * (1.0/n)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        return hist

results_per_a = []
models_store = {}

for a_try in a_list:
    y_all = gen_y(a_try, bet, gam, delv, xs)
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

    ml = SimpleMLP(m_in, m_h, 1)
    h = ml.train(Xtr_n, Ytr_n, epochs=n_epochs, lr=lr)

    pred_te_n = ml.predict(Xte_n)
    pred_te = pred_te_n * sy + my
    mse_test = float(np.mean((pred_te - Y_te)**2))
    results_per_a.append((a_try, mse_test))
    models_store[a_try] = {
        'model': ml,
        'mu': mu, 'sigma': sigma, 'my': my, 'sy': sy,
        'X_tr': X_tr, 'Y_tr': Y_tr,
        'X_te': X_te, 'Y_te': Y_te,
        'history': h
    }
    print(a_try, mse_test)

best_a, best_mse = min(results_per_a, key=lambda p: p[1])
best = models_store[best_a]

model = best['model']
mu = best['mu']
sigma = best['sigma']
my = best['my']
sy = best['sy']
X_tr = best['X_tr']
Y_tr = best['Y_tr']
X_te = best['X_te']
Y_te = best['Y_te']
history = best['history']

Xtr_n = (X_tr - mu) / sigma
pred_tr_n = model.predict(Xtr_n)
pred_tr = pred_tr_n * sy + my
dev_tr = pred_tr.flatten() - Y_tr.flatten()

df_train_res = pd.DataFrame({
    'ref': Y_tr.flatten(),
    'pred': pred_tr.flatten(),
    'diff': dev_tr
})

time_tr = xs[m_in : m_in + X_tr.shape[0]]
plt.figure(figsize=(10,4))
plt.plot(time_tr, Y_tr.flatten())
plt.plot(time_tr, pred_tr.flatten())
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history)
plt.tight_layout()
plt.show()

initial_window = X_te[0].copy()
win = initial_window.copy()
steps = len(Y_te)
recursive_preds = []

for i in range(steps):
    w_norm = (win.reshape(1, -1) - mu) / sigma
    pr_n = model.predict(w_norm)
    pr = (pr_n * sy + my).flatten()[0]
    recursive_preds.append(pr)
    win = np.concatenate([win[1:], [pr]])

df_test_res = pd.DataFrame({
    'ref': Y_te.flatten(),
    'pred': np.array(recursive_preds),
    'diff': np.array(recursive_preds) - Y_te.flatten()
})


print("best a:", best_a, "mse:", best_mse)
print("train mae:", np.mean(np.abs(df_train_res['diff'])))
print("test mae:", np.mean(np.abs(df_test_res['diff'])))

df_train_res.to_csv("train_results.csv", index=False)
df_test_res.to_csv("test_results.csv", index=False)
