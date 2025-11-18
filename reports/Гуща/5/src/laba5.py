
import numpy as np
import matplotlib.pyplot as plt

def y(x):
  return 0.3*np.cos(0.3*x)+0.07*np.sin(0.3*x)

x=np.linspace(0,20,100)

y=y(x)

plt.plot(x, y, label='y', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Пример графика')
plt.legend()
plt.grid(True)
plt.show()

#print(x)
#print(y)
input=10
hidden=4
train_size=80
def make_dataset(y, window):
    X, Y = [], []
    for i in range(len(y) - window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    return np.array(X), np.array(Y)

X, Y = make_dataset(y, input)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test,  Y_test  = X[train_size:], Y[train_size:]
Y_train = Y_train.reshape(-1,1)
Y_test  = Y_test.reshape(-1,1)
#print(Y_train)



W1 = np.random.uniform(-0.5, 0.5, (input, hidden))
b1 = np.zeros((1, hidden))

W2 = np.random.uniform(-0.5, 0.5, (hidden, 1))
b2 = np.zeros((1, 1))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2

def train(X, Y):
    lr=0.01
    global W1, b1, W2, b2
    z1, a1, y_pred = forward(X)

    delta_output = y_pred - Y  # (batch_size, 1)

    W2 -= lr * a1.T @ delta_output      # (hidden, batch) @ (batch, 1) → (hidden,1)
    b2 -= lr * np.sum(delta_output, axis=0, keepdims=True)

    delta_hidden = (delta_output @ W2.T) * (a1 * (1 - a1))  # (batch,1) @ (1,hidden) → (batch,hidden)
    W1 -= lr * X.T @ delta_hidden                            # (input,batch) @ (batch,hidden) → (input,hidden)
    b1 -= lr * np.sum(delta_hidden, axis=0, keepdims=True)

    loss = np.mean((y_pred - Y)**2)
    return loss

epochs = 2000
loss_history = []

for epoch in range(epochs):
    loss = train(X_train, Y_train)
    loss_history.append(loss)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")

_, _, y_test_pred = forward(X_test)

print("    Эталон | Предсказано | Отклонение")
for y,y_pred in zip(Y_test,y_test_pred):
  print(f"{y[0]:.4f}", end="     |")
  print(f"{y_pred[0]:.4f}", end="       |")
  print(f"{(y[0]-y_pred[0]):.4f}")

plt.plot(loss_history)
plt.title("Изменение ошибки MSE")
plt.xlabel("Итерация")
plt.ylabel("Ошибка")
plt.grid()
plt.show()