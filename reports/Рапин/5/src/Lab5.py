import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функция формирования оконных данных
def prepare_windows(y, window=8):
    X = []  # входы
    Y = []  # выходы
    for i in range(window, len(y)):
        X.append(y[i - window:i])
        Y.append(y[i])
    return np.array(X), np.array(Y)


# Генерация данных по заданной функции
def generate_signal(n_points=600, a=0.3, b=0.5, c=0.05, d=0.5, x_start=0.0, x_stop=40.0):
    x = np.linspace(x_start, x_stop, n_points)
    y = a * np.cos(b * x) + c * np.sin(d * x)
    return x, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1.0 - a)

class MLP:
    def __init__(self, n_in=8, n_hidden=3, n_out=1, rng=None):
        self.rng = np.random.default_rng(rng)
        self.W1 = self.rng.normal(0.0, 0.1, size=(n_in, n_hidden))
        self.b1 = np.zeros((n_hidden,))
        self.W2 = self.rng.normal(0.0, 0.1, size=(n_hidden, n_out))
        self.b2 = np.zeros((n_out,))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        yhat = z2
        return a1, yhat

    def loss_mse(self, yhat, y):
        return np.mean((yhat.flatten() - y.flatten()) ** 2)

    def backward(self, X, y, a1, yhat):
        N = X.shape[0]
        diff = (yhat - y.reshape(-1, 1))
        dL_dyhat = (2.0 / N) * diff

        dL_dW2 = a1.T @ dL_dyhat
        dL_db2 = np.sum(dL_dyhat, axis=0)

        dL_da1 = dL_dyhat @ self.W2.T
        dL_dz1 = dL_da1 * sigmoid_deriv(a1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def step(self, grads, alpha):
        dW1, db1, dW2, db2 = grads
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def fit(self, X, y, alpha=0.01, epochs=1000, verbose=False):
        history = []
        for epoch in range(epochs):
            a1, yhat = self.forward(X)
            loss = self.loss_mse(yhat, y)
            history.append(loss)
            grads = self.backward(X, y, a1, yhat)
            self.step(grads, alpha)
            if verbose and (epoch % max(1, (epochs // 10)) == 0):
                print(f"Эпоха {epoch:4d} | MSE = {loss:.6f}")
        return history

    def predict(self, X):
        _, yhat = self.forward(X)
        return yhat.flatten()

def split_windows(X, Y, train_ratio=0.6, val_ratio=0.2):
    N = X.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def grid_search_alpha(X_train, Y_train, X_val, Y_val, alphas, epochs=800, seed=42):
    results = []
    for alpha in alphas:
        mlp = MLP(rng=seed)
        mlp.fit(X_train, Y_train, alpha=alpha, epochs=epochs, verbose=False)
        y_val_pred = mlp.predict(X_val)
        val_mse = np.mean((y_val_pred - Y_val) ** 2)
        results.append((alpha, val_mse, mlp))
    results.sort(key=lambda t: t[1])
    best_alpha, best_mse, best_model = results[0]
    return best_alpha, best_mse, best_model, results

def main():
    a, b, c, d = 0.3, 0.5, 0.05, 0.5

    # Генерация исходных данных
    x, y = generate_signal(n_points=600, a=a, b=b, c=c, d=d, x_start=0.0, x_stop=40.0)

    window = 8
    X, Y = prepare_windows(y, window=window)

    # Разделение на обучающий тестовый набор
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_windows(X, Y, train_ratio=0.6, val_ratio=0.2)

    # Подбор лучшего темпа обучения alpha
    alphas = [0.001, 0.005, 0.01, 0.05]
    best_alpha, best_val_mse, best_model, alpha_results = grid_search_alpha(X_train, Y_train, X_val, Y_val, alphas, epochs=1200, seed=123)

    print(f"Лучший alpha: {best_alpha:.4f}, MSE: {best_val_mse:.6f}")

    X_train_full = np.vstack([X_train, X_val])
    Y_train_full = np.concatenate([Y_train, Y_val])
    final_mlp = MLP(rng=123)
    loss_history = final_mlp.fit(X_train_full, Y_train_full, alpha=best_alpha, epochs=1500, verbose=False)

    # График прогнозируемой функции на участке обучения
    y_train_pred = final_mlp.predict(X_train_full)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(Y_train_full)), Y_train_full, label="Эталон", linewidth=2)
    plt.plot(np.arange(len(Y_train_full)), y_train_pred, label="Прогноз", linewidth=2)
    plt.title("Прогнозируемая функция на участке обучения")
    plt.xlabel("Индекс окна")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Результаты обучения
    train_table = pd.DataFrame({
        "Эталонное значение": Y_train_full,
        "Полученное значение": y_train_pred,
    })
    train_table["Отклонение"] = train_table["Полученное значение"] - train_table["Эталонное значение"]
    print("\nТаблица результатов на обучающем участке:")
    print(train_table.head(10))

    # График MSE по эпохам обучения
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="MSE")
    plt.title("Изменение ошибки (MSE) в зависимости от итерации")
    plt.xlabel("Итерация (эпоха)")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Результаты прогнозирования таблица
    y_test_pred = final_mlp.predict(X_test)
    test_table = pd.DataFrame({
        "Эталонное значение": Y_test,
        "Полученное значение": y_test_pred,
    })
    test_table["Отклонение"] = test_table["Полученное значение"] - test_table["Эталонное значение"]
    print("\nТаблица результатов прогнозирования:")
    print(test_table.head(10))


    train_mse = np.mean((y_train_pred - Y_train_full) ** 2)
    test_mse = np.mean((y_test_pred - Y_test) ** 2)
    print(f"\nИтоговая MSE на обучении: {train_mse:.6f}")
    print(f"Итоговая MSE на тесте: {test_mse:.6f}")

if __name__ == "__main__":
    main()
  
