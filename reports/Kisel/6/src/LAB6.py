import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# класс JordanRNN 
class JordanRNN:
    def __init__(self, in_dim=1, hidden=3, out_dim=1, lr=0.01, window=8, seed=42):
        self.in_dim = in_dim #В каждый момент времени в RNN подаётся одно число - элемент окна. В MLP мы подавали весь вектор - числе
        self.hidden = hidden
        self.out_dim = out_dim
        self.lr = lr
        self.window = window
        rng = np.random.RandomState(seed)
        self.Wx = rng.normal(scale=0.5, size=(in_dim, hidden)) #веса от входа x_t -> скрытый слой [ w11   w12   w13 ]
        self.Wy = rng.normal(scale=0.5, size=(out_dim, hidden)) #веса от предыдущего выхода сети y_{t−1} -> скрытый слой [ u11   u12   u13 ]
        self.Wh = rng.normal(scale=0.5, size=(hidden, out_dim)) #веса из скрытого слоя в выход 
        #Векторы смещений
        self.bh = np.zeros((1, hidden))
        self.by = np.zeros((1, out_dim))

    def forward(self, seq):
        h_list, y_list, z_list = [], [], []
        prev_y = np.zeros((1,1))
        for t in range(self.window):
            x_t = seq[t].reshape(1,1) #Например 0.123 → [[0.123]] делает число совместимым для матричного умножения
            z_t = x_t.dot(self.Wx) + prev_y.dot(self.Wy) + self.bh    #z₀ = x₀ * Wx + prev_y * Wy + bh    входной слой 
            h_t = sigmoid(z_t)   #h₀ = sigmoid(z₀)     скрытый слой
            y_t = h_t.dot(self.Wh) + self.by   # y₀ = h₀ ⋅ Wh + by      Выход  Мы получили новое y₀ ≈ 0.695(например) 
            z_list.append(z_t); h_list.append(h_t); y_list.append(y_t)
            prev_y = y_t  # теперь наше y₀ ≈ 0.695 теперь новое prev_y и так 8 раз(так как окно 8) !!!!!предыдущий выход → влияет на текущий вход
        return z_list, h_list, y_list

    def backward(self, seq, target, z_list, h_list, y_list):#нужно пройти назад во времени и аккумулировать вклад ошибки в каждый параметр. 
        #Ошибка на последнем шаге «расплывается» назад по всем 8 шагам через элемент Wy
        y_pred = y_list[-1]
        err = (y_pred - target.reshape(1,1)) 
        loss = 0.5 * (err**2).mean() #Ошибка есть только на последнем шаге. Но веса использовались на каждом шаге t=0..7поэтому нужно пройти назад по времени.
        dWx = np.zeros_like(self.Wx); dWy = np.zeros_like(self.Wy); dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh); dby = np.zeros_like(self.by)


        dy = err  # это «первичная» ошибка на выходе
        dWh += h_list[-1].T.dot(dy) #"если h_7 изменится на немного, как изменится y_7?"
        dby += dy
        dh = dy.dot(self.Wh.T) #“если h_7 изменится, как изменится ошибка?”

        for t in reversed(range(self.window)): #Backprop Through Time  
            h_t = h_list[t]; z_t = z_list[t] #Берём скрытое состояние h_t Берём вход в скрытый слой z_t
            dz = dh * dsigmoid(z_t) #Так же, как в MLP, но для каждого шага времени.
            dbh += dz #вклад в bias скрытого слоя
            x_t = seq[t].reshape(1,1)
            dWx += x_t.T.dot(dz) #вклад в веса от входа
            prev_y = y_list[t-1] if t>0 else np.zeros((1,1)) #вклад в веса от предыдущего выхода если связь prev_y сильнее влияет на z,её вес Wy должен быть скорректирован сильнее
            dWy += prev_y.T.dot(dz) 
            # распространение градиента на прошлый шаг через prev_y
            dh = dz.dot(self.Wy.T)

        # обновление параметров (внутри backward: можно сразу обновить или вернуть градиенты)
        self.Wx -= self.lr * dWx
        self.Wy -= self.lr * dWy
        self.Wh -= self.lr * dWh
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

        return loss

    def fit(self, X, Y, epochs=1000, verbose=True):
        loss_hist = []
        for epoch in range(1, epochs+1):
            total_loss = 0.0
            for i in range(X.shape[0]):
                seq = X[i]
                target = Y[i]
                z, h, y = self.forward(seq)
                loss = self.backward(seq, target, z, h, y)
                total_loss += loss
            avg_loss = total_loss / X.shape[0]
            loss_hist.append(avg_loss)
            if verbose and (epoch==1 or epoch%50==0):
                print(f"[Jordan RNN] Epoch {epoch}/{epochs}, loss={avg_loss:.6f}")
        return loss_hist

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            _, _, y_list = self.forward(X[i])
            preds.append(y_list[-1].ravel()[0])
        return np.array(preds).reshape(-1,1)


# Генерация данных и подготовка 


# Генерация функции y(x) = a*cos(bx) + c*sin(dx)
def generate_series(a=0.4, b=0.2, c=0.07, d=0.2, N=300, x_max=50):
    x = np.linspace(0, x_max, N)   # равномерно расположенные 300 точек от 0 до 50
    y = a * np.cos(b * x) + c * np.sin(d * x)
    return x, y

# Формирование данных типа "окно"
def prepare_windows(y, window=8):
    X = []#входы
    Y = []#выходы
    for i in range(window, len(y)):
        X.append(y[i-window:i])   # [0:8], [1:9], [2:10], [3:11]
        Y.append(y[i]) # [9], [10], [11], [12]
    return np.array(X), np.array(Y).reshape(-1, 1)  #reshape из вектора в матрицу 

# Стандартизатор
class SimpleScaler:
    def fit(self, X):# Только вычисляет параметры стандартизации
        self.mean = X.mean(axis=0)#считает среднее
        self.std = X.std(axis=0)#считает стандартное отклонение
        self.std[self.std == 0] = 1#защита от деления на ноль (если признак константный)
    def transform(self, X): # Применяет формулу на основы подготовленных параметров
        return (X - self.mean) / self.std #Формула стандартизации 
    def fit_transform(self, X): # И подготавливает и рассчитывает 
        self.fit(X)
        return self.transform(X)


# Параметры функции
a, b, c, d = 0.4, 0.2, 0.07, 0.2

# Генерируем ряд
x_vals, y_vals = generate_series(a=a, b=b, c=c, d=d, N=360, x_max=60)

# Подготавливаем окна
window = 8
X_all, Y_all = prepare_windows(y_vals, window=window)
# Разделяем train/test (70% train)
train_size = int(0.7 * len(X_all))
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

# Стандартизация признаков 
scaler = SimpleScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# ОБУЧЕНИЕ RNN 

rnn = JordanRNN(in_dim=1, hidden=3, out_dim=1, lr=0.01, window=window, seed=42)
loss_hist = rnn.fit(X_train_s, Y_train, epochs=1000, verbose=True)

# Предсказания на train и test
y_train_pred = rnn.predict(X_train_s)   
y_test_pred  = rnn.predict(X_test_s)

# MSE
mse_train = np.mean((y_train_pred - Y_train)**2)
mse_test  = np.mean((y_test_pred  - Y_test )**2)
print(f"\nRNN (Jordan) Train MSE = {mse_train:.6f}")
print(f"RNN (Jordan) Test  MSE = {mse_test:.6f}")

x_targets = x_vals[window: window + len(X_all)]
x_train_targets = x_targets[:train_size]
x_test_targets  = x_targets[train_size:]


#График прогнозируемой функции на участке обучения
plt.figure(figsize=(10,5))
plt.plot(x_vals, y_vals, label='Истинная функция', linewidth=1)
plt.scatter(x_train_targets, Y_train.flatten(), s=20, label='Train - эталон', alpha=0.6)
plt.scatter(x_train_targets, y_train_pred.flatten(), s=20, label='Train - предсказание', marker='x')
plt.scatter(x_test_targets, Y_test.flatten(), s=20, label='Test - эталон', alpha=0.6)
plt.scatter(x_test_targets, y_test_pred.flatten(), s=20, label='Test - предсказание', marker='x')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Истинная функция и предсказания (train/test)')
plt.legend(); plt.grid(True)
plt.show()

#Результаты обучения: таблица эталон/полученное/отклонение (для train)
#    и график изменения ошибки в зависимости от итерации (эпохи)
df_train = pd.DataFrame({
    'x': x_train_targets,
    'target': Y_train.flatten(),
    'pred': y_train_pred.flatten(),
})
df_train['error'] = df_train['pred'] - df_train['target']

print("\nПример первых 10 строк таблицы результатов (train):")
print(df_train.head(10))

# график ошибки по эпохам (loss_hist содержит среднюю ошибку за эпоху)
plt.figure(figsize=(8,4))
plt.plot(np.arange(1, len(loss_hist)+1), loss_hist, '-o', markersize=3)
plt.xlabel('Эпоха'); plt.ylabel('Средняя loss за эпоху'); plt.title('Изменение ошибки по эпохам (RNN)')
plt.grid(True)
plt.show()

#Результаты прогнозирования: таблица для test

df_test = pd.DataFrame({
    'x': x_test_targets,
    'target': Y_test.flatten(),
    'pred': y_test_pred.flatten()
})
df_test['error'] = df_test['pred'] - df_test['target']

print("\nПример первых 10 строк таблицы результатов (test):")
print(df_test.head(10))
