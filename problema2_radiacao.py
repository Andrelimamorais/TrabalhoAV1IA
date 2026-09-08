import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt("dose_radiacao_expandido.csv", delimiter=",", skiprows=1,
                   usecols=(1, 2, 3))
print("Dados de entrada:\n", dados[:5])

X = dados[:, [1, 2]] 
y = dados[:, 0]        
N = X.shape[0]
print("Quantidade de observações:", N)

class MRegression:
    def __init__(self, X, y, intercepto=True):
        self.X = X
        self.y = y
        self.beta = None
        self.N = X.shape[0]
        self.intercepto = intercepto
    def fit(self):
        if self.intercepto:
            X_ = np.column_stack((np.ones(self.N), self.X))
        else:
            X_ = self.X
        self.beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ self.y
        return self
    def predict(self, X_new):
        X_new = np.atleast_2d(X_new)
        N = X_new.shape[0]
        if self.intercepto:
            X_new = np.column_stack((np.ones(N), X_new))
        return X_new @ self.beta

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.b0 = None
        self.b1 = None
    def fit(self):
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)
        self.b1 = np.sum((self.y - ybar) * (self.x - xbar)) / np.sum((self.x - xbar) ** 2)
        self.b0 = ybar - self.b1 * xbar
        return self
    def predict(self, x_new):
        return self.b0 + self.b1 * np.array(x_new)
    def summary(self):
        print(f"Modelo: y = {self.b0} + {self.b1} * x")
        print(f"Intercepto = {self.b0}")
        print(f"Coeficiente Angular = {self.b1}")

def r2_score(y_true, y_pred):
    numerador = np.sum((y_true - y_pred) ** 2)
    denominador = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerador / denominador)

def r2_ajustado(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

modelo = MRegression(X, y)
modelo.fit()
nomes = ['Intercepto', 'mAmp', 'Tempo_de_Exposicao']
print("\n(a) COEFICIENTES DO MODELO COMPLETO:")
for nome, b in zip(nomes, modelo.beta):
    print(f"  {nome:20s} = {b:.6f}")

x_novo = np.array([[15, 5]])
print("\n(b) Previsão (15 mA, 5 min):", modelo.predict(x_novo)[0], "rad")

y_pred = modelo.predict(X)

print("\n(c) R² =", r2_score(y, y_pred))

print("(d) R² ajustado =", r2_ajustado(y, y_pred, p=2))

modelo_alt = LinearRegression(X[:, 0], y)   
modelo_alt.fit()
y_pred_alt = modelo_alt.predict(X[:, 0])
print("\n(e) MODELO ALTERNATIVO (só Corrente):")
modelo_alt.summary()
print("  R² =", r2_score(y, y_pred_alt))
print("  R² ajustado =", r2_ajustado(y, y_pred_alt, p=1))

modelo_zero = MRegression(X, y, intercepto=False)
modelo_zero.fit()
y_pred_zero = modelo_zero.predict(X)
print("\n(f) MODELO COM INTERCEPTO ZERO:")
print("  Coeficientes:", modelo_zero.beta)
print("  R² =", r2_score(y, y_pred_zero), " RMSE =", rmse(y, y_pred_zero))
print("  Modelo c/ intercepto: R² =", r2_score(y, y_pred), " RMSE =", rmse(y, y_pred))

print("\n(h) COMPARAÇÃO DE MÉTRICAS:")
print(f"  {'Métrica':10s} {'Completo':>12s} {'Alt (corrente)':>15s}")
for nome_m, f in [('MSE', mse), ('RMSE', rmse), ('MAE', mae)]:
    print(f"  {nome_m:10s} {f(y, y_pred):12.4f} {f(y, y_pred_alt):15.4f}")

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ma, te = X[:, 0], X[:, 1]
ax1.scatter(ma, te, y, c='red', s=8, alpha=0.4, label='Dados originais')
ax1.scatter(ma, te, y_pred, c='green', s=8, alpha=0.4, label='Previstos')
ma_grid, te_grid = np.meshgrid(
    np.linspace(min(ma), max(ma), 10),
    np.linspace(min(te), max(te), 10))
y_grid = modelo.beta[0] + modelo.beta[1]*ma_grid + modelo.beta[2]*te_grid
ax1.plot_surface(ma_grid, te_grid, y_grid, alpha=0.4, color='cyan')
ax1.grid(False)  
ax1.set_xlabel('Corrente (mA)'); ax1.set_ylabel('Tempo (min)'); ax1.set_zlabel('Dose (rad)')
ax1.set_title('Hiperplano de Regressão')

residuos = y - y_pred
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(y_pred, residuos, color='steelblue', s=8, alpha=0.5)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('Valores Ajustados'); ax2.set_ylabel('Resíduos e')
ax2.set_title('Resíduos vs Ajustados')

ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(y, y_pred, color='darkgreen', s=8, alpha=0.5)
lim = [0, max(y.max(), y_pred.max())*1.05]
ax3.plot(lim, lim, 'r--')
ax3.set_xlabel('Dose Observada'); ax3.set_ylabel('Dose Prevista')
ax3.set_title('Observado vs Previsto')

plt.tight_layout()
plt.savefig('graficos_problema2.png', dpi=150, bbox_inches='tight')
plt.show()