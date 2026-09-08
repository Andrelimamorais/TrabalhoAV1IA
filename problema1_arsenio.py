"""
Universidade de Fortaleza - UNIFOR
Disciplina: Inteligência Artificial Computacional
Problema 1 - Arsênio em unhas do pé (regressão linear múltipla)
Implementação 100% manual com numpy (sem sklearn, sem pandas) - padrão da aula
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== LEITURA DOS DADOS (np.loadtxt, padrão aula) ====================
# Idade, Sexo, Uso_Beber, Uso_Cozinhar, Arsenio_Agua, Arsenio_Unhas
dados = np.loadtxt("arsenio_dataset (1).csv", delimiter=",", skiprows=1)
print("Dados de entrada:\n", dados)

X = dados[:, [0, 2, 3, 4]]   # Idade, Uso_Beber, Uso_Cozinhar, Arsenio_Agua
y = dados[:, 5]              # Arsenio_Unhas
N = X.shape[0]
print("Quantidade de observações:", N)

# ==================== CLASSES (PADRÃO DA AULA) ====================
class MRegression:
    """Regressão Linear Múltipla via pseudo-inversa de Moore-Penrose"""
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
    """Regressão Linear Simples (padrão aula)"""
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

# ==================== MÉTRICAS ====================
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

# ==================== (a) MODELO COMPLETO ====================
modelo = MRegression(X, y)
modelo.fit()
nomes = ['Intercepto', 'Idade', 'Uso_Beber', 'Uso_Cozinhar', 'Arsenio_Agua']
print("\n(a) COEFICIENTES DO MODELO COMPLETO:")
for nome, b in zip(nomes, modelo.beta):
    print(f"  {nome:15s} = {b:.6f}")

# ==================== (b) PREVISÃO ====================
x_novo = np.array([[30, 5, 5, 0.135]])
print("\n(b) Previsão (idade=30, beber=5, cozinhar=5, As água=0.135):",
      modelo.predict(x_novo)[0], "ppm")

y_pred = modelo.predict(X)

# ==================== (d) R² ====================
print("\n(d) R² =", r2_score(y, y_pred))

# ==================== (e) R² AJUSTADO ====================
print("(e) R² ajustado =", r2_ajustado(y, y_pred, p=4))

# ==================== (f) MODELO ALTERNATIVO (só arsênio na água) ====================
modelo_alt = LinearRegression(X[:, 3], y)   # Arsenio_Agua é a coluna 3 de X
modelo_alt.fit()
y_pred_alt = modelo_alt.predict(X[:, 3])
print("\n(f) MODELO ALTERNATIVO (só Arsênio na água):")
modelo_alt.summary()
print("  R² =", r2_score(y, y_pred_alt))
print("  R² ajustado =", r2_ajustado(y, y_pred_alt, p=1))

# ------- Análise de resíduos: tabela y, ŷ, e (sem pandas) -------
residuos = y - y_pred
print("\nTABELA DE RESÍDUOS:")
print(f"  {'Obs i':>6s} {'y observado':>12s} {'ŷ ajustado':>12s} {'e resíduo':>12s}")
print("  " + "-" * 44)
for i in range(N):
    print(f"  {i+1:6d} {y[i]:12.6f} {y_pred[i]:12.6f} {residuos[i]:12.6f}")

# salva a tabela em CSV sem pandas
tabela = np.column_stack((np.arange(1, N+1), y, y_pred, residuos))
np.savetxt("tabela_residuos_problema1.csv", tabela, delimiter=",",
           header="Observacao,y_observado,y_ajustado,residuo", comments="", fmt="%.6f")

# ==================== (g) INTERCEPTO ZERO ====================
modelo_zero = MRegression(X, y, intercepto=False)
modelo_zero.fit()
y_pred_zero = modelo_zero.predict(X)
print("\n(g) MODELO COM INTERCEPTO ZERO:")
print("  Coeficientes:", modelo_zero.beta)
print("  R² =", r2_score(y, y_pred_zero), " RMSE =", rmse(y, y_pred_zero))
print("  Modelo c/ intercepto: R² =", r2_score(y, y_pred), " RMSE =", rmse(y, y_pred))

# ==================== (h) MÉTRICAS DE ERRO ====================
print("\n(h) COMPARAÇÃO DE MÉTRICAS:")
print(f"  {'Métrica':10s} {'Completo':>12s} {'Alt (água)':>12s}")
for nome_m, f in [('MSE', mse), ('RMSE', rmse), ('MAE', mae)]:
    print(f"  {nome_m:10s} {f(y, y_pred):12.6f} {f(y, y_pred_alt):12.6f}")

# ==================== GRÁFICOS DE RESÍDUOS ====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(y_pred, residuos, color='steelblue', edgecolor='k')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Valores Ajustados'); axes[0].set_ylabel('Resíduos e')
axes[0].set_title('Resíduos vs Ajustados')
axes[1].scatter(np.arange(1, N+1), residuos, color='darkorange', edgecolor='k')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Observação i'); axes[1].set_ylabel('Resíduos e')
axes[1].set_title('Resíduos vs Ordem')
stats.probplot(residuos, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot dos Resíduos')
plt.tight_layout()
plt.savefig('residuos_problema1.png', dpi=150, bbox_inches='tight')
plt.show()
