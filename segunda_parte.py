import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

R = 100.0
L = 0.1
Cap = 1e-6

A = np.array([[-R/L, -1/L], [1/Cap, 0]])
B = np.array([[1/L], [0]])
C = np.array([[0, 1]])
D = 0.0

filename = 'entrada_arbitraria.csv'
data = pd.read_csv(filename, header=None) 


t_csv = data.iloc[:, 0].values  # Columna 1 (tiempo)
u_csv = data.iloc[:, 1].values  # Columna 2 (señal)

u_func = interp1d(t_csv, u_csv, 
                  kind='linear', 
                  bounds_error=False, 
                  fill_value=(u_csv[0], u_csv[-1]))


def modelRLC_arbitrary(t, x, A_mat, B_mat, u_function):
    v = u_function(t) 
    
    dx = A_mat @ x + B_mat.flatten() * v
    return dx

ts = 0.05       
tspan = (0, ts)  
x0 = [0, 0]     


sol = solve_ivp(
    modelRLC_arbitrary, 
    tspan, 
    x0, 
    args=(A, B, u_func),
    dense_output=True, 
    max_step=1e-4     
)

t_sol = sol.t  # Tiempos calculados por el solver
X_sol = sol.y  # Estados [x1, x2] en esos tiempos


u_sol = u_func(t_sol)

y_sol = (C @ X_sol + D * u_sol).flatten()



plt.figure(figsize=(10, 6))
plt.plot(t_sol, y_sol, 'b', linewidth=2, label='Salida (Vc) - Python')
plt.plot(t_sol, u_sol, 'r--', linewidth=1.5, label='Entrada (u) - Leída de CSV')
plt.title('Simulación RLC con Entrada Arbitraria (CSV)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()
