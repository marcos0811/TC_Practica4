import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.integrate import solve_ivp

R = 100.0
L = 0.1
Cap = 1e-6

A = np.array([
    [-R/L, -1/L],
    [1/Cap, 0]
])

B = np.array([
    [1/L],
    [0]
])

C = np.array([[0, 1]])

C2 = np.array([[1, 0]])

D = 0.0

sys = ct.ss(A, B, C, D)


plt.figure(1)

# respuesta al escalon
plt.subplot(2, 1, 1)
T_step, yout_step = ct.step_response(sys)
plt.plot(T_step, yout_step)
plt.title('Respuesta al Escalón (Voltaje del Capacitor)')
plt.xlabel('tiempo [s]')
plt.ylabel('Vc[V]')
plt.grid(True)

# respuesta al impulso
plt.subplot(2, 1, 2)
T_impulse, yout_impulse = ct.impulse_response(sys)
plt.plot(T_impulse, yout_impulse)
plt.title('Respuesta al Impulso (Voltaje del Capacitor)')
plt.xlabel('tiempo [s]')
plt.ylabel('Vc[V]')
plt.grid(True)

plt.tight_layout()


def modelRLC(t, x, A_mat, B_mat, v):
    
    dx = A_mat @ x + B_mat.flatten() * v
    return dx

ts = 0.015       # tiempo de simulación
tspan = (0, ts)  # Rango de tiempo para solve_ivp
u1 = 1           # Voltaje de entrada
x0 = [0, 0]      # Condiciones iniciales

sol = solve_ivp(modelRLC, tspan, x0, args=(A, B, u1), dense_output=True)

t = sol.t
X = sol.y  


y = C @ X + D * u1
y = y.flatten() 

# Graficar la simulación de ode45
plt.figure(2)
plt.plot(t, y, 'b', linewidth=2)
plt.title('Simulación ODE (Voltaje en el capacitor)')
plt.ylabel('Voltaje en el capacitor [V]')
plt.xlabel('Tiempo [s]')
plt.grid(True)

# Mostrar todas las gráficas
plt.show()
