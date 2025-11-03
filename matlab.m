clear;clc;
%% Declaracion de las varaibles y matrices a usar
R = 100;
L = 0.1;
Cap = 1e-6;

A = [-R/L -1/L ;1/Cap 0]; %Matrzi de estado
B = [1/L; 0]; % Matriz de Entrada
C = [0 1]; % Matriz de Salida
C2 = [1 0];
D = 0;
%% Primera parte de la practica
[num, dem] = ss2tf(A,B,C,D);

sys = ss(A,B,C,D);
figure();
subplot(2,1,1);
step(sys);
xlabel('tiempo [s]'); 
ylabel('Vc[V]');

subplot(2,1,2);
impulse(sys);
xlabel('tiempo [s]'); 
ylabel('Vc[V]');

%% Segunda parte
ts = 0.015; % Tiempo de simulación
tspan = [0 ts];
u1 = 1; % Voltaje de entrada
x0 = [0; 0]; % Condiciones iniciales
[t, X] = ode45(@(t,x) modelRLC(t, x, A, B, u1), tspan, x0);
y = C * X.' + D * u1;

figure;
plot(t,y,'b', 'LineWidth',2)
ylabel('Voltaje en el capacitor [V]')
xlabel('Tiempo [s]')
grid on;

function dx = modelRLC(t, x, A, B, v)
    dx = A * x + B * v; % Ecuación de Estado
end


%% tecera parte
u = @(t) (t >= 0.01) + (t>=0.02) - (t>=0.03) + (t>=0.04);

x0 = [0; 0];

ts = 0.05;               
tspan = [0 ts]; %tiempo simulacion         

[t, x] = ode45(@(t,x) A*x + B*u(t), tspan, x0);

y = (C * x.').' + D * u(t);

% Gráfica
figure;
plot(t, y, 'b', 'LineWidth', 1.5); hold on;
plot(t, u(t), 'r--', 'LineWidth', 1);
legend('Salida (V_C)', 'Entrada (u)');
xlabel('Tiempo (s)');
ylabel('Amplitud [V]');
grid on;
