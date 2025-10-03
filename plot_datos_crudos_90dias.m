% plot_datos_crudos_90dias.m
% Este script carga los datos crudos de 90 días y grafica los perfiles
% de demanda y generación para las 3 microrredes.

clear; clc; close all;

% --- 1. Carga de datos crudos (raw) ---
fprintf('--- Cargando datos crudos de 90 días ---\n');
addpath('data');

d_mr1_raw = load('data/winter_30D.mat').winter_30D;
g_mr1_raw = load('data/pv_wint.mat').pv_wint * 22;
d_mr2_raw = load('data/winter_60D.mat').winter_60D;
g_mr2_raw = load('data/wind_inv.mat').wind_inv * 8.49;
d_mr3_raw = load('data/School_inv.mat').School_inv * 0.45;
g_mr3_raw = load('data/pv_wint.mat').pv_wint * 30 + load('data/wind_inv.mat').wind_inv * 5;

% --- 2. Preparación de Datos y Eje de Tiempo ---
% Se busca la longitud mínima para asegurar que todos los vectores tengan el mismo tamaño
min_len_raw = min([length(d_mr1_raw), length(g_mr1_raw), length(d_mr2_raw), length(g_mr2_raw), length(d_mr3_raw), length(g_mr3_raw)]);

% Se combinan los datos en matrices para facilitar el graficado
P_dem_raw_full = [d_mr1_raw(1:min_len_raw), d_mr2_raw(1:min_len_raw), d_mr3_raw(1:min_len_raw)];
P_gen_raw_full = [g_mr1_raw(1:min_len_raw), g_mr2_raw(1:min_len_raw), g_mr3_raw(1:min_len_raw)];

% Se crea el vector de tiempo en unidades de días (1 muestra por minuto)
tiempo_dias = (0:min_len_raw-1) / (60 * 24);
fprintf('Datos cargados. Total de %.1f días de datos a 1 minuto de resolución.\n', tiempo_dias(end));

% --- 3. Creación de la Gráfica ---
fprintf('--- Generando gráfico ---\n');
fig = figure('Name', 'Perfiles de Demanda y Generación Crudos - 90 Días', 'Position', [100, 100, 1200, 800]);

% Nombres para los títulos de cada subgráfico
titulos = {'Micro-red 1 (30 Viviendas)', 'Micro-red 2 (60 Viviendas)', 'Micro-red 3 (Escuela)'};
ax = gobjects(3,1); % Arreglo para guardar los ejes y sincronizarlos

for i = 1:3
    ax(i) = subplot(3, 1, i);
    hold on;
    % Se grafica la demanda y la generación
    plot(tiempo_dias, P_dem_raw_full(:, i), 'b-', 'LineWidth', 0.5, 'DisplayName', 'Demanda');
    plot(tiempo_dias, P_gen_raw_full(:, i), 'r-', 'LineWidth', 0.5, 'DisplayName', 'Generación');
    hold off;
    
    % Formato del gráfico
    title(titulos{i});
    ylabel('Potencia [W]');
    grid on;
    box on;
    legend('show', 'Location', 'northeast');
    xlim([0, tiempo_dias(end)]); % Asegurar que el eje X cubra todo el período
end

% Etiqueta final para el eje X
xlabel('Tiempo [días]');

% Sincronizar los ejes X para que el zoom funcione en todos los subgráficos a la vez
linkaxes(ax, 'x');

fprintf('Gráfico generado exitosamente.\n');