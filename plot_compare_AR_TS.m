%% --- File: plot_compare_AR_TS.m ---
% SCRIPT DE VISUALIZACIÓN COMPARATIVA (AR vs TS)
%
% Objetivo: Generar gráficos superpuestos para comparar el desempeño
% de los modelos AR (Auto-Regresivos) vs TS (Takagi-Sugeno).
%
% Variables graficadas:
% 1. SoC Baterías
% 2. Volumen Estanques
% 3. Volumen Acuífero
% 4. Caudal de Bombeo
% 5. Potencia de Bombeo
% 6. Potencia de Red (P_grid)
% 7. Agua Comprada (Q_DNO)
% 8. Intercambio Hídrico (Q_t)
% 9. Descenso de Pozo (Drawdown)
% 10. Costos Acumulados
%--------------------------------------------------------------------------
clear; clc; close all;

%% 1. CONFIGURACIÓN Y CARGA DE DATOS
fprintf('--- GENERANDO GRÁFICOS COMPARATIVOS (AR vs TS) ---\n');

% Definición de archivos
file_AR = 'results_mpc/resultados_mpc_AR_3mg_7dias.mat';
file_TS = 'results_mpc/resultados_mpc_TS_3mg_7dias.mat';

% Verificación de existencia
if ~isfile(file_AR) || ~isfile(file_TS)
    error('Faltan archivos de resultados. Asegúrate de tener: \n %s \n %s', file_AR, file_TS);
end

% Carga de datos
data_AR = load(file_AR);
data_TS = load(file_TS);

% Parámetros generales (asumimos que son iguales en ambas simulaciones)
mg = data_AR.mg;
n_mg = length(mg);
Ts_sim = mg(1).Ts_sim;
leyendas_mg = {mg.nombre};

% Vector de tiempo (Horas)
N_samples = size(data_AR.SoC, 1);
t = (0:N_samples-1)' * Ts_sim / 3600; 

% Carpeta de salida
out_dir = 'results_comparison';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% 2. ESTILOS DE GRÁFICO
fontSizeLabels = 10;
lineWidth = 1.2;

% Colores para Modelos
color_AR = [0, 0.4470, 0.7410];       % Azul (AR)
color_TS = [0.8500, 0.3250, 0.0980];  % Naranja (TS)
estilo_AR = '--'; 
estilo_TS = '-';

% Límites de Gráficos (Para que se vea bien la leyenda)
y_margin = 0.1;

%% 3. GENERACIÓN DE FIGURAS

% --- FIG 1: SoC Baterías ---
create_subplot_figure(1, 'SoC Baterías', 'Estado de Carga [%]', ...
    data_AR.SoC*100, data_TS.SoC*100, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [0 100], out_dir, 'SoC_Comparison');

% --- FIG 2: Volumen Estanques ---
% Convertimos a m3
create_subplot_figure(2, 'Volumen Estanques', 'Volumen [m^3]', ...
    data_AR.V_tank/1000, data_TS.V_tank/1000, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Tank_Volume_Comparison');

% --- FIG 3: Volumen Acuífero (Global) ---
fig3 = figure('Name', 'Volumen Acuífero', 'Position', [100, 100, 600, 400]);
plot(t, data_AR.V_aq/1000, estilo_AR, 'Color', color_AR, 'LineWidth', lineWidth);
hold on;
plot(t, data_TS.V_aq/1000, estilo_TS, 'Color', color_TS, 'LineWidth', lineWidth);
title('Volumen Acuífero Compartido');
ylabel('Volumen [m^3]'); xlabel('Tiempo [horas]');
legend('Modelo AR', 'Modelo TS', 'Location', 'best');
grid on; set(gca, 'FontSize', fontSizeLabels);
export_fig(fig3, out_dir, 'Aquifer_Volume_Comparison');

% --- FIG 4: Caudal de Bombeo (Qp) ---
create_subplot_figure(4, 'Caudal de Bombeo', 'Caudal [L/s]', ...
    data_AR.Q_p, data_TS.Q_p, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Pumping_Flow_Comparison');

% --- FIG 5: Potencia de Bombeo ---
create_subplot_figure(5, 'Potencia Bombeo', 'Potencia [kW]', ...
    data_AR.P_pump, data_TS.P_pump, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Pumping_Power_Comparison');

% --- FIG 6: Intercambio con Red (P_grid) ---
create_subplot_figure(6, 'Potencia de Red', 'Potencia [kW]', ...
    data_AR.P_grid, data_TS.P_grid, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Grid_Power_Comparison');

% --- FIG 7: Agua Comprada (Q_DNO) ---
create_subplot_figure(7, 'Agua Comprada', 'Caudal [L/s]', ...
    data_AR.Q_DNO, data_TS.Q_DNO, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Water_Bought_Comparison');

% --- FIG 8: Intercambio Hídrico (Qt) ---
create_subplot_figure(8, 'Intercambio Hídrico (Qt)', 'Caudal [L/s]', ...
    data_AR.Q_t, data_TS.Q_t, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Water_Exchange_Comparison');

% --- FIG 9: Descenso de Pozo (Drawdown) ---
% Calculamos el descenso relativo s = h_p(t) - h_p0
s_AR = data_AR.h_p - [mg.h_p0]; 
s_TS = data_TS.h_p - [mg.h_p0];
create_subplot_figure(9, 'Descenso Pozo', 'Descenso [m]', ...
    s_AR, s_TS, t, n_mg, leyendas_mg, ...
    color_AR, color_TS, estilo_AR, estilo_TS, lineWidth, fontSizeLabels, ...
    [], out_dir, 'Well_Drawdown_Comparison');

% --- FIG 10: Costos Acumulados ---
fig10 = figure('Name', 'Costo Acumulado', 'Position', [100, 100, 600, 400]);
C_p = 110; C_q = 644;
% Costo AR
cost_AR_inst = (sum(data_AR.P_grid, 2)*(Ts_sim/3600))*C_p + (sum(data_AR.Q_DNO, 2)*Ts_sim/1000)*C_q;
cum_cost_AR = cumsum(cost_AR_inst);
% Costo TS
cost_TS_inst = (sum(data_TS.P_grid, 2)*(Ts_sim/3600))*C_p + (sum(data_TS.Q_DNO, 2)*Ts_sim/1000)*C_q;
cum_cost_TS = cumsum(cost_TS_inst);

plot(t, cum_cost_AR, estilo_AR, 'Color', color_AR, 'LineWidth', lineWidth+0.5);
hold on;
plot(t, cum_cost_TS, estilo_TS, 'Color', color_TS, 'LineWidth', lineWidth+0.5);
title('Costo de Operación Acumulado Total');
ylabel('Costo [CLP]'); xlabel('Tiempo [horas]');
legend('Modelo AR', 'Modelo TS', 'Location', 'northwest');
grid on; set(gca, 'FontSize', fontSizeLabels);
export_fig(fig10, out_dir, 'Total_Cost_Comparison');

fprintf('--- PROCESO COMPLETADO. Gráficos guardados en "%s" ---\n', out_dir);


%% --- FUNCIONES AUXILIARES ---

function create_subplot_figure(fig_num, title_str, ylab, ...
    Y_AR, Y_TS, t, n_mg, legends, c_AR, c_TS, s_AR, s_TS, lw, fsz, ylims, dir, fname)

    fig = figure('Name', title_str, 'Position', [100, 100, 800, 700]);
    sgtitle([title_str ' (Comparación AR vs TS)']);
    
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        % Plot AR
        plot(t, Y_AR(:, i), s_AR, 'Color', c_AR, 'LineWidth', lw, 'DisplayName', 'AR');
        hold on;
        % Plot TS
        plot(t, Y_TS(:, i), s_TS, 'Color', c_TS, 'LineWidth', lw, 'DisplayName', 'TS');
        
        title(legends{i});
        ylabel(ylab);
        grid on; set(gca, 'FontSize', fsz);
        xlim([t(1) t(end)]);
        if ~isempty(ylims), ylim(ylims); end
        
        % Leyenda solo en el primer subplot para no saturar
        if i == 1
            legend('Location', 'best');
        end
    end
    xlabel('Tiempo [horas]');
    
    export_fig(fig, dir, fname);
end

function export_fig(fig_handle, dir, fname)
    full_path = fullfile(dir, fname);
    % Guardar como PNG y FIG
    saveas(fig_handle, [full_path '.png']);
    savefig(fig_handle, [full_path '.fig']);
    % Opcional: PDF Vectorial
    % exportgraphics(fig_handle, [full_path '.pdf'], 'ContentType', 'vector');
end