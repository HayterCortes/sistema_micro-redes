%% --- File: plot_quality_comparison_AR_TS_all.m ---
%
% COMPARA LA CALIDAD (R2) DE LAS EXPLICACIONES: AR vs TS (BATCH 3 MGs)
%
% FUNCIONALIDADES:
% 1. Genera gráficos comparativos para MG1, MG2 y MG3 automáticamente.
% 2. Calcula y muestra tablas estadísticas de desempeño (R2) por microrred.
% 3. CORRECCIÓN FINAL: Uso de intérprete TeX (sin warnings) y solo exporta PDF.
%
% Requiere haber ejecutado:
% - lime_temporal_main_MEAN.m (para AR y TS)
% - O lime_temporal_pumping_main_MEAN.m (si SCENARIO = 'PUMP')
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURACIÓN ---
SCENARIO_TYPE = 'PUMP';  % <--- 'STD' (Intercambio Qt) o 'PUMP' (Bombeo Qp)
TARGETS = [1, 2, 3];    % Microrredes a analizar
LIMITE_CALIDAD = 0.5;   % Umbral visual

% Directorio de salida
out_dir = 'figures_comparison_quality';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('==========================================================\n');
fprintf('   ANÁLISIS COMPARATIVO DE CALIDAD LIME (AR vs TS)\n');
fprintf('   Escenario: %s\n', SCENARIO_TYPE);
fprintf('==========================================================\n\n');

% Definir patrones de archivo y títulos (Sintaxis TeX, sin $)
if strcmp(SCENARIO_TYPE, 'PUMP')
    file_pattern = 'lime_temporal_PUMP_%s_MG%d_7days_MEAN.mat';
    title_var = 'Bombeo (Q_p)';
else
    file_pattern = 'lime_temporal_%s_MG%d_7days_MEAN.mat';
    title_var = 'Intercambio (Q_t)';
end

% --- BUCLE PRINCIPAL POR MICRORRED ---
for t_idx = TARGETS
    
    fprintf('>>> PROCESANDO MICRORRED %d <<<\n', t_idx);
    
    % 1. CARGAR DATOS (AR y TS)
    fname_AR = sprintf(file_pattern, 'AR', t_idx);
    fname_TS = sprintf(file_pattern, 'TS', t_idx);
    
    if ~isfile(fname_AR) || ~isfile(fname_TS)
        fprintf('  [!] Faltan archivos para MG%d (AR o TS). Saltando...\n', t_idx);
        continue;
    end
    
    data_AR = load(fname_AR); 
    data_TS = load(fname_TS);
    
    R2_AR = data_AR.temporal_results.quality_history;
    R2_TS = data_TS.temporal_results.quality_history;
    t_days = data_AR.temporal_results.time_days + 1;
    
    % Sincronizar longitudes (por seguridad)
    min_len = min(length(R2_AR), length(R2_TS));
    R2_AR = R2_AR(1:min_len);
    R2_TS = R2_TS(1:min_len);
    t_days = t_days(1:min_len);
    
    % 2. CÁLCULO DE ESTADÍSTICAS Y TABLA
    stats_AR = [mean(R2_AR, 'omitnan'), min(R2_AR), std(R2_AR, 'omitnan')];
    stats_TS = [mean(R2_TS, 'omitnan'), min(R2_TS), std(R2_TS, 'omitnan')];
    
    % Crear Tabla Formal
    Maus = {'AR (Lineal)'; 'TS (No-Lineal)'};
    R2_Promedio = [stats_AR(1); stats_TS(1)];
    R2_Minimo   = [stats_AR(2); stats_TS(2)];
    Desv_Std    = [stats_AR(3); stats_TS(3)];
    
    T = table(R2_Promedio, R2_Minimo, Desv_Std, 'RowNames', Maus);
    
    fprintf('  Tabla Comparativa de Fidelidad (R^2) - MG%d:\n', t_idx);
    disp(T);
    fprintf('\n');
    
    % 3. GENERAR GRÁFICO COMPARATIVO
    fig = figure('Name', sprintf('Quality_MG%d', t_idx), 'Position', [100, 100, 900, 500], 'Color', 'w', 'Visible', 'off');
    
    % Curvas
    plot(t_days, R2_AR, 'b-', 'LineWidth', 2, 'DisplayName', 'Modelo AR (Lineal)');
    hold on;
    plot(t_days, R2_TS, 'r-', 'LineWidth', 2, 'DisplayName', 'Modelo TS (No-Lineal)');
    
    % Zona de Baja Fidelidad (Área gris)
    yline(LIMITE_CALIDAD, 'k--', 'Umbral de Confianza', 'LabelHorizontalAlignment', 'left', 'FontSize', 10);
    fill([t_days, fliplr(t_days)], [ones(1,min_len)*LIMITE_CALIDAD, zeros(1,min_len)], ...
         [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Zona de Baja Fidelidad');
    
    % Estética (USANDO INTÉRPRETE TEX - MÁS ROBUSTO)
    % Cell array para múltiples líneas
    linea1 = sprintf('Comparación de Fidelidad Local (R^2): %s', title_var);
    linea2 = sprintf('Agente: Microgrid %d', t_idx);
    
    title({linea1, linea2}, 'Interpreter', 'tex', 'FontSize', 14, 'FontName', 'Times New Roman');
    
    ylabel('Coeficiente de Determinación (R^2)', 'Interpreter', 'tex', 'FontSize', 12, 'FontName', 'Times New Roman');
    xlabel('Tiempo [Días]', 'FontSize', 12, 'FontName', 'Times New Roman');
    
    ylim([0, 1.05]); 
    xlim([1, 8]); 
    xticks(1:8);
    grid on;
    legend('Location', 'southwest', 'FontSize', 11, 'Interpreter', 'tex');
    
    % 4. EXPORTAR (SOLO PDF)
    out_name = sprintf('Comparison_Quality_R2_%s_MG%d', SCENARIO_TYPE, t_idx);
    full_path = fullfile(out_dir, out_name);
    
    % Exportar vectorialmente
    exportgraphics(fig, [full_path '.pdf'], 'ContentType', 'vector');
    
    fprintf('  > Gráfico guardado en: %s.pdf\n\n', full_path);
    close(fig);
    
end

fprintf('--- PROCESO COMPLETADO PARA TODAS LAS MICRORREDES ---\n');