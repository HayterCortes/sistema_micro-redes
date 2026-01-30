%% --- File: plot_paper_Qt_Qp_strict_notation_v4.m ---
% VISUALIZATION SCRIPT: WATER SHARING & EXTRACTION (IEEE FORMAT)
%
% Correction v4: 
% - Fixes 'K_TARGET' vs 'k_target' case sensitivity crash.
% - Uses robust path handling for ../results_mpc
%--------------------------------------------------------------------------
clear; clc; close all;

% --- USER CONFIGURATION ---
TIPO_MODELO = 'AR'; % 'AR' or 'TS'
dir_figs = 'figures_paper'; 

% Colors
colors_mg = [0.00, 0.4470, 0.7410;   % MG1 (Blue)
             0.8500, 0.3250, 0.0980; % MG2 (Red/Orange)
             0.9290, 0.6940, 0.1250];% MG3 (Yellow/Gold)

scenarios = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
markers_scn = {'o', 's', '^'}; 

% --- 1. LOAD DATA ---
fprintf('--- INICIANDO GRAFICADO (%s) ---\n', TIPO_MODELO);
try
    % Ruta relativa para salir de XAI y entrar a results_mpc
    fname = sprintf('../results_mpc/resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    
    if ~isfile(fname)
        fprintf('No se halló %s, intentando genérico...\n', fname);
        fname = '../results_mpc/resultados_mpc_3mg_7dias.mat'; 
    end
    
    if ~isfile(fname)
        % Último intento: buscar en la carpeta actual por si moviste los archivos
        fname = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
        if ~isfile(fname), error('No se encuentran los resultados .mat'); end
    end

    res = load(fname);
    Q_t = res.Q_t; Q_p = res.Q_p;
    Ts_sim = res.mg(1).Ts_sim;
    t_days = (0:size(Q_t, 1)-1)' * (Ts_sim / 86400); 
    fprintf('Datos de simulación cargados correctamente.\n');

catch ME
    error('Error cargando datos: %s', ME.message);
end

% --- 2. GET LIME MARKERS (ROBUST LOAD) ---
events = [];
fprintf('\n--- BUSCANDO MARCADORES LIME ---\n');

for s = 1:length(scenarios)
    for m = 1:3
        % Definir nombres potenciales
        f_scenario = sprintf('lime_Scenario_%s_%s_MG%d_MEAN.mat', scenarios{s}, TIPO_MODELO, m);
        f_pump = sprintf('lime_PUMP_%s_%s_MG%d_MEAN.mat', scenarios{s}, TIPO_MODELO, m);
        
        file_to_load = '';
        
        % Prioridad para Altruismo MG1 (usar archivo PUMP)
        if strcmp(scenarios{s}, 'Altruismo') && m == 1 && isfile(f_pump)
            file_to_load = f_pump;
        elseif isfile(f_scenario)
            file_to_load = f_scenario;
        end
        
        if ~isempty(file_to_load)
            % --- CORRECCIÓN CRÍTICA AQUÍ ---
            % Cargamos TODO el archivo para evitar errores de nombre variable
            dat = load(file_to_load); 
            
            % Buscamos la variable correcta
            if isfield(dat, 'K_TARGET')
                k = dat.K_TARGET;
            elseif isfield(dat, 'k_target')
                k = dat.k_target;
            else
                warning('Archivo %s existe pero no tiene variable k_target. Saltando.', file_to_load);
                continue;
            end
            % -------------------------------
            
            e.k = k; 
            e.t = t_days(k); 
            e.val_Qs = Q_t(k, m); 
            e.val_Qp = Q_p(k, m);
            e.mg = m; 
            e.scn_idx = s;
            
            if isempty(events), events = e; else, events(end+1) = e; end
            
            fprintf('[FOUND] %-18s MG%d -> t=%.2f dias (Archivo: %s)\n', ...
                scenarios{s}, m, e.t, file_to_load);
        end
    end
end

if isempty(events)
    warning('NO SE ENCONTRARON MARCADORES. Verifica los nombres de archivo.');
end

% --- 3. PLOT GENERATION (IEEE DOUBLE COLUMN) ---
fig_width = 7.16; fig_height = 3.5;
fig = figure('Units', 'inches', 'Position', [1, 1, fig_width, fig_height], ...
    'Color', 'w', 'PaperPositionMode', 'auto');

% --- SUBPLOT A: SHARED FLOW (Q_s) ---
ax1 = subplot(1, 2, 1); hold on; grid on; box on;
yline(0, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');

% Plot Lines
p_h = gobjects(3,1);
for m = 1:3
    p_h(m) = plot(t_days, Q_t(:, m), '-', 'Color', colors_mg(m,:), ...
        'LineWidth', 1.0, 'DisplayName', sprintf('MG%d', m));
end

% Plot Markers
for i = 1:length(events)
    ev = events(i);
    scatter(ev.t, ev.val_Qs, 45, colors_mg(ev.mg,:), 'filled', ...
        markers_scn{ev.scn_idx}, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

ylabel('Shared Water Flow $Q_{s}^{i}$ [L/s]', 'Interpreter', 'latex');
xlabel('Time [Days]', 'Interpreter', 'latex');
title('\textbf{(a) Water Sharing Dynamics}', 'Interpreter', 'latex');
xlim([0 7]); ylim([-5 5]);
legend(p_h, 'Location', 'southwest', 'FontSize', 8, 'NumColumns', 3, 'Interpreter', 'latex');

% --- SUBPLOT B: PUMPED FLOW (Q_p) ---
ax2 = subplot(1, 2, 2); hold on; grid on; box on;

% Plot Lines
for m = 1:3
    plot(t_days, Q_p(:, m), '-', 'Color', colors_mg(m,:), 'LineWidth', 1.0);
end

% Legend Entries for Scenarios
scn_h = gobjects(3,1);
display_names = {'Global Peak', 'Altruismo', 'Direct Satisfaction'};
for s = 1:3
    scn_h(s) = plot(nan, nan, markers_scn{s}, 'Color', 'k', ...
        'MarkerFaceColor', 'w', 'DisplayName', display_names{s});
end

% Plot Real Markers
for i = 1:length(events)
    ev = events(i);
    if ev.val_Qp > 0.1 || strcmp(scenarios{ev.scn_idx}, 'Altruismo')
        scatter(ev.t, ev.val_Qp, 45, colors_mg(ev.mg,:), 'filled', ...
            markers_scn{ev.scn_idx}, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    end
end

ylabel('Pumped Water Flow $Q_{p}^{i}$ [L/s]', 'Interpreter', 'latex');
xlabel('Time [Days]', 'Interpreter', 'latex');
title('\textbf{(b) Groundwater Extraction}', 'Interpreter', 'latex');
xlim([0 7]); ylim([0 15]);
legend(scn_h, 'Location', 'northeast', 'FontSize', 8, 'Interpreter', 'latex');

% --- EXPORT ---
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 9);
linkaxes([ax1, ax2], 'x');

outfile = sprintf('Fig_Paper_StrictNotation_%s.pdf', TIPO_MODELO);

try
    if ~exist(dir_figs, 'dir'), mkdir(dir_figs); end
    full_out = fullfile(dir_figs, outfile);
    exportgraphics(fig, full_out, 'ContentType', 'vector');
    fprintf('\nSUCCESS: Figura guardada en:\n  -> %s\n', full_out);
catch
    fprintf('Guardando en directorio local por error de ruta...\n');
    exportgraphics(fig, outfile, 'ContentType', 'vector');
    fprintf('SUCCESS: Figura guardada localmente: %s\n', outfile);
end