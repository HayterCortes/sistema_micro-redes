%% --- File: plot_paper_FINAL_Stacked_AR.m ---
% FINAL FIGURE GENERATION FOR WCCI 2026 PAPER
%
% CONFIGURATION:
% - Layout: 2 Rows x 1 Column (Vertical Stack).
% - Size: SINGLE COLUMN (3.5 in) x ULTRA-COMPACT HEIGHT (2.2 in).
% - Legend: Top Centered (Wrapped to 2 columns).
% - Model: AR.
% - Strategy: Mark events discussed in Table II + MG3 in Scenario A.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- 1. CONFIGURATION ---
TIPO_MODELO = 'AR'; 
dir_figs = 'figures_paper';

% IEEE Colors
c_mg1 = [0.00, 0.4470, 0.7410]; % Blue
c_mg2 = [0.8500, 0.3250, 0.0980]; % Red/Orange
c_mg3 = [0.9290, 0.6940, 0.1250]; % Yellow/Gold

% Markers
mk_A = 'o'; % Global Peak
mk_B = 's'; % Altruism
mk_C = '^'; % Direct Satisfaction

% --- 2. DATA LOADING ---
fprintf('--- GENERATING FINAL STACKED FIGURE (%s) ---\n', TIPO_MODELO);
try
    fname = sprintf('../results_mpc/resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    if ~isfile(fname)
        fname = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    end
    res = load(fname);
    Q_t = res.Q_t; 
    Q_p = res.Q_p;
    
    Ts = res.mg(1).Ts_sim; 
    t_raw = (0:size(Q_t, 1)-1)' * (Ts / 86400); 
    t_plot = t_raw + 1; 
    
    target_A = 2 + (8.5/24);
    target_BC = 7 + (7.5/24);
    
    [~, k_A] = min(abs(t_plot - target_A));
    [~, k_BC] = min(abs(t_plot - target_BC));
    
catch ME
    error('Error loading data: %s', ME.message);
end

% --- 3. PLOT GENERATION ---
% CAMBIO CLAVE: Altura reducida al mínimo viable (2.2 pulgadas)
fig = figure('Units', 'inches', 'Position', [1, 1, 3.5, 2.2], ...
    'Color', 'w', 'PaperPositionMode', 'auto');

% CAMBIO CLAVE: TileSpacing 'tight' para pegar los gráficos verticalmente
t = tiledlayout(2, 1, 'TileSpacing', 'tight', 'Padding', 'tight');

% =========================================================================
% TILE 1: SHARED WATER FLOW (Q_s)
% =========================================================================
nexttile;
hold on; grid on; box on;
yline(0, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');

% Plot Lines
p1 = plot(t_plot, Q_t(:,1), '-', 'Color', c_mg1, 'LineWidth', 1.0, 'DisplayName', 'MG1');
p2 = plot(t_plot, Q_t(:,2), '-', 'Color', c_mg2, 'LineWidth', 1.0, 'DisplayName', 'MG2');
p3 = plot(t_plot, Q_t(:,3), '-', 'Color', c_mg3, 'LineWidth', 1.0, 'DisplayName', 'MG3');

% Markers
scatter(t_plot(k_A), Q_t(k_A, 1), 30, c_mg1, 'filled', mk_A, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
scatter(t_plot(k_A), Q_t(k_A, 2), 30, c_mg2, 'filled', mk_A, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
scatter(t_plot(k_A), Q_t(k_A, 3), 30, c_mg3, 'filled', mk_A, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');

scatter(t_plot(k_BC), Q_t(k_BC, 1), 35, c_mg1, 'filled', mk_B, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
scatter(t_plot(k_BC), Q_t(k_BC, 2), 30, c_mg2, 'filled', mk_C, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
scatter(t_plot(k_BC), Q_t(k_BC, 3), 30, c_mg3, 'filled', mk_C, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');

ylabel('$Q_{s}^{i}$ [L/s]', 'Interpreter', 'latex');
xlim([1 8]); xticks(1:7); xticklabels({}); 
ylim([-2 3]); 
title('\textbf{(a) Shared Water Flow}', 'Interpreter', 'latex', 'FontSize', 8);

% =========================================================================
% TILE 2: PUMPED FLOW (Q_p)
% =========================================================================
nexttile;
hold on; grid on; box on;

plot(t_plot, Q_p(:,1), '-', 'Color', c_mg1, 'LineWidth', 1.0, 'HandleVisibility', 'off');
plot(t_plot, Q_p(:,2), '-', 'Color', c_mg2, 'LineWidth', 1.0, 'HandleVisibility', 'off');
plot(t_plot, Q_p(:,3), '-', 'Color', c_mg3, 'LineWidth', 1.0, 'HandleVisibility', 'off');

scatter(t_plot(k_BC), Q_p(k_BC, 1), 35, c_mg1, 'filled', mk_B, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');

h_A = plot(nan, nan, mk_A, 'Color', 'k', 'MarkerFaceColor', 'w', 'DisplayName', 'A: Global Peak');
h_B = plot(nan, nan, mk_B, 'Color', 'k', 'MarkerFaceColor', 'w', 'DisplayName', 'B: Altruism');
h_C = plot(nan, nan, mk_C, 'Color', 'k', 'MarkerFaceColor', 'w', 'DisplayName', 'C: Direct Satisfaction');

ylabel('$Q_{p}^{i}$ [L/s]', 'Interpreter', 'latex');
xlabel('Time [Days]', 'Interpreter', 'latex');
xlim([1 8]); xticks(1:7);
ylim([0 5]); 
title('\textbf{(b) Pumped Water Flow}', 'Interpreter', 'latex', 'FontSize', 8);

% =========================================================================
% GLOBAL LEGEND
% =========================================================================
lgd = legend([p1, p2, p3, h_A, h_B, h_C], ...
    'Orientation', 'horizontal', ...
    'NumColumns', 2, ... 
    'Interpreter', 'latex', ...
    'FontSize', 8);

lgd.Layout.Tile = 'North'; 

% =========================================================================
% EXPORT
% =========================================================================
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 8); 

outfile = sprintf('Fig_Paper_Stacked_%s_SingleCol_UltraCompact.pdf', TIPO_MODELO);

try
    if ~exist(dir_figs, 'dir'), mkdir(dir_figs); end
    full_out = fullfile(dir_figs, outfile);
    exportgraphics(fig, full_out, 'ContentType', 'vector');
    fprintf('\nSUCCESS: Figura FINAL (Ultra Compact) generada en:\n  -> %s\n', full_out);
catch
    exportgraphics(fig, outfile, 'ContentType', 'vector');
    fprintf('SUCCESS: Figura guardada localmente: %s\n', outfile);
end