%% --- File: plot_temporal_evolution_3mg_MEAN.m ---
%
% MASTER VISUALIZATION SCRIPT FOR TEMPORAL LIME (MEAN VERSION)
% Compatible with: lime_temporal_main_3mg_MEAN.m
%
% Updates:
% 1. Handles TIPO_MODELO (AR/TS).
% 2. Reads '_MEAN.mat' files.
% 3. Converts 'Mean_' features to LaTeX bar notation (\bar{x}).
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
TIPO_MODELO = 'TS';      % <--- ELIGE 'AR' O 'TS'
TARGETS = [1, 2, 3];

% Colors for Area Plot
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray
colors_map = [color_mg1; color_mg2; color_mg3; color_aq];

% Output Directories (Model Specific)
dir_paper = sprintf('figures_paper_temporal_%s', TIPO_MODELO);
dir_pres  = sprintf('figures_presentation_temporal_%s', TIPO_MODELO);

if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING TEMPORAL PLOTS (MEAN FEATURES) - Model: %s ---\n', TIPO_MODELO);

for t_idx = TARGETS
    
    % Load Data (Updated Filename Pattern)
    filename = sprintf('lime_temporal_%s_MG%d_7days_MEAN.mat', TIPO_MODELO, t_idx);
    
    if ~exist(filename, 'file')
        fprintf('  [!] File not found: %s (Skipping)\n', filename);
        continue;
    end
    
    fprintf('  > Processing MG%d...\n', t_idx);
    load(filename, 'temporal_results');
    
    % TIME SHIFT: Transform 0-7 days to 1-8 days range
    t_days = temporal_results.time_days + 1; 
    
    weights_hist = temporal_results.weights_history; 
    target_real = temporal_results.target_real_history;
    feature_names = temporal_results.feature_names;
    [N_feats, N_steps] = size(weights_hist);
    
    % --- 1. PREPARE AREA PLOT DATA ---
    influence_groups = zeros(4, N_steps); 
    for i = 1:N_feats
        name = feature_names{i};
        % Logic remains valid even with 'Mean_' prefix
        if contains(name, 'MG1', 'IgnoreCase', true), g=1;
        elseif contains(name, 'MG2', 'IgnoreCase', true), g=2;
        elseif contains(name, 'MG3', 'IgnoreCase', true), g=3;
        else, g=4; end
        influence_groups(g, :) = influence_groups(g, :) + abs(weights_hist(i, :));
    end
    total_infl = sum(influence_groups, 1);
    total_infl(total_infl < 1e-9) = 1; 
    influence_pct = (influence_groups ./ total_infl) * 100;
    
    % Neighbor Influence
    mask_neighbors = true(1, 3); mask_neighbors(t_idx) = false;
    vecinos_infl = sum(influence_pct(find(mask_neighbors), :), 1);
    
    % --- 2. PREPARE WEIGHTS EVOLUTION DATA (TOP 8) ---
    % Calculate total absolute influence for ranking
    total_abs_influence = sum(abs(weights_hist), 2);
    [~, sorted_idx] = sort(total_abs_influence, 'descend');
    
    % Select Top 8
    top_k = min(8, N_feats);
    top_indices = sorted_idx(1:top_k);
    
    top_weights_hist = weights_hist(top_indices, :);
    top_raw_names = feature_names(top_indices);
    
    % Generate Latex Labels for Top 8 (Updated for MEAN)
    top_labels = cell(top_k, 1);
    for k = 1:top_k
        top_labels{k} = get_latex_label(top_raw_names{k});
    end
    
    % --- 3. GENERATE PLOTS ---
    
    % A. AREA PLOT
    fname_area_pap = fullfile(dir_paper, sprintf('Temporal_Area_MG%d_Paper', t_idx));
    create_area_plot(t_days, influence_pct, colors_map, t_idx, 'paper', fname_area_pap);
    
    fname_area_pres = fullfile(dir_pres, sprintf('Temporal_Area_MG%d_Slide', t_idx));
    create_area_plot(t_days, influence_pct, colors_map, t_idx, 'presentation', fname_area_pres);
    
    % B. CORRELATION PLOT
    fname_corr_pap = fullfile(dir_paper, sprintf('Temporal_Corr_MG%d_Paper', t_idx));
    create_corr_plot(t_days, target_real, vecinos_infl, t_idx, 'paper', fname_corr_pap);
    
    fname_corr_pres = fullfile(dir_pres, sprintf('Temporal_Corr_MG%d_Slide', t_idx));
    create_corr_plot(t_days, target_real, vecinos_infl, t_idx, 'presentation', fname_corr_pres);
    
    % C. WEIGHTS EVOLUTION PLOT
    fname_w_pap = fullfile(dir_paper, sprintf('Temporal_Weights_MG%d_Paper', t_idx));
    create_weight_evolution_plot(t_days, top_weights_hist, top_labels, t_idx, 'paper', fname_w_pap);
    
    fname_w_pres = fullfile(dir_pres, sprintf('Temporal_Weights_MG%d_Slide', t_idx));
    create_weight_evolution_plot(t_days, top_weights_hist, top_labels, t_idx, 'presentation', fname_w_pres);
    
end
fprintf('--- ALL TEMPORAL PLOTS EXPORTED SUCCESSFULLY FOR %s ---\n', TIPO_MODELO);


%% --- HELPER: LABEL PARSER (UPDATED FOR MEAN) ---
function label = get_latex_label(raw_name)
    % 1. Detect and strip 'Mean_'
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    clean_name = regexprep(raw_name, 'Mean_', '', 'ignorecase');
    
    % 2. Identify Owner
    g = 4; 
    if contains(clean_name, 'MG1', 'IgnoreCase', true), g=1;
    elseif contains(clean_name, 'MG2', 'IgnoreCase', true), g=2;
    elseif contains(clean_name, 'MG3', 'IgnoreCase', true), g=3;
    end
    
    % 3. Identify Symbol
    if contains(clean_name, 'SoC', 'IgnoreCase', true)
        sym = 'SoC';
    elseif contains(clean_name, 'tank', 'IgnoreCase', true) || contains(clean_name, 'Estanque', 'IgnoreCase', true)
        sym = 'V_{Tank}';
    elseif contains(clean_name, 'P_dem', 'IgnoreCase', true)
        sym = 'P_{L}';
    elseif contains(clean_name, 'P_gen', 'IgnoreCase', true)
        sym = 'P_{G}';
    elseif contains(clean_name, 'Q_dem', 'IgnoreCase', true)
        sym = 'Q_{L}';
    elseif contains(clean_name, 'aq', 'IgnoreCase', true)
        sym = 'EAW'; g=4;
    else
        % Fallback cleanup
        core = regexprep(clean_name, 'MG\d_', ''); 
        sym = strrep(core, '_', '\_');
    end
    
    % 4. Apply Mean Bar (\bar) vs Estimate Hat (\hat) vs Plain
    % Logic: If it was 'Mean_', use \bar. If it's a prediction variable but not mean (unlikely here), use \hat.
    if is_mean
        final_sym = sprintf('\\bar{%s}', sym);
    elseif ismember(sym, {'P_{L}', 'P_{G}', 'Q_{L}'}) 
        % If it's a prediction variable but for some reason not flagged as mean
        final_sym = sprintf('\\hat{%s}', sym);
    else
        final_sym = sym;
    end
    
    % 5. Construct Final Label with Owner
    if g < 4
        label = sprintf('$%s^{%d}$', final_sym, g);
    else
        label = sprintf('$%s$', final_sym);
    end
end


%% --- PLOT 1: STACKED AREA ---
function create_area_plot(t, data_pct, cmap, mg_idx, mode, filename)
    if strcmp(mode, 'paper')
        fig_width = 7; fig_height = 4.5; font_ax = 10; font_tit = 12;
    else
        fig_width = 14; fig_height = 8; font_ax = 14; font_tit = 18;
    end
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    h = area(t, data_pct');
    h(1).FaceColor = cmap(1,:); h(1).DisplayName = 'Microgrid 1';
    h(2).FaceColor = cmap(2,:); h(2).DisplayName = 'Microgrid 2';
    h(3).FaceColor = cmap(3,:); h(3).DisplayName = 'Microgrid 3';
    h(4).FaceColor = cmap(4,:); h(4).DisplayName = 'Aquifer';
    
    title(sprintf('Evolution of Cooperative Dependency by Agent for $Q_{t}^{%d}$', mg_idx), 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    ylabel('Relative Influence [%]', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    xlim([1 8]); ylim([0 100]); xticks(1:7);
    
    ax = gca; set(ax, 'FontName', 'Times New Roman', 'FontSize', font_ax, 'Layer', 'top');
    grid on; box on;
    for d = 2:7, xline(d, 'k--', 'Alpha', 0.4); end
    legend(h, 'Location', 'eastoutside', 'Interpreter', 'latex', 'FontSize', font_ax);
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); close(fig);
end


%% --- PLOT 2: CORRELATION ---
function create_corr_plot(t, real_Qt, neigh_infl, mg_idx, mode, filename)
    if strcmp(mode, 'paper')
        fig_width = 7; fig_height = 5; font_ax = 10; font_tit = 12; line_w = 1.2;
    else
        fig_width = 14; fig_height = 8; font_ax = 14; font_tit = 18; line_w = 2.0;
    end
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    yyaxis left
    plot(t, real_Qt, 'b-', 'LineWidth', line_w);
    ylabel(sprintf('Actual Water Exchange $Q_{t}^{%d}$ [L/s]', mg_idx), 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold', 'Interpreter', 'latex');
    ax = gca; ax.YColor = [0 0 0.8];
    yline(0, 'k-', 'LineWidth', 1.0, 'Alpha', 0.5);
    
    yl = ylim; range_y = yl(2)-yl(1);
    text(1.2, yl(2)-range_y*0.05, 'EXPORT ZONE (Sending)', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', font_ax-2, 'FontWeight', 'bold');
    text(1.2, yl(1)+range_y*0.05, 'IMPORT ZONE (Receiving)', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', font_ax-2, 'FontWeight', 'bold');
    
    yyaxis right
    plot(t, neigh_infl, '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', line_w);
    ylabel('Neighbor Influence [%] (LIME)', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    ax.YColor = [0.8500 0.3250 0.0980]; ylim([0 100]);
    
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    title(sprintf('Physical Flow vs. Algorithmic Detection (MG%d)', mg_idx), 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    xlim([1 8]); xticks(1:7);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', font_ax); grid on; box on;
    for d = 2:7, xline(d, 'k:', 'Alpha', 0.3); end
    legend({sprintf('$Q_{t}^{%d}$ (Real)', mg_idx), 'Neighbor Influence'}, 'Location', 'north', 'Orientation', 'horizontal', 'Interpreter', 'latex');
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); close(fig);
end


%% --- PLOT 3: WEIGHTS EVOLUTION ---
function create_weight_evolution_plot(t, weights, labels, mg_idx, mode, filename)
    if strcmp(mode, 'paper')
        fig_width = 7; fig_height = 5; font_ax = 10; font_tit = 12; line_w = 1.5;
    else
        fig_width = 14; fig_height = 8; font_ax = 14; font_tit = 18; line_w = 2.5;
    end
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    % Simple Colors (High Contrast)
    colors = lines(length(labels));
    
    hold on;
    h_lines = gobjects(length(labels), 1); 
    
    for k = 1:size(weights, 1)
        h_lines(k) = plot(t, weights(k, :), ...
             'LineStyle', '-', ...   
             'Color', colors(k,:), ...
             'LineWidth', line_w);
    end
    
    title(sprintf('Temporal Evolution of LIME Weights for $Q_{t}^{%d}$', mg_idx), ...
          'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    ylabel('Average Influence (Mean Features)', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    
    xlim([1 8]); xticks(1:7);
    yline(0, 'k-', 'LineWidth', 1.0, 'Alpha', 0.5); 
    
    ax = gca;
    set(ax, 'FontName', 'Times New Roman', 'FontSize', font_ax);
    grid on; box on;
    
    for d = 2:7, xline(d, 'k:', 'Alpha', 0.3); end
    
    legend(h_lines, labels, 'Location', 'eastoutside', 'Interpreter', 'latex', 'FontSize', font_ax);
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); 
    close(fig);
end