%% --- File: plot_lime_final_2lines_FIXED.m ---
%
% VISUALIZATION SCRIPT FOR PAPER - FIXED LATEX SYNTAX
% Logic: Top features until |w_k| > sum(abs(remaining weights)).
% Titles: Corrected 2-line format (fixed $ delimiters).
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURACIÓN DE CASOS SELECCIONADOS ---
selected_cases = {
    'GlobalPeak',         1, 'Qs'; ... % Scenario A
    'GlobalPeak',         2, 'Qs'; ... % Scenario A
    'Altruismo',          1, 'Qs'; ... % Scenario B
    'DirectSatisfaction', 2, 'Qs'; ... % Scenario C
    'DirectSatisfaction', 3, 'Qs'; ... % Scenario C
    'EnergyEfficiency',   1, 'Qp'; ... % Scenario D
    'EnergyEfficiency',   2, 'Qp'; ... % Scenario D
    'EnergyEfficiency',   3, 'Qp'      % Scenario D
};

% Colores de Agentes
color_mg1 = [0 0.4470 0.7410];      
color_mg2 = [0.8500 0.3250 0.0980]; 
color_mg3 = [0.9290 0.6940 0.1250]; 
color_aq  = [0.5 0.5 0.5];          
color_others = [0.7 0.7 0.7]; 

dir_paper = 'figures_paper_final';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end

fprintf('--- GENERANDO GRÁFICOS LIME (SINTAXIS LATEX CORREGIDA) ---\n');

for i = 1:size(selected_cases, 1)
    scn_raw  = selected_cases{i, 1};
    t_idx    = selected_cases{i, 2};
    var_type = selected_cases{i, 3};
    
    if strcmp(var_type, 'Qs')
        filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scn_raw, t_idx);
        prefix_save = 'Qs_';
        switch scn_raw
            case 'GlobalPeak', s_title = 'Scenario A: Global Peak Interaction';
            case 'Altruismo', s_title = 'Scenario B: Active Water Export';
            case 'DirectSatisfaction', s_title = 'Scenario C: Direct Demand Satisfaction';
        end
        sym_tex = 'Q_{s}';
    else
        filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scn_raw, t_idx);
        prefix_save = 'Qp_';
        s_title = 'Scenario D: Coordinated Pumping';
        sym_tex = 'Q_{p}';
    end

    if ~exist(filename, 'file'), continue; end
    data = load(filename);
    
    % --- TIEMPO ---
    Ts_sim = 60; 
    t_seconds = (data.K_TARGET - 1) * Ts_sim;
    day_num = floor(t_seconds / 86400) + 1;
    rem_seconds = mod(t_seconds, 86400);
    hour_val = floor(rem_seconds / 3600);
    min_val = round((rem_seconds - hour_val*3600) / 60);
    
    % --- PESOS ---
    feature_names = data.feature_names;
    X_values = data.estado.X_original;
    num_runs = length(data.all_explanations);
    weights_matrix = zeros(length(feature_names), num_runs);
    for r = 1:num_runs
        run_data = data.all_explanations{r};
        map_temp = containers.Map(run_data(:,1), [run_data{:,2}]);
        for f = 1:length(feature_names)
            weights_matrix(f, r) = map_temp(feature_names{f});
        end
    end
    avg_weights = mean(weights_matrix, 2);
    std_weights = std(weights_matrix, 0, 2);
    
    % --- CORTE DINÁMICO ---
    [sorted_abs_w, abs_sort_idx] = sort(abs(avg_weights), 'descend');
    tail_sums = zeros(length(avg_weights), 1);
    for f = 1:length(avg_weights)-1, tail_sums(f) = sum(sorted_abs_w(f+1:end)); end
    
    cutoff_idx = find(sorted_abs_w > tail_sums, 1, 'first');
    if isempty(cutoff_idx) || cutoff_idx < 3, cutoff_idx = min(3, length(avg_weights)); end
    
    top_orig_idx = abs_sort_idx(1:cutoff_idx);
    final_weights = avg_weights(top_orig_idx);
    final_std = std_weights(top_orig_idx);
    
    % Otros
    others_orig_idx = abs_sort_idx(cutoff_idx+1:end);
    if ~isempty(others_orig_idx)
        final_weights = [final_weights; sum(avg_weights(others_orig_idx))];
        final_std = [final_std; sqrt(sum(std_weights(others_orig_idx).^2))];
        is_others = [zeros(length(top_orig_idx), 1); 1];
    else, is_others = zeros(length(top_orig_idx), 1); end
    
    % Etiquetas
    final_labels = cell(length(final_weights), 1);
    groups_vec = zeros(length(feature_names), 1);
    for f = 1:length(feature_names)
        g_owner = 4;
        if contains(feature_names{f}, 'MG1'), g_owner=1;
        elseif contains(feature_names{f}, 'MG2'), g_owner=2;
        elseif contains(feature_names{f}, 'MG3'), g_owner=3; end
        groups_vec(f) = g_owner;
    end
    for k = 1:length(top_orig_idx)
        idx_orig = top_orig_idx(k);
        final_labels{k} = get_mean_latex_label(feature_names{idx_orig}, X_values(idx_orig), groups_vec(idx_orig));
    end
    if any(is_others), final_labels{end} = 'Others (Sum)'; end
    
    % Interacción
    influence_per_agent = zeros(1, 4);
    for g = 1:4, influence_per_agent(g) = sum(abs(avg_weights(groups_vec == g))); end
    influence_pct = (influence_per_agent / sum(influence_per_agent)) * 100;
    
    % Dirección
    real_val = data.estado.Y_target_real_vector(t_idx);
    if strcmp(var_type, 'Qs')
        if real_val > 0, flow_str = 'Exporting'; else, flow_str = 'Importing'; end
    else
        if real_val > 0, flow_str = 'Extracting'; else, flow_str = 'Idle'; end
    end
    
    % --- PLOT ---
    fname_base = sprintf('%s%s_MG%d_Final', prefix_save, scn_raw, t_idx);
    create_ranking_plot_2lines_FIXED(final_weights, final_std, final_labels, is_others, s_title, ...
        day_num, hour_val, min_val, real_val, t_idx, sym_tex, flow_str, ...
        fullfile(dir_paper, [fname_base '_Rank']));
        
    create_interaction_plot_2lines_FIXED(influence_pct, [color_mg1; color_mg2; color_mg3; color_aq], s_title, ...
        day_num, hour_val, min_val, real_val, t_idx, sym_tex, fullfile(dir_paper, [fname_base '_Int']));
end

fprintf('--- PROCESO FINALIZADO SIN ERRORES ---\n');

%% --- FUNCIONES CORREGIDAS ---

function label = get_mean_latex_label(raw_name, val, g_owner)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    core_name = regexprep(raw_name, '(MG\d_|Mean_)', '', 'ignorecase');
    if contains(core_name, 'SoC'), sym = 'SoC'; val_fmt = '%.1f\\%%'; val = val*100;
    elseif contains(core_name, 'tank'), sym = 'V_{T}'; val_fmt='%.0f L';
    elseif contains(core_name, 'P_dem'), sym = 'P_{L}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'P_gen'), sym = 'P_{G}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'Q_dem'), sym = 'Q_{L}'; val_fmt='%.2f L/s';
    elseif contains(core_name, 'aq'), sym = 'EAW'; val_fmt='%.0f L'; g_owner=4;
    else, sym = 'X'; val_fmt='%.2f'; end
    final_sym = sym; if is_mean, final_sym = sprintf('\\bar{%s}', sym); end
    val_str = sprintf(val_fmt, val);
    if g_owner < 4, label = sprintf('$%s^{%d}$ (%s)', final_sym, g_owner, val_str);
    else, label = sprintf('$%s$ (%s)', final_sym, val_str); end
end

function create_ranking_plot_2lines_FIXED(weights, errors, labels, is_others, title_text, d, h, m, val, mg, sym_tex, flow_str, fname)
    fig = figure('Units','inches','Position',[0 0 7 6],'Visible','off','Color','w');
    N = length(weights); colors = zeros(N,3);
    for i=1:N
        if is_others(i), colors(i,:) = [0.7 0.7 0.7];
        elseif weights(i)>=0, colors(i,:) = [0.466 0.674 0.188];
        else, colors(i,:) = [0.635 0.078 0.184]; end
    end
    barh(weights, 0.6, 'FaceColor','flat', 'CData', colors); hold on;
    errorbar(weights, 1:N, errors, 'horizontal', 'k', 'LineStyle','none');
    ax = gca;
    set(ax, 'YTick', 1:N, 'YTickLabel', labels, 'YDir','reverse', ...
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', 'FontSize',10);
    xlabel('Average Influence (LIME Weight)', 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','bold');
    
    % CORRECCIÓN DE SINTAXIS AQUÍ: Se eliminó el $ extra después de L/s
    line1 = title_text;
    line2 = sprintf('Target: MG%d | Day %d, %02d:%02d | $%s^{%d} = %.2f$ L/s (%s)', ...
                    mg, d, h, m, sym_tex, mg, val, flow_str);
    
    title({line1; line2}, 'FontName','Times New Roman', 'FontSize',11, 'Interpreter','latex');
    grid on; xline(0,'k-');
    xlim_val = max(abs(weights))*1.3; if xlim_val < 0.1, xlim_val = 0.1; end
    xlim([-xlim_val xlim_val]);
    ax.Position = [0.35 0.12 0.60 0.75];
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end

function create_interaction_plot_2lines_FIXED(pct, cmap, title_text, d, h, m, val, mg, sym_tex, fname)
    fig = figure('Units','inches','Position',[0 0 6 4.5],'Visible','off','Color','w');
    b = bar(1:4, pct, 0.6, 'FaceColor','flat'); b.CData = cmap;
    ylabel('Total Influence [%]', 'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'MG 1','MG 2','MG 3','Aquifer'}, ...
        'FontName','Times New Roman','FontSize',10);
    for i=1:4, text(i, pct(i)+4, sprintf('%.1f%%', pct(i)), 'HorizontalAlignment','center', ...
        'FontName','Times New Roman','FontSize',10,'FontWeight','bold'); end
    
    line1 = ['Interaction Analysis: ' title_text];
    line2 = sprintf('Target: MG%d | Day %d, %02d:%02d | $%s^{%d} = %.2f$ L/s', ...
                    mg, d, h, m, sym_tex, mg, val);
    
    title({line1; line2}, 'FontName','Times New Roman','FontSize',11, 'Interpreter','latex');
    ylim([0 120]); grid on;
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end