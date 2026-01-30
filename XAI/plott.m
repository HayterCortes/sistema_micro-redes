%% --- File: plot_lime_selected_cases_v8.m ---
%
% VISUALIZATION SCRIPT FOR PAPER - FINAL FINAL (V8)
% Changes based on v7: 
% 1. REMOVED "(Export)" and "(Import)" text from titles.
% 2. SPECIAL CASE: Scenario C (DirectSatisfaction) MG2 -> Fixed Top 8 cutoff.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURACIÓN DE CASOS SELECCIONADOS ---
selected_cases = {
    'GlobalPeak',         1, 'Qs'; ... 
    'GlobalPeak',         2, 'Qs'; ... 
    'Altruismo',          1, 'Qs'; ... 
    'Altruismo',          1, 'Qp'; ... 
    'DirectSatisfaction', 2, 'Qs'; ... 
    'DirectSatisfaction', 3, 'Qs'      
};

% Colores
color_mg1 = [0 0.4470 0.7410];      
color_others = [0.7 0.7 0.7]; 

% Directorio de salida
dir_paper = 'figures_paper_final';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end

fprintf('--- GENERANDO GRÁFICOS (NO IMPORT/EXPORT TEXT + SCEN C MG2 FIX) ---\n');

for i = 1:size(selected_cases, 1)
    scn_raw  = selected_cases{i, 1};
    t_idx    = selected_cases{i, 2};
    var_type = selected_cases{i, 3};
    
    if strcmp(var_type, 'Qs')
        filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scn_raw, t_idx);
        prefix_save = 'Qs_';
        sym_tex = 'Q_{s}';
    else
        filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scn_raw, t_idx);
        prefix_save = 'Qp_';
        sym_tex = 'Q_{p}';
    end

    switch scn_raw
        case 'GlobalPeak', s_title = 'Scenario A: Global Peak';
        case 'Altruismo',  s_title = 'Scenario B: Altruism';
        case 'DirectSatisfaction', s_title = 'Scenario C: Direct Satisfaction';
    end

    if ~exist(filename, 'file')
        fprintf('Archivo no encontrado: %s\n', filename); continue; 
    end
    data = load(filename);
    
    % --- RECUPERAR DATOS ---
    if isfield(data, 'K_TARGET'), k_actual = data.K_TARGET;
    elseif isfield(data, 'k_target'), k_actual = data.k_target;
    else, k_actual = 1; end
    
    % Tiempo
    Ts_sim = 60; t_seconds = (k_actual - 1) * Ts_sim;
    day_num = floor(t_seconds / 86400) + 1;
    rem_seconds = mod(t_seconds, 86400);
    hour_val = floor(rem_seconds / 3600);
    min_val = round((rem_seconds - hour_val*3600) / 60);
    
    % Pesos
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
    
    % --- CORTE (LÓGICA MODIFICADA AQUÍ) ---
    [sorted_abs_w, abs_sort_idx] = sort(abs(avg_weights), 'descend');
    
    % >> MODIFICACIÓN 2: Lógica especial para Escenario C, MG 2 <<
    if strcmp(scn_raw, 'DirectSatisfaction') && t_idx == 2
        % Forzar Top 8 para este caso específico
        cutoff_idx = 8;
    else
        % Lógica dinámica normal para el resto
        tail_sums = zeros(length(avg_weights), 1);
        for f = 1:length(avg_weights)-1, tail_sums(f) = sum(sorted_abs_w(f+1:end)); end
        
        cutoff_idx = find(sorted_abs_w > tail_sums, 1, 'first');
        if isempty(cutoff_idx) || cutoff_idx < 4, cutoff_idx = min(4, length(avg_weights)); end
    end
    
    % Seguridad: no exceder el número total de features disponibles
    cutoff_idx = min(cutoff_idx, length(avg_weights));
    
    top_orig_idx = abs_sort_idx(1:cutoff_idx);
    final_weights = avg_weights(top_orig_idx);
    final_std = std_weights(top_orig_idx);
    
    % Others
    others_orig_idx = abs_sort_idx(cutoff_idx+1:end);
    if ~isempty(others_orig_idx)
        final_weights = [final_weights; sum(avg_weights(others_orig_idx))];
        final_std = [final_std; sqrt(sum(std_weights(others_orig_idx).^2))];
        is_others = [zeros(length(top_orig_idx), 1); 1];
    else, is_others = zeros(length(top_orig_idx), 1); end
    
    % --- ETIQUETAS ---
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
        final_labels{k} = get_mean_latex_label_with_units(feature_names{idx_orig}, X_values(idx_orig), groups_vec(idx_orig));
    end
    if any(is_others), final_labels{end} = '\textbf{Others}'; end
    
    % --- TEXTO DE ESTADO (MODIFICACIÓN 1) ---
    real_val = data.estado.Y_target_real_vector(t_idx);
    
    % >> MODIFICACIÓN 1: Eliminada lógica de Export/Import <<
    % Dejamos flow_str vacío siempre para que la función de plot
    % solo imprima el valor numérico y la unidad.
    flow_str = ''; 
    
    % --- PLOT ---
    fname_base = sprintf('%s%s_MG%d_Final', prefix_save, scn_raw, t_idx);
    create_ranking_plot_optimized(final_weights, final_std, final_labels, is_others, s_title, ...
        day_num, hour_val, min_val, real_val, t_idx, sym_tex, flow_str, ...
        fullfile(dir_paper, [fname_base '_Rank']));
end
fprintf('--- FINALIZADO. GRÁFICOS EN: %s ---\n', dir_paper);


%% --- FUNCIONES AUXILIARES ACTUALIZADAS ---

function create_ranking_plot_optimized(weights, errors, labels, is_others, title_text, d, h, m, val, mg, sym_tex, flow_str, fname)
    % AUMENTO DE TAMAÑO DE FIGURA: 6.0 x 5.0 pulgadas
    fig = figure('Units','inches','Position',[0 0 6.0 5.0],'Visible','off','Color','w');
    
    N = length(weights); colors = zeros(N,3);
    for i=1:N
        if is_others(i), colors(i,:) = [0.8 0.8 0.8]; 
        elseif weights(i)>=0, colors(i,:) = [0.466 0.674 0.188]; 
        else, colors(i,:) = [0.75 0.1 0.2]; 
        end
    end
    
    % BARRAS: LineWidth aumentado a 1.5
    barh(weights, 0.70, 'FaceColor','flat', 'CData', colors, 'EdgeColor','k', 'LineWidth', 1.5); hold on;
    
    % ERROR BARS: LineWidth 2.5, CapSize 12
    errorbar(weights, 1:N, errors, 'horizontal', 'k', 'LineStyle','none', 'LineWidth', 2.5, 'CapSize', 12);
    
    ax = gca;
    
    % EJES: FontSize 20, LineWidth 1.5
    set(ax, 'YTick', 1:N, 'YTickLabel', labels, 'YDir','reverse', ...
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', ...
        'FontSize', 20, 'LineWidth', 1.5);
    
    xlabel('Mean Feature Attribution', 'FontName','Times New Roman', 'FontSize', 20, 'FontWeight','bold');
    
    % TÍTULO
    line1 = sprintf('\\textbf{%s}', title_text);
    if isempty(flow_str)
        % Este caso se ejecutará siempre ahora
        line2 = sprintf('MG%d | D%d %02d:%02d | $%s=%.2f$ L/s', mg, d, h, m, sym_tex, val);
    else
        line2 = sprintf('MG%d | D%d %02d:%02d | $%s=%.2f$ L/s (%s)', mg, d, h, m, sym_tex, val, flow_str);
    end
    
    % FontSize del título: 22
    title({line1; line2}, 'FontName','Times New Roman', 'FontSize', 20, 'Interpreter','latex');
    
    grid on; ax.XGrid = 'on'; ax.YGrid = 'off';
    xline(0,'k-', 'LineWidth', 2.0); 
    
    xlim_val = max(abs(weights))*1.35; if xlim_val < 0.1, xlim_val = 0.1; end
    xlim([-xlim_val xlim_val]);
    
    % Ajuste de márgenes
    ax.Position = [0.48 0.15 0.49 0.70];
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end

function label = get_mean_latex_label_with_units(raw_name, val, g_owner)
    core_name = regexprep(raw_name, '(MG\d_|Mean_)', '', 'ignorecase');
    
    if contains(core_name, 'SoC')
        sym = 'SoC'; val_fmt = '%.0f\\%%'; val = val*100;
    elseif contains(core_name, 'tank')
        sym = 'V_{T}'; val_fmt = '%.0f\\,L';
    elseif contains(core_name, 'P_dem')
        sym = '\bar{P}_{L}'; val_fmt = '%.1f\\,kW';
    elseif contains(core_name, 'P_gen')
        sym = '\bar{P}_{G}'; val_fmt = '%.1f\\,kW';
    elseif contains(core_name, 'Q_dem')
        sym = '\bar{Q}_{L}'; val_fmt = '%.2f\\,L/s';
    elseif contains(core_name, 'aq')
        sym = 'EAW'; g_owner = 4;
        val_fmt = '%.0f\\,L'; 
    else
        sym = 'X'; val_fmt = '%.2f';
    end
    
    val_str = sprintf(val_fmt, val);
    
    if g_owner < 4
        label = sprintf('$\\mathbf{%s^{%d}}$ (%s)', sym, g_owner, val_str);
    else
        label = sprintf('$\\mathbf{%s}$ (%s)', sym, val_str);
    end
end