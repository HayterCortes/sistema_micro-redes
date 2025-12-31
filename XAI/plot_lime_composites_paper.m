%% --- File: plot_lime_composites_paper_FINAL.m ---
% Genera figuras compuestas (múltiples paneles) por escenario para el paper.
% Incluye: Corte dinámico, Barra de Otros, Títulos de 2 líneas y Estética Paper.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURACIÓN DE COMPOSICIONES ---
% {Escenario, Lista_MGs, Tipo_Variable, Nombre_Archivo_Salida}
composites = {
    'GlobalPeak',         [1, 2],    'Qs', 'Scenario_A_Composite'; ...
    'DirectSatisfaction', [2, 3],    'Qs', 'Scenario_C_Composite'; ...
    'EnergyEfficiency',   [1, 2, 3], 'Qp', 'Scenario_D_Composite'
};

dir_paper = 'figures_paper_composites';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end

fprintf('--- GENERANDO FIGURAS COMPUESTAS PARA EL PAPER ---\n');

for c = 1:size(composites, 1)
    scn_raw  = composites{c,1};
    mgs      = composites{c,2};
    var_type = composites{c,3};
    out_name = composites{c,4};
    
    num_panels = length(mgs);
    % Ajuste de tamaño: Ancho dinámico según el número de sub-gráficos
    fig_width = 5.5 * num_panels; 
    fig = figure('Units','inches','Position',[0 0 fig_width 7],'Visible','off','Color','w');
    tlo = tiledlayout(1, num_panels, 'TileSpacing', 'Compact', 'Padding', 'Loose');
    
    for p = 1:num_panels
        t_idx = mgs(p);
        
        % 1. Determinar Archivo y Títulos
        if strcmp(var_type, 'Qs')
            filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scn_raw, t_idx);
            switch scn_raw
                case 'GlobalPeak', s_title = 'Scenario A: Global Peak Interaction';
                case 'DirectSatisfaction', s_title = 'Scenario C: Direct Satisfaction';
            end
            sym_tex = 'Q_{s}';
        else
            filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scn_raw, t_idx);
            s_title = 'Scenario D: Coordinated Pumping';
            sym_tex = 'Q_{p}';
        end
        
        if ~exist(filename, 'file')
            fprintf('  [!] Saltando: %s (No encontrado)\n', filename);
            continue; 
        end
        data = load(filename);
        
        % 2. Procesamiento Temporal
        Ts_sim = 60; 
        t_sec = (data.K_TARGET - 1) * Ts_sim;
        d = floor(t_sec / 86400) + 1;
        rem_s = mod(t_sec, 86400);
        h = floor(rem_s / 3600);
        m = round((rem_s - h*3600) / 60);
        
        % 3. Procesamiento de Pesos y Corte Dinámico
        f_names = data.feature_names;
        X_val   = data.estado.X_original;
        num_r   = length(data.all_explanations);
        N_f     = length(f_names);
        w_mat   = zeros(N_f, num_r);
        for r = 1:num_r
            run_d = data.all_explanations{r};
            map_t = containers.Map(run_d(:,1), [run_d{:,2}]);
            for f = 1:N_f, w_mat(f, r) = map_t(f_names{f}); end
        end
        avg_w = mean(w_mat, 2);
        std_w = std(w_mat, 0, 2);
        
        [sorted_abs_w, abs_sort_idx] = sort(abs(avg_w), 'descend');
        tail_sums = zeros(N_f, 1);
        for f = 1:N_f-1, tail_sums(f) = sum(sorted_abs_w(f+1:end)); end
        
        cutoff = find(sorted_abs_w > tail_sums, 1, 'first');
        if isempty(cutoff) || cutoff < 3, cutoff = min(3, N_f); end
        
        % Variables Top y "Otros"
        top_idx = abs_sort_idx(1:cutoff);
        others_idx = abs_sort_idx(cutoff+1:end);
        
        f_weights = avg_w(top_idx);
        f_std     = std_w(top_idx);
        f_is_others = zeros(length(top_idx), 1);
        
        if ~isempty(others_idx)
            f_weights = [f_weights; sum(avg_w(others_idx))];
            f_std     = [f_std; sqrt(sum(std_w(others_idx).^2))];
            f_is_others = [f_is_others; 1];
        end
        
        % 4. Generación de Etiquetas LaTeX
        f_labels = cell(length(f_weights), 1);
        for k = 1:length(top_idx)
            idx_orig = top_idx(k);
            g_owner = 4;
            if contains(f_names{idx_orig}, 'MG1'), g_owner=1;
            elseif contains(f_names{idx_orig}, 'MG2'), g_owner=2;
            elseif contains(f_names{idx_orig}, 'MG3'), g_owner=3; end
            f_labels{k} = get_mean_latex_label_internal(f_names{idx_orig}, X_val(idx_orig), g_owner);
        end
        if any(f_is_others), f_labels{end} = 'Others (Sum)'; end
        
        % 5. Dirección y Valor Real
        real_v = data.estado.Y_target_real_vector(t_idx);
        if strcmp(var_type, 'Qs')
            if real_v > 0, f_str = 'Exporting'; else, f_str = 'Importing'; end
        else
            if real_v > 0, f_str = 'Extracting'; else, f_str = 'Idle'; end
        end
        
        % 6. Renderizar Subpanel (Tile)
        nexttile;
        render_ranking_tile(f_weights, f_std, f_labels, f_is_others, s_title, ...
                            d, h, m, real_v, t_idx, sym_tex, f_str);
    end
    
    % Guardar figura compuesta
    exportgraphics(fig, fullfile(dir_paper, [out_name '.pdf']), 'ContentType','vector');
    close(fig);
    fprintf('  [OK] Compuesto guardado: %s.pdf\n', out_name);
end

%% --- FUNCIONES INTERNAS (Encapsuladas para evitar errores de variables) ---

function label = get_mean_latex_label_internal(raw_name, val, g_owner)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    core_name = regexprep(raw_name, '(MG\d_|Mean_)', '', 'ignorecase');
    if contains(core_name, 'SoC'), sym = 'SoC'; val_fmt = '%.1f\\%%'; val = val*100;
    elseif contains(core_name, 'tank'), sym = 'V_{T}'; val_fmt='%.0f L';
    elseif contains(core_name, 'P_dem'), sym = 'P_{L}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'P_gen'), sym = 'P_{G}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'Q_dem'), sym = 'Q_{L}'; val_fmt='%.2f L/s';
    elseif contains(core_name, 'aq'), sym = 'EAW'; val_fmt='%.0f L'; g_owner=4;
    else, sym = 'X'; val_fmt='%.2f'; end
    f_sym = sym; if is_mean, f_sym = sprintf('\\bar{%s}', sym); end
    val_s = sprintf(val_fmt, val);
    if g_owner < 4, label = sprintf('$%s^{%d}$ (%s)', f_sym, g_owner, val_s);
    else, label = sprintf('$%s$ (%s)', f_sym, val_s); end
end

function render_ranking_tile(weights, errors, labels, is_others, title_txt, d, h, m, val, mg, sym_tex, flow_s)
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
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', 'FontSize',9);
    xlabel('Average Influence', 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','bold');
    
    % Títulos de 2 líneas con sintaxis corregida
    l1 = title_txt;
    l2 = sprintf('MG%d | Day %d, %02d:%02d | $%s^{%d} = %.2f$ L/s (%s)', ...
                 mg, d, h, m, sym_tex, mg, val, flow_s);
    title({l1; l2}, 'FontName','Times New Roman', 'FontSize',10, 'Interpreter','latex');
    grid on; xline(0,'k-');
    xlim_val = max(abs(weights))*1.3; if xlim_val < 0.1, xlim_val = 0.1; end
    xlim([-xlim_val xlim_val]);
end