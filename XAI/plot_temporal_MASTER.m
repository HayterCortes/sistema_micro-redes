%% --- File: plot_temporal_MASTER.m ---
%
% SCRIPT MAESTRO DE VISUALIZACIÓN TEMPORAL (Todas las combinaciones)
%
% Barre automáticamente la estructura de carpetas 'temporal_results'
% y genera:
% 1. Gráficos Individuales: Área, Correlación y Evolución de Pesos.
% 2. Gráficos Comparativos Globales R^2: Evolución de Fidelidad promedio.
% 3. Gráficos Comparativos Globales RBO: Evolución de Estabilidad promedio.
%    (Comparando Mean/Std/AE x Gauss/Pareto en 1 sola figura por modelo).
%
% FIX: Notación matemática rigurosa para \max y \sigma.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURACIÓN GLOBAL ---
MODELS = {'AR', 'TS'};
TARGETS = [1, 2, 3];
PERTURBATIONS = {'GAUSSIAN', 'PARETO'};
CASES = [1, 2, 3];

base_in  = 'temporal_results';
dirs_ex_in = {fullfile(base_in, '16_features_exchange'), fullfile(base_in, '34_features_exchange'), fullfile(base_in, '34_features_AE_exchange')};
dirs_pp_in = {fullfile(base_in, 'pumping', '16_features_pump'), fullfile(base_in, 'pumping', '34_features_pump'), fullfile(base_in, 'pumping', '34_features_AE_pump')};

case_tags = {'MEAN', 'STANDARD', 'AE_MOMENTS'};

base_out = 'figures_Temporal';

% Colors for Area Plot (Agent Colors)
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray
colors_map = [color_mg1; color_mg2; color_mg3; color_aq];

fprintf('--- INICIANDO GENERACIÓN MASIVA DE GRÁFICOS TEMPORALES ---\n');

if ~exist(base_out, 'dir'), mkdir(base_out); end

% =========================================================================
% PARTE 1: GRÁFICOS INDIVIDUALES POR AGENTE Y COMBINACIÓN
% =========================================================================
for m_idx = 1:length(MODELS)
    curr_model = MODELS{m_idx};
    
    for c_idx = CASES
        curr_case = case_tags{c_idx};
        
        for p_idx = 1:length(PERTURBATIONS)
            curr_pert = PERTURBATIONS{p_idx};
            
            % Crear carpeta de salida específica
            dir_out = fullfile(base_out, curr_model, curr_case, curr_pert);
            if ~exist(dir_out, 'dir'), mkdir(dir_out); end
            
            fprintf('\n>>> Procesando Visualización Individual: %s | %s | %s <<<\n', curr_model, curr_case, curr_pert);
            
            for t_idx = TARGETS
                
                % -----------------------------------------------------
                % 1. PROCESAR INTERCAMBIO (Q_t)
                % -----------------------------------------------------
                fname_ex = fullfile(dirs_ex_in{c_idx}, sprintf('lime_temporal_EXCHANGE_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_idx, curr_case, curr_pert));
                if isfile(fname_ex)
                    load(fname_ex, 'temporal_results');
                    target_sym = sprintf('$Q_{t}^{%d}$', t_idx);
                    prefix_out = sprintf('EXCHANGE_MG%d', t_idx);
                    generate_temporal_plots(temporal_results, colors_map, t_idx, target_sym, dir_out, prefix_out);
                end
                
                % -----------------------------------------------------
                % 2. PROCESAR BOMBEO (Q_p)
                % -----------------------------------------------------
                fname_pp = fullfile(dirs_pp_in{c_idx}, sprintf('lime_temporal_PUMPING_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_idx, curr_case, curr_pert));
                if isfile(fname_pp)
                    load(fname_pp, 'temporal_results');
                    target_sym = sprintf('$Q_{p}^{%d}$', t_idx);
                    prefix_out = sprintf('PUMPING_MG%d', t_idx);
                    generate_temporal_plots(temporal_results, colors_map, t_idx, target_sym, dir_out, prefix_out);
                end
                
            end
        end
    end
end

% =========================================================================
% PARTE 2: GRÁFICOS COMPARATIVOS GLOBALES DE R^2 y RBO
% =========================================================================
fprintf('\n--- GENERANDO GRÁFICOS COMPARATIVOS GLOBALES (R^2 y RBO) ---\n');
generate_comparative_r2_plots(MODELS, case_tags, PERTURBATIONS, dirs_ex_in, dirs_pp_in, base_out);
generate_comparative_rbo_plots(MODELS, case_tags, PERTURBATIONS, dirs_ex_in, dirs_pp_in, base_out);

fprintf('\n=== GENERACIÓN DE FIGURAS COMPLETADA ===\n');
fprintf('Revisa la carpeta "%s".\n', base_out);


%% ========================================================================
%  FUNCIONES DE PLOTEO PRINCIPALES
%  ========================================================================

function generate_comparative_r2_plots(MODELS, case_tags, PERTURBATIONS, dirs_ex_in, dirs_pp_in, base_out)
    % Colores y Estilos para las 6 curvas
    % MEAN = Azul, STANDARD = Naranja, AE_MOMENTS = Verde
    curve_colors = [0 0.4470 0.7410;  
                    0.8500 0.3250 0.0980; 
                    0.4660 0.6740 0.1880];
    % GAUSSIAN = Línea continua, PARETO = Línea Punteada
    curve_styles = {'-', '--'};
    
    target_names = {'Water Exchange ($Q_{t}$)', 'Pumping ($Q_{p}$)'};
    
    for m_idx = 1:length(MODELS)
        curr_model = MODELS{m_idx};
        
        fig = figure('Units', 'inches', 'Position', [0, 0, 10, 8], 'Visible', 'off', 'Color', 'w');
        t = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        for tgt_idx = 1:2 % 1 = Exchange, 2 = Pumping
            nexttile;
            hold on;
            
            legend_lines = gobjects(0);
            legend_labels = {};
            t_days = [];
            
            % Recorrer las 6 combinaciones
            for c_idx = 1:3
                for p_idx = 1:2
                    
                    avg_r2_curve = [];
                    valid_mgs = 0;
                    
                    % Promediar sobre MG1, MG2, MG3
                    for t_mg = 1:3
                        if tgt_idx == 1
                            f_name = sprintf('lime_temporal_EXCHANGE_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_mg, case_tags{c_idx}, PERTURBATIONS{p_idx});
                            full_path = fullfile(dirs_ex_in{c_idx}, f_name);
                        else
                            f_name = sprintf('lime_temporal_PUMPING_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_mg, case_tags{c_idx}, PERTURBATIONS{p_idx});
                            full_path = fullfile(dirs_pp_in{c_idx}, f_name);
                        end
                        
                        if isfile(full_path)
                            data = load(full_path, 'temporal_results');
                            if isempty(t_days), t_days = data.temporal_results.time_days + 1; end
                            
                            % Evitar NaNs al sumar
                            r2_hist = data.temporal_results.quality_history;
                            r2_hist(isnan(r2_hist)) = 0; 
                            
                            if isempty(avg_r2_curve)
                                avg_r2_curve = r2_hist;
                            else
                                avg_r2_curve = avg_r2_curve + r2_hist;
                            end
                            valid_mgs = valid_mgs + 1;
                        end
                    end
                    
                    % Si hay datos, graficar la curva
                    if valid_mgs > 0
                        avg_r2_curve = avg_r2_curve / valid_mgs;
                        
                        % Formatear etiqueta de la leyenda
                        lbl_case = case_tags{c_idx};
                        lbl_case = strrep(lbl_case, 'AE_MOMENTS', 'AE');
                        lbl_pert = PERTURBATIONS{p_idx};
                        lbl_pert = strrep(lbl_pert, 'GAUSSIAN', 'Gaussian');
                        lbl_pert = strrep(lbl_pert, 'PARETO', 'Pareto');
                        final_label = sprintf('%s-%s', lbl_case, lbl_pert);
                        
                        h = plot(t_days, avg_r2_curve, 'Color', curve_colors(c_idx,:), ...
                                 'LineStyle', curve_styles{p_idx}, 'LineWidth', 2.0);
                             
                        legend_lines(end+1) = h;
                        legend_labels{end+1} = final_label;
                    end
                end
            end
            
            % Formato del Subgráfico
            title(sprintf('Explanability Fidelity ($R^2$) for %s', target_names{tgt_idx}), 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('Average $R^2$ (MG1, MG2, MG3)', 'FontName', 'Times New Roman', 'FontSize', 11, 'Interpreter', 'latex');
            
            if tgt_idx == 2
                xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            xlim([1 8]); ylim([0 1.05]); xticks(1:7);
            yline(1.0, 'k-', 'Alpha', 0.3);
            for d = 2:7, xline(d, 'k:', 'Alpha', 0.2); end
            
            set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
            grid on; box on;
            
            % Poner leyenda solo en el gráfico superior para ahorrar espacio
            if tgt_idx == 1
                legend(legend_lines, legend_labels, 'Location', 'eastoutside', 'FontSize', 10, 'Interpreter', 'none');
            end
        end
        
        title(t, sprintf('Global Temporal Fidelity Comparison - %s Model', curr_model), 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
        
        f_out = fullfile(base_out, sprintf('Comparative_R2_Evolution_%s', curr_model));
        exportgraphics(fig, [f_out '.pdf'], 'ContentType', 'vector');
        close(fig);
    end
end

function generate_comparative_rbo_plots(MODELS, case_tags, PERTURBATIONS, dirs_ex_in, dirs_pp_in, base_out)
    % Colores y Estilos para las 6 curvas
    curve_colors = [0 0.4470 0.7410;  
                    0.8500 0.3250 0.0980; 
                    0.4660 0.6740 0.1880];
    curve_styles = {'-', '--'};
    
    target_names = {'Water Exchange ($Q_{t}$)', 'Pumping ($Q_{p}$)'};
    
    for m_idx = 1:length(MODELS)
        curr_model = MODELS{m_idx};
        
        fig = figure('Units', 'inches', 'Position', [0, 0, 10, 8], 'Visible', 'off', 'Color', 'w');
        t = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        for tgt_idx = 1:2 % 1 = Exchange, 2 = Pumping
            nexttile;
            hold on;
            
            legend_lines = gobjects(0);
            legend_labels = {};
            t_days = [];
            
            % Recorrer las 6 combinaciones
            for c_idx = 1:3
                for p_idx = 1:2
                    
                    avg_rbo_curve = [];
                    valid_mgs = 0;
                    
                    % Promediar sobre MG1, MG2, MG3
                    for t_mg = 1:3
                        if tgt_idx == 1
                            f_name = sprintf('lime_temporal_EXCHANGE_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_mg, case_tags{c_idx}, PERTURBATIONS{p_idx});
                            full_path = fullfile(dirs_ex_in{c_idx}, f_name);
                        else
                            f_name = sprintf('lime_temporal_PUMPING_%s_MG%d_7days_%s_RAW_RBO_%s.mat', curr_model, t_mg, case_tags{c_idx}, PERTURBATIONS{p_idx});
                            full_path = fullfile(dirs_pp_in{c_idx}, f_name);
                        end
                        
                        if isfile(full_path)
                            data = load(full_path, 'temporal_results');
                            if isempty(t_days), t_days = data.temporal_results.time_days + 1; end
                            
                            % Extraer RBO en lugar de R^2
                            rbo_hist = data.temporal_results.rbo_history;
                            rbo_hist(isnan(rbo_hist)) = 0; 
                            
                            if isempty(avg_rbo_curve)
                                avg_rbo_curve = rbo_hist;
                            else
                                avg_rbo_curve = avg_rbo_curve + rbo_hist;
                            end
                            valid_mgs = valid_mgs + 1;
                        end
                    end
                    
                    % Si hay datos, graficar la curva
                    if valid_mgs > 0
                        avg_rbo_curve = avg_rbo_curve / valid_mgs;
                        
                        lbl_case = case_tags{c_idx};
                        lbl_case = strrep(lbl_case, 'AE_MOMENTS', 'AE');
                        lbl_pert = PERTURBATIONS{p_idx};
                        lbl_pert = strrep(lbl_pert, 'GAUSSIAN', 'Gaussian');
                        lbl_pert = strrep(lbl_pert, 'PARETO', 'Pareto');
                        final_label = sprintf('%s-%s', lbl_case, lbl_pert);
                        
                        h = plot(t_days, avg_rbo_curve, 'Color', curve_colors(c_idx,:), ...
                                 'LineStyle', curve_styles{p_idx}, 'LineWidth', 2.0);
                             
                        legend_lines(end+1) = h;
                        legend_labels{end+1} = final_label;
                    end
                end
            end
            
            % Formato del Subgráfico
            title(sprintf('Explanability Stability (RBO) for %s', target_names{tgt_idx}), 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('Average RBO (MG1, MG2, MG3)', 'FontName', 'Times New Roman', 'FontSize', 11, 'Interpreter', 'latex');
            
            if tgt_idx == 2
                xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            xlim([1 8]); ylim([0 1.05]); xticks(1:7);
            yline(1.0, 'k-', 'Alpha', 0.3);
            for d = 2:7, xline(d, 'k:', 'Alpha', 0.2); end
            
            set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
            grid on; box on;
            
            % Poner leyenda solo en el gráfico superior para ahorrar espacio
            if tgt_idx == 1
                legend(legend_lines, legend_labels, 'Location', 'eastoutside', 'FontSize', 10, 'Interpreter', 'none');
            end
        end
        
        title(t, sprintf('Global Temporal Stability Comparison - %s Model', curr_model), 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
        
        f_out = fullfile(base_out, sprintf('Comparative_RBO_Evolution_%s', curr_model));
        exportgraphics(fig, [f_out '.pdf'], 'ContentType', 'vector');
        close(fig);
    end
end

function generate_temporal_plots(temporal_results, colors_map, t_idx, target_sym, dir_out, prefix_out)
    
    t_days = temporal_results.time_days + 1; % Shift 0-7 -> 1-8
    weights_hist = temporal_results.weights_history; 
    target_real = temporal_results.target_real_history;
    feature_names = temporal_results.feature_names;
    [N_feats, N_steps] = size(weights_hist);
    
    % --- 1. PREPARAR DATOS PARA ÁREA (Agrupación por Agentes) ---
    influence_groups = zeros(4, N_steps); 
    for i = 1:N_feats
        name = feature_names{i};
        if contains(name, 'MG1', 'IgnoreCase', true), g=1;
        elseif contains(name, 'MG2', 'IgnoreCase', true), g=2;
        elseif contains(name, 'MG3', 'IgnoreCase', true), g=3;
        else, g=4; end
        influence_groups(g, :) = influence_groups(g, :) + abs(weights_hist(i, :));
    end
    total_infl = sum(influence_groups, 1);
    total_infl(total_infl < 1e-9) = 1; 
    influence_pct = (influence_groups ./ total_infl) * 100;
    
    % --- 2. PREPARAR DATOS DE CORRELACIÓN (Influencia Vecinos) ---
    mask_neighbors = true(1, 3); mask_neighbors(t_idx) = false;
    vecinos_infl = sum(influence_pct(find(mask_neighbors), :), 1);
    
    % --- 3. PREPARAR DATOS EVOLUCIÓN DE PESOS (Top 8 Absoluto Histórico) ---
    total_abs_influence = sum(abs(weights_hist), 2);
    [~, sorted_idx] = sort(total_abs_influence, 'descend');
    top_k = min(8, N_feats);
    top_indices = sorted_idx(1:top_k);
    
    top_weights_hist = weights_hist(top_indices, :);
    top_raw_names = feature_names(top_indices);
    
    top_labels = cell(top_k, 1);
    for k = 1:top_k
        top_labels{k} = get_latex_label(top_raw_names{k});
    end
    
    % --- 4. GENERAR LOS 3 GRÁFICOS ---
    f_area = fullfile(dir_out, sprintf('%s_Area', prefix_out));
    create_area_plot(t_days, influence_pct, colors_map, target_sym, f_area);
    
    f_corr = fullfile(dir_out, sprintf('%s_Corr', prefix_out));
    create_corr_plot(t_days, target_real, vecinos_infl, target_sym, f_corr);
    
    f_weight = fullfile(dir_out, sprintf('%s_Weights', prefix_out));
    create_weight_evolution_plot(t_days, top_weights_hist, top_labels, target_sym, f_weight);
end


%% --- HELPER: LABEL PARSER (RIGUROSIDAD MATEMÁTICA) ---
function label = get_latex_label(raw_name)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    is_max  = contains(raw_name, 'Max_', 'IgnoreCase', true);
    is_std  = contains(raw_name, 'Std_', 'IgnoreCase', true);
    
    core_name = regexprep(raw_name, '(Mean_|Max_|Std_|MG\d_)', '', 'ignorecase');
    
    g = 4; 
    if contains(raw_name, 'MG1', 'IgnoreCase', true), g=1;
    elseif contains(raw_name, 'MG2', 'IgnoreCase', true), g=2;
    elseif contains(raw_name, 'MG3', 'IgnoreCase', true), g=3;
    end
    
    if contains(core_name, 'SoC', 'IgnoreCase', true)
        sym = 'SoC';
    elseif contains(core_name, 'tank', 'IgnoreCase', true) || contains(core_name, 'Estanque', 'IgnoreCase', true)
        sym = 'V_{Tank}';
    elseif contains(core_name, 'P_dem', 'IgnoreCase', true)
        sym = 'P_{L}';
    elseif contains(core_name, 'P_gen', 'IgnoreCase', true)
        sym = 'P_{G}';
    elseif contains(core_name, 'Q_dem', 'IgnoreCase', true)
        sym = 'Q_{L}';
    elseif contains(core_name, 'aq', 'IgnoreCase', true)
        sym = 'EAW'; g=4;
    else
        core = regexprep(core_name, 'MG\d_', ''); 
        sym = strrep(core, '_', '\_');
    end
    
    if is_mean
        final_sym = sprintf('\\bar{%s}', sym);
    elseif is_max
        final_sym = sprintf('{%s}_{\\max}', sym);
    elseif is_std
        final_sym = sprintf('\\sigma(%s)', sym);
    else
        final_sym = sym;
    end
    
    if g < 4
        label = sprintf('$%s^{%d}$', final_sym, g);
    else
        label = sprintf('$%s$', final_sym);
    end
end


%% --- PLOT 1: STACKED AREA ---
function create_area_plot(t, data_pct, cmap, target_sym, filename)
    fig = figure('Units', 'inches', 'Position', [0, 0, 7, 4.5], 'Visible', 'off', 'Color', 'w');
    
    h = area(t, data_pct');
    h(1).FaceColor = cmap(1,:); h(1).DisplayName = 'Microgrid 1';
    h(2).FaceColor = cmap(2,:); h(2).DisplayName = 'Microgrid 2';
    h(3).FaceColor = cmap(3,:); h(3).DisplayName = 'Microgrid 3';
    h(4).FaceColor = cmap(4,:); h(4).DisplayName = 'Aquifer';
    
    title(sprintf('Evolution of Cooperative Dependency (%s)', target_sym), 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'latex');
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel('Relative Total Influence [%]', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    xlim([1 8]); ylim([0 100]); xticks(1:7);
    
    ax = gca; set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, 'Layer', 'top');
    grid on; box on;
    for d = 2:7, xline(d, 'k--', 'Alpha', 0.4); end
    legend(h, 'Location', 'eastoutside', 'Interpreter', 'latex', 'FontSize', 10);
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); close(fig);
end


%% --- PLOT 2: CORRELATION ---
function create_corr_plot(t, real_val, neigh_infl, target_sym, filename)
    fig = figure('Units', 'inches', 'Position', [0, 0, 7, 5], 'Visible', 'off', 'Color', 'w');
    
    is_pumping = contains(target_sym, 'p');
    
    yyaxis left
    plot(t, real_val, 'b-', 'LineWidth', 1.2);
    
    if is_pumping
        ylabel(sprintf('Actual Pump Flow %s [L/s]', target_sym), 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
    else
        ylabel(sprintf('Actual Water Exchange %s [L/s]', target_sym), 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
    end
    
    ax = gca; ax.YColor = [0 0 0.8];
    yline(0, 'k-', 'LineWidth', 1.0, 'Alpha', 0.5);
    
    if ~is_pumping
        yl = ylim; range_y = yl(2)-yl(1);
        text(1.2, yl(2)-range_y*0.05, 'EXPORT', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 8, 'FontWeight', 'bold');
        text(1.2, yl(1)+range_y*0.05, 'IMPORT', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 8, 'FontWeight', 'bold');
    end
    
    yyaxis right
    plot(t, neigh_infl, '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.2);
    ylabel('Neighbor Influence [%] (LIME)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    ax.YColor = [0.8500 0.3250 0.0980]; ylim([0 100]);
    
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    title(sprintf('Physical Flow vs. Algorithmic Detection (%s)', target_sym), 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'latex');
    xlim([1 8]); xticks(1:7);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10); grid on; box on;
    for d = 2:7, xline(d, 'k:', 'Alpha', 0.3); end
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); close(fig);
end


%% --- PLOT 3: WEIGHTS EVOLUTION ---
function create_weight_evolution_plot(t, weights, labels, target_sym, filename)
    fig = figure('Units', 'inches', 'Position', [0, 0, 7, 5], 'Visible', 'off', 'Color', 'w');
    
    colors = lines(length(labels));
    hold on;
    h_lines = gobjects(length(labels), 1); 
    
    for k = 1:size(weights, 1)
        h_lines(k) = plot(t, weights(k, :), 'LineStyle', '-', 'Color', colors(k,:), 'LineWidth', 1.5);
    end
    
    title(sprintf('Temporal Evolution of LIME Weights (%s)', target_sym), 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'latex');
    xlabel('Time [Days]', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel('Average Influence Weight', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
    
    xlim([1 8]); xticks(1:7);
    yline(0, 'k-', 'LineWidth', 1.0, 'Alpha', 0.5); 
    
    ax = gca;
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10);
    grid on; box on;
    for d = 2:7, xline(d, 'k:', 'Alpha', 0.3); end
    
    legend(h_lines, labels, 'Location', 'eastoutside', 'Interpreter', 'latex', 'FontSize', 10);
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector'); close(fig);
end