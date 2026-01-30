%% --- File: plot_lime_results_3mg_MEAN_v3.m ---
%
% VISUALIZATION SCRIPT FOR LIME (MEAN/AVERAGE VERSION)
% Supports: AR and TS models.
% FIX v3: Generates BOTH Q_s (Exchange) and Q_p (Pumping) plots 
%         for MG1 in the Altruismo scenario.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
TIPO_MODELO = 'AR'; % <--- CAMBIA ESTO A 'AR' O 'TS'
scenarios_list = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
targets_list = [1, 2, 3];

% Colors for Interaction Plot
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray

% Output Directories
dir_paper = sprintf('figures_paper_mean_%s', TIPO_MODELO);
dir_pres  = sprintf('figures_presentation_mean_%s', TIPO_MODELO);

if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING LIME PLOTS (%s MODEL) ---\n', TIPO_MODELO);

for s_idx = 1:length(scenarios_list)
    scn_name = scenarios_list{s_idx};
    
    switch scn_name
        case 'GlobalPeak', s_title = 'Scenario A: Global Peak Interaction';
        case 'Altruismo', s_title = 'Scenario B: Active Water Export';
        case 'DirectSatisfaction', s_title = 'Scenario C: Direct Demand Satisfaction';
        otherwise, s_title = scn_name;
    end
    
    for t_idx = targets_list
        
        % --- 1. CONSTRUCCIÓN DE NOMBRES DE ARCHIVO ---
        f_scn = sprintf('lime_Scenario_%s_%s_MG%d_MEAN.mat', scn_name, TIPO_MODELO, t_idx);
        f_pump = sprintf('lime_PUMP_%s_%s_MG%d_MEAN.mat', scn_name, TIPO_MODELO, t_idx);
        
        % --- 2. LISTA DE TAREAS (JOB QUEUE) ---
        % En lugar de elegir uno, creamos una lista de archivos a procesar
        files_to_process = {};
        is_pump_flags = [];
        
        % A) Siempre intentar procesar el Intercambio (Q_s) si existe
        if exist(f_scn, 'file')
            files_to_process{end+1} = f_scn;
            is_pump_flags(end+1) = false;
        end
        
        % B) Si es Altruismo MG1, TAMBIÉN procesar el Bombeo (Q_p) si existe
        if strcmp(scn_name, 'Altruismo') && t_idx == 1 && exist(f_pump, 'file')
            files_to_process{end+1} = f_pump;
            is_pump_flags(end+1) = true;
        end
        
        if isempty(files_to_process)
            continue; 
        end
        
        % --- 3. PROCESAR CADA ARCHIVO EN LA LISTA ---
        for k = 1:length(files_to_process)
            filename = files_to_process{k};
            is_pump_analysis = is_pump_flags(k);
            
            fprintf('  > Processing: %s...\n', filename);
            data = load(filename);
            
            all_explanations = data.all_explanations;
            feature_names = data.feature_names;
            X_values = data.estado.X_original;
            
            % Recuperar K_TARGET
            if isfield(data, 'K_TARGET'), K_TARGET = data.K_TARGET;
            elseif isfield(data, 'k_target'), K_TARGET = data.k_target;
            else, K_TARGET = 1; warning('K_TARGET missing, using 1.'); end
            
            % Información de Tiempo
            Ts_sim = 60; 
            t_seconds = (K_TARGET - 1) * Ts_sim;
            day_num = floor(t_seconds / 86400) + 1;
            rem_seconds = mod(t_seconds, 86400);
            hour_val = floor(rem_seconds / 3600);
            min_val = round((rem_seconds - hour_val*3600) / 60);
            
            % Obtener Valor Real (Y)
            if isfield(data.estado, 'Y_target_real_vector')
                real_val = data.estado.Y_target_real_vector(t_idx);
            else
                real_val = 0; 
            end
            
            % Procesamiento de Pesos (Promedio)
            num_runs = length(all_explanations);
            N_features = length(feature_names);
            weights_matrix = zeros(N_features, num_runs);
            
            for i = 1:num_runs
                run_data = all_explanations{i};
                map_temp = containers.Map(run_data(:,1), [run_data{:,2}]);
                for j = 1:N_features
                    weights_matrix(j, i) = map_temp(feature_names{j});
                end
            end
            avg_weights = mean(weights_matrix, 2);
            std_weights = std(weights_matrix, 0, 2);
            
            % --- GENERAR ETIQUETAS Y GRUPOS ---
            plot_labels = cell(N_features, 1);
            groups_vec = zeros(N_features, 1); 
            
            for i = 1:N_features
                raw_name = feature_names{i};
                val = X_values(i);
                
                % Identificar Dueño (Agente)
                g_owner = 4; 
                if contains(raw_name, 'MG1', 'IgnoreCase', true), g_owner=1;
                elseif contains(raw_name, 'MG2', 'IgnoreCase', true), g_owner=2;
                elseif contains(raw_name, 'MG3', 'IgnoreCase', true), g_owner=3;
                end
                if contains(raw_name, 'aq', 'IgnoreCase', true), g_owner = 4; end
                
                groups_vec(i) = g_owner; 
                plot_labels{i} = get_mean_latex_label(raw_name, val, g_owner);
            end
            
            % Ordenar por importancia
            [sorted_w, sort_idx] = sort(abs(avg_weights), 'descend');
            sorted_labels = plot_labels(sort_idx);
            sorted_real_w = avg_weights(sort_idx);
            sorted_std = std_weights(sort_idx);
            
            % Agregación por Agente
            influence_per_agent = zeros(1, 4);
            for g = 1:4
                idx_group = (groups_vec == g);
                influence_per_agent(g) = sum(abs(avg_weights(idx_group)));
            end
            total_infl = sum(influence_per_agent);
            if total_infl < 1e-9, total_infl = 1; end
            influence_pct = (influence_per_agent / total_infl) * 100;
            
            % --- DEFINICIÓN DE TÍTULOS Y SUFIJOS DE ARCHIVO ---
            if is_pump_analysis
                target_str = sprintf('$Q_{p}^{%d}$', t_idx);
                s_title_plot = [s_title ' (Pumping)'];
                file_suffix = '_PUMP'; % Sufijo para diferenciar el archivo PDF
            else
                target_str = sprintf('$Q_{s}^{%d}$', t_idx);
                s_title_plot = s_title;
                file_suffix = '';      % Sin sufijo para el estándar (Qs)
            end
    
            % 1. RANKING PLOT
            fname_rank_pap = fullfile(dir_paper, sprintf('Ranking_%s_%s_MG%d%s_Paper', scn_name, TIPO_MODELO, t_idx, file_suffix));
            create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, s_title_plot, ...
                day_num, hour_val, min_val, real_val, t_idx, 'paper', fname_rank_pap, target_str);
                
            % 2. INTERACTION PLOT
            agent_colors = [color_mg1; color_mg2; color_mg3; color_aq];
            fname_int_pap = fullfile(dir_paper, sprintf('Interaction_%s_%s_MG%d%s_Paper', scn_name, TIPO_MODELO, t_idx, file_suffix));
            create_interaction_plot(influence_pct, agent_colors, s_title_plot, ...
                day_num, hour_val, min_val, real_val, t_idx, 'paper', fname_int_pap, target_str);
        end    
    end
end
fprintf('--- ALL MEAN PLOTS EXPORTED SUCCESSFULLY FOR %s ---\n', TIPO_MODELO);


%% --- HELPER: LABEL PARSER (ROBUST + LATEX FIX) ---
function label = get_mean_latex_label(raw_name, val, g_owner)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    core_name = regexprep(raw_name, 'MG\d_', '', 'ignorecase');
    core_name = regexprep(core_name, 'Mean_', '', 'ignorecase');
    
    if contains(core_name, 'SoC', 'IgnoreCase', true)
        sym = 'SoC'; val_fmt = '%.1f\\%%'; val = val * 100;
    elseif contains(core_name, 'tank', 'IgnoreCase', true) || contains(core_name, 'Estanque', 'IgnoreCase', true)
        sym = 'V_{Tank}'; val_fmt='%.0f L';
    elseif contains(core_name, 'P_dem', 'IgnoreCase', true)
        sym = 'P_{L}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'P_gen', 'IgnoreCase', true)
        sym = 'P_{G}'; val_fmt='%.1f kW';
    elseif contains(core_name, 'Q_dem', 'IgnoreCase', true)
        sym = 'Q_{L}'; val_fmt='%.2f L/s';
    elseif contains(core_name, 'aq', 'IgnoreCase', true)
        sym = 'EAW'; val_fmt='%.0f L'; g_owner=4;
    else
        sym = 'X'; val_fmt='%.2f'; 
    end
    
    if is_mean, final_sym = sprintf('\\bar{%s}', sym); else, final_sym = sym; end
    val_str = sprintf(val_fmt, val);
    
    if g_owner < 4
        label = sprintf('$%s^{%d}$ (%s)', final_sym, g_owner, val_str);
    else
        label = sprintf('$%s$ (%s)', final_sym, val_str);
    end
end


%% --- PLOT 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_text, d, h, m, val, mg, mode, fname, target_sym)
    N = length(weights);
    fig_w=7; fig_h=6; font_ax=10; font_t=11; bar_w=0.6;
    pos_ax = [0.35 0.12 0.60 0.78];
    
    fig = figure('Units','inches','Position',[0 0 fig_w fig_h],'Visible','off','Color','w');
    
    colors = zeros(N,3);
    for i=1:N
        if weights(i)>=0, colors(i,:) = [0.466 0.674 0.188];
        else, colors(i,:) = [0.635 0.078 0.184]; end
    end
    
    barh(weights, bar_w, 'FaceColor','flat', 'CData', colors); hold on;
    errorbar(weights, 1:N, errors, 'horizontal', 'k', 'LineStyle','none');
    
    ax = gca;
    set(ax, 'YTick', 1:N, 'YTickLabel', labels, 'YDir','reverse', ...
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', 'FontSize',font_ax);
    
    xlabel('Average Influence (Mean Features)', 'FontName','Times New Roman', 'FontSize',font_ax, 'FontWeight','bold');
    
    full_title = {['LIME Analysis: ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', ...
                  mg, d, h, m, target_sym, val)};
              
    title(full_title, 'FontName','Times New Roman', 'FontSize',font_t, 'Interpreter','latex');
    
    xlim_val = max(abs(weights))*1.2; if xlim_val<1e-6, xlim_val=1; end
    xlim([-xlim_val, xlim_val]); xline(0,'k-'); grid on;
    
    ax.Position = pos_ax;
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end


%% --- PLOT 2: INTERACTION ---
function create_interaction_plot(pct, cmap, title_text, d, h, m, val, mg, mode, fname, target_sym)
    fig_w=6; fig_h=4.5; font_ax=10; font_t=11;
    fig = figure('Units','inches','Position',[0 0 fig_w fig_h],'Visible','off','Color','w');
    
    b = bar(1:4, pct, 0.6, 'FaceColor','flat'); b.CData = cmap;
    
    ylabel('Relative Total Influence [%]', 'FontName','Times New Roman','FontSize',font_ax,'FontWeight','bold');
    xticks(1:4); 
    xticklabels({'Microgrid 1','Microgrid 2','Microgrid 3','Aquifer'});
    ylim([0 100]); grid on;
    
    set(gca, 'FontName','Times New Roman','FontSize',font_ax);
    
    for i=1:4
        text(i, pct(i)+2, sprintf('%.1f%%', pct(i)), 'HorizontalAlignment','center', ...
            'FontName','Times New Roman','FontSize',font_ax,'FontWeight','bold');
    end
    
    full_title = {['Interaction: ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', mg, d, h, m, target_sym, val)};
              
    title(full_title, 'FontName','Times New Roman','FontSize',font_t, 'Interpreter','latex');
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end