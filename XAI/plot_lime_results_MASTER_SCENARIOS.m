%% --- File: plot_lime_results_MASTER_SCENARIOS.m ---
%
% SCRIPT MAESTRO DE VISUALIZACIÓN DE ESCENARIOS (Q_t y Q_p)
% Soporta: 3 Casos x 2 Perturbaciones x 2 Modelos.
%
% ESTRUCTURA DE ENTRADA (Lectura de .mat):
%   - Exchange: 16_features_exchange, 34_features_exchange, 34_features_AE_exchange
%   - Pumping: pumping/16_features_pump, pumping/34_features_pump, pumping/34_features_AE_pump
%
% ESTRUCTURA DE SALIDA (Guardado de .pdf):
%   figures_Scenarios / Case_X / Perturbación / Modelo
%
% FIX 1: Solucionado el error de sintaxis LaTeX en los títulos.
% FIX 2: Cambio de notación para valores máximos a X_{max}.
% FIX 3: Extracción del valor real (Ground Truth) de la simulación original.
% FIX 4: Reconstrucción en vivo de X_values (Features) para evitar 0.00
%--------------------------------------------------------------------------
clear; clc; close all;

% --- 1. CONFIGURACIÓN GLOBAL ---
SCENARIOS_LIST = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
TARGETS_LIST = [1, 2, 3];

% Definición de los 3 Casos de Estudio (Incluyendo directorios de entrada)
% ---------------------------------------------------------------------
% CASO 1: 16 Features
CASES(1).name = 'Case_1_16_Features_Mean';   
CASES(1).suffix_ex = 'MEAN_RBO';     
CASES(1).suffix_pp = 'MEAN_RBO';
CASES(1).dir_ex = '16_features_exchange'; 
CASES(1).dir_pp = fullfile('pumping', '16_features_pump');

% CASO 2: 34 Features Standard
CASES(2).name = 'Case_2_34_Features_Std';    
CASES(2).suffix_ex = 'STANDARD_RBO'; 
CASES(2).suffix_pp = 'STANDARD_RBO';
CASES(2).dir_ex = '34_features_exchange'; 
CASES(2).dir_pp = fullfile('pumping', '34_features_pump');

% CASO 3: 34 Features + AE Moments
CASES(3).name = 'Case_3_34_Features_AE';     
CASES(3).suffix_ex = 'MOMENTS_RBO';  
CASES(3).suffix_pp = 'MOMENTS_AE'; 
CASES(3).dir_ex = '34_features_AE_exchange'; 
CASES(3).dir_pp = fullfile('pumping', '34_features_AE_pump');
% ---------------------------------------------------------------------

PERTURBATIONS = {'PARETO', 'GAUSSIAN'};
MODELS = {'AR', 'TS'};

% Colores
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray
agent_colors = [color_mg1; color_mg2; color_mg3; color_aq];

fprintf('--- INICIANDO GENERACIÓN MASIVA DE FIGURAS ---\n');

% --- 2. BUCLE MAESTRO ---
for c_idx = 1:length(CASES)
    curr_case = CASES(c_idx);
    
    for p_idx = 1:length(PERTURBATIONS)
        curr_pert = PERTURBATIONS{p_idx};
        
        for m_idx = 1:length(MODELS)
            curr_model = MODELS{m_idx};
            
            % Cargar los datos reales (Ground Truth) de la simulación original
            gt_file = fullfile('..', 'results_mpc', sprintf('resultados_mpc_%s_3mg_7dias.mat', curr_model));
            if isfile(gt_file)
                gt_data = load(gt_file);
            else
                error('No se pudo encontrar el archivo original de resultados: %s', gt_file);
            end
            
            % Crear Estructura de Directorios de Salida
            base_dir = fullfile('figures_Scenarios', curr_case.name, curr_pert, curr_model);
            if ~exist(base_dir, 'dir'), mkdir(base_dir); end
            
            fprintf('\n>>> Procesando: %s | %s | %s\n', curr_case.name, curr_pert, curr_model);
            
            for s_idx = 1:length(SCENARIOS_LIST)
                scn_name = SCENARIOS_LIST{s_idx};
                
                % Títulos bonitos para gráficos
                switch scn_name
                    case 'GlobalPeak', s_title = 'Scenario A: Global Peak';
                    case 'Altruismo', s_title = 'Scenario B: Altruism';
                    case 'DirectSatisfaction', s_title = 'Scenario C: Direct Satisfaction';
                    otherwise, s_title = scn_name;
                end
                
                for t_idx = TARGETS_LIST
                    
                    % -----------------------------------------------------
                    % A. PROCESAR INTERCAMBIO (Q_t) 
                    % -----------------------------------------------------
                    f_ex_name = sprintf('lime_Scenario_%s_%s_MG%d_%s_%s.mat', ...
                        scn_name, curr_model, t_idx, curr_case.suffix_ex, curr_pert);
                    
                    f_ex = fullfile(curr_case.dir_ex, f_ex_name);
                    process_and_plot(f_ex, base_dir, s_title, t_idx, false, agent_colors, gt_data, curr_model);
                    
                    % -----------------------------------------------------
                    % B. PROCESAR BOMBEO (Q_p) - Solo Escenario Altruismo
                    % -----------------------------------------------------
                    if strcmp(scn_name, 'Altruismo')
                        f_pp_name = sprintf('lime_pumping_Scenario_%s_%s_MG%d_%s_%s.mat', ...
                            scn_name, curr_model, t_idx, curr_case.suffix_pp, curr_pert);
                        
                        f_pp = fullfile(curr_case.dir_pp, f_pp_name);
                        process_and_plot(f_pp, base_dir, s_title, t_idx, true, agent_colors, gt_data, curr_model);
                    end
                    
                end % targets
            end % scenarios
        end % models
    end % perturbations
end % cases

fprintf('\n=== PROCESO COMPLETADO EXITOSAMENTE ===\n');
fprintf('Verifique la carpeta "figures_Scenarios".\n');


%% --- FUNCIÓN DE PROCESAMIENTO ---
function process_and_plot(filename, output_dir, s_title_base, t_idx, is_pumping, colors, gt_data, curr_model)
    
    if ~isfile(filename)
        return; 
    end
    
    fprintf('    > Plotting: %s\n', filename);
    data = load(filename);
    
    all_explanations = data.all_explanations;
    feature_names = data.feature_names;
    
    % Recuperar K_TARGET para cálculos temporales
    if isfield(data, 'K_TARGET'), K = data.K_TARGET; else, K = 1; end
    
    % --- RECONSTRUCCIÓN DE FEATURES (X_values) ---
    if isfield(data, 'estado')
        X_values = data.estado.X_original;
    elseif isfield(data, 'X_original')
        X_values = data.X_original;
    else
        X_values = zeros(1, length(feature_names));
    end
    
    % Si X_values son puros ceros, significa que no se guardaron. Los reconstruimos:
    if all(X_values == 0)
        try
            [estado_rec, params_rec] = reconstruct_state_matlab_3mg(K, curr_model);
            
            try
                P_dem_pred = estado_rec.constants.p_dem_pred_full;
                P_gen_pred = estado_rec.constants.p_gen_pred_full; 
                Q_dem_pred = estado_rec.constants.q_dem_pred_full;
            catch
                P_dem_pred = params_rec.P_dem_pred; 
                P_gen_pred = params_rec.P_gen_pred; 
                Q_dem_pred = params_rec.Q_dem_pred;
            end
            
            m_P_dem = mean(P_dem_pred, 1); m_P_gen = mean(P_gen_pred, 1); m_Q_dem = mean(Q_dem_pred, 1);
            max_P_gen = max(P_gen_pred, [], 1); max_P_dem = max(P_dem_pred, [], 1); max_Q_dem = max(Q_dem_pred, [], 1);
            std_P_gen = std(P_gen_pred, 0, 1); std_P_dem = std(P_dem_pred, 0, 1); std_Q_dem = std(Q_dem_pred, 0, 1);
            
            x_base = estado_rec.X_original;
            x_base([3,8,13]) = m_P_dem; x_base([4,9,14]) = m_P_gen; x_base([5,10,15]) = m_Q_dem;
            
            if length(feature_names) == 16
                X_values = x_base;
            elseif length(feature_names) == 34
                X_values = [x_base, max_P_gen, max_P_dem, max_Q_dem, std_P_gen, std_P_dem, std_Q_dem];
            end
        catch ME
            fprintf('      [!] Warning: No se pudo reconstruir X_values: %s\n', ME.message);
        end
    end
    % ----------------------------------------------
    
    if isfield(data, 'lime_stats'), r2_val = data.lime_stats.R2_mean; else, r2_val = NaN; end
    if isfield(data, 'rbo_stats'), rbo_val = data.rbo_stats.mean; elseif isfield(data, 'rbo_mean'), rbo_val = data.rbo_mean; else, rbo_val = NaN; end
    num_runs = length(all_explanations);
    
    Ts_sim = 60; t_sec = (K-1)*Ts_sim;
    d = floor(t_sec/86400)+1; 
    rem = mod(t_sec, 86400); h = floor(rem/3600); m = round((rem - h*3600)/60);
    
    % Extracción del valor real preciso desde el Ground Truth
    real_val = 0;
    if is_pumping
        if isfield(gt_data, 'Q_p'), real_val = gt_data.Q_p(K, t_idx); end
    else
        if isfield(gt_data, 'Q_t'), real_val = gt_data.Q_t(K, t_idx); end
    end
    
    N_feat = length(feature_names);
    w_mat = zeros(N_feat, num_runs);
    
    for i = 1:num_runs
        run_d = all_explanations{i};
        map_t = containers.Map(run_d(:,1), [run_d{:,2}]);
        for j = 1:N_feat
            if isKey(map_t, feature_names{j})
                w_mat(j, i) = map_t(feature_names{j});
            end
        end
    end
    avg_w = mean(w_mat, 2);
    std_w = std(w_mat, 0, 2);
    
    plot_labels = cell(N_feat, 1);
    groups = zeros(N_feat, 1);
    
    for i = 1:N_feat
        raw = feature_names{i};
        if contains(raw,'MG1'), g=1; elseif contains(raw,'MG2'), g=2; elseif contains(raw,'MG3'), g=3; elseif contains(raw,'aq','IgnoreCase',true), g=4; else, g=4; end
        groups(i) = g;
        plot_labels{i} = parse_label_latex(raw, X_values(i));
    end
    
    [~, sort_idx] = sort(abs(avg_w), 'descend');
    sorted_labels = plot_labels(sort_idx);
    sorted_real_w = avg_w(sort_idx);
    sorted_std = std_w(sort_idx);
    
    infl_agents = zeros(1,4);
    for g=1:4, infl_agents(g) = sum(abs(avg_w(groups==g))); end
    total = sum(infl_agents); if total<1e-9, total=1; end
    pct = (infl_agents/total)*100;
    
    if is_pumping
        target_sym = sprintf('$Q_{p}^{%d}$', t_idx);
        plot_title = [s_title_base ' (Pumping)'];
    else
        target_sym = sprintf('$Q_{s}^{%d}$', t_idx);
        plot_title = s_title_base;
    end
    
    [~, name_core, ~] = fileparts(filename);
    f_rank = fullfile(output_dir, ['Ranking_' name_core]);
    f_int = fullfile(output_dir, ['Interaction_' name_core]);
    
    create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, plot_title, ...
        d, h, m, real_val, t_idx, target_sym, r2_val, rbo_val, num_runs, f_rank);
        
    create_interaction_plot(pct, colors, plot_title, ...
        d, h, m, real_val, t_idx, target_sym, r2_val, rbo_val, num_runs, f_int);
end


%% --- HELPER: LABEL PARSER (Robust + Latex Fix) ---
function label = parse_label_latex(raw_name, val)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    is_max  = contains(raw_name, 'Max_', 'IgnoreCase', true);
    is_std  = contains(raw_name, 'Std_', 'IgnoreCase', true);
    
    core_name = regexprep(raw_name, '(Mean_|Max_|Std_|MG\d_)', '', 'ignorecase');
    
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
        sym = 'EAW'; val_fmt='%.0f L';
    else
        sym = 'X'; val_fmt='%.2f'; 
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
    
    val_str = sprintf(val_fmt, val);
    
    if contains(raw_name, 'MG1'), sub='1'; elseif contains(raw_name, 'MG2'), sub='2'; elseif contains(raw_name, 'MG3'), sub='3'; else, sub=''; end
    
    if ~isempty(sub)
        label = sprintf('$%s^{%s}$ (%s)', final_sym, sub, val_str);
    else
        label = sprintf('$%s$ (%s)', final_sym, val_str);
    end
end


%% --- PLOT 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_txt, d, h, m, val, mg, target_sym, r2, rbo, runs, fname)
    N = min(length(weights), 20); % Top 20 features max
    weights = weights(1:N); errors = errors(1:N); labels = labels(1:N);
    
    fig = figure('Visible','off','Units','inches','Position',[0 0 7 6],'Color','w');
    
    % Colores barras
    cdata = repmat([0.466 0.674 0.188], N, 1);
    cdata(weights < 0, :) = repmat([0.635 0.078 0.184], sum(weights<0), 1);
    
    b = barh(weights, 0.6, 'FaceColor','flat', 'CData',cdata); hold on;
    errorbar(weights, 1:N, errors, 'horizontal', 'k', 'LineStyle','none');
    
    set(gca, 'YTick', 1:N, 'YTickLabel', labels, 'YDir','reverse', ...
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', 'FontSize',10);
    
    xlabel('Influence Weight', 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','bold');
    
    % TÍTULO SEGURO PARA LATEX
    stats_str = sprintf('$R^2=%.2f$ | RBO=%.2f | Runs=%d', r2, rbo, runs);
    full_title = {['LIME Analysis: ' title_txt]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', mg, d, h, m, target_sym, val); ...
                  stats_str};
              
    title(full_title, 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',11);
    
    xlim_val = max(abs(weights))*1.2; if xlim_val<1e-6, xlim_val=1; end
    xlim([-xlim_val, xlim_val]); xline(0,'k-'); grid on;
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end


%% --- PLOT 2: INTERACTION ---
function create_interaction_plot(pct, colors, title_txt, d, h, m, val, mg, target_sym, r2, rbo, runs, fname)
    fig = figure('Visible','off','Units','inches','Position',[0 0 6 4.5],'Color','w');
    
    b = bar(1:4, pct, 0.6, 'FaceColor','flat'); b.CData = colors;
    
    ylabel('Relative Total Influence [\%]', 'Interpreter', 'latex', 'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
    xticks(1:4); xticklabels({'MG1','MG2','MG3','Aquifer'}); ylim([0 100]); grid on;
    
    for i=1:4, text(i, pct(i)+3, sprintf('%.1f\\%%',pct(i)), 'Interpreter', 'latex', 'HorizontalAlignment','center','FontSize',9); end
    
    % TÍTULO SEGURO PARA LATEX
    stats_str = sprintf('$R^2=%.2f$ | RBO=%.2f | Runs=%d', r2, rbo, runs);
    full_title = {['Interaction: ' title_txt]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', mg, d, h, m, target_sym, val); ...
                  stats_str};
              
    title(full_title, 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',11);
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end