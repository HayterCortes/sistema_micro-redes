%% --- File: plot_lime_results_MASTER_SCENARIOS.m ---
%
% SCRIPT MAESTRO DE VISUALIZACIÓN DE ESCENARIOS (Q_t y Q_p)
% Soporta: 3 Casos x 2 Perturbaciones x 2 Modelos.
%
% ESTRUCTURA DE SALIDA:
%   figures_Scenarios /
%       |-- Case_1_16_Features /
%             |-- PARETO / AR, TS
%             |-- GAUSSIAN / AR, TS
%       |-- Case_2_34_Features_Std / ...
%       |-- Case_3_34_Features_AE / ...
%
% INCLUYE: R2 promedio, RBO y Nro de ejecuciones en el título.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- 1. CONFIGURACIÓN GLOBAL ---
SCENARIOS_LIST = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
TARGETS_LIST = [1, 2, 3];

% Definición de los 3 Casos de Estudio
% Name: Nombre carpeta, Suffix: Identificador en nombre de archivo
CASES(1).name = 'Case_1_16_Features_Mean';   CASES(1).suffix_ex = 'MEAN_RBO';     CASES(1).suffix_pp = 'MEAN_RBO';
CASES(2).name = 'Case_2_34_Features_Std';    CASES(2).suffix_ex = 'STANDARD_RBO'; CASES(2).suffix_pp = 'STANDARD_RBO';
CASES(3).name = 'Case_3_34_Features_AE';     CASES(3).suffix_ex = 'MOMENTS_RBO';  CASES(3).suffix_pp = 'MOMENTS_AE'; 
% Nota: En Caso 3, Exchange se guardó como _MOMENTS_RBO_ y Pumping como _MOMENTS_AE_

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
            
            % Crear Estructura de Directorios
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
                    % A. PROCESAR INTERCAMBIO (Q_t) - Todos los escenarios
                    % -----------------------------------------------------
                    f_ex = sprintf('lime_Scenario_%s_%s_MG%d_%s_%s.mat', ...
                        scn_name, curr_model, t_idx, curr_case.suffix_ex, curr_pert);
                    
                    process_and_plot(f_ex, base_dir, s_title, t_idx, false, agent_colors);
                    
                    % -----------------------------------------------------
                    % B. PROCESAR BOMBEO (Q_p) - Solo Escenario Altruismo
                    % -----------------------------------------------------
                    if strcmp(scn_name, 'Altruismo')
                        f_pp = sprintf('lime_pumping_Scenario_%s_%s_MG%d_%s_%s.mat', ...
                            scn_name, curr_model, t_idx, curr_case.suffix_pp, curr_pert);
                        
                        process_and_plot(f_pp, base_dir, s_title, t_idx, true, agent_colors);
                    end
                    
                end % targets
            end % scenarios
        end % models
    end % perturbations
end % cases

fprintf('\n=== PROCESO COMPLETADO EXITOSAMENTE ===\n');
fprintf('Verifique la carpeta "figures_Scenarios".\n');


%% --- FUNCIÓN DE PROCESAMIENTO ---
function process_and_plot(filename, output_dir, s_title_base, t_idx, is_pumping, colors)
    
    if ~isfile(filename)
        return; % Si el archivo no existe, saltar silenciosamente
    end
    
    fprintf('    > Plotting: %s\n', filename);
    data = load(filename);
    
    % 1. Extraer Datos
    all_explanations = data.all_explanations;
    feature_names = data.feature_names;
    
    % Compatibilidad con diferentes versiones de guardado de X_original
    if isfield(data, 'estado')
        X_values = data.estado.X_original;
    elseif isfield(data, 'X_original')
        X_values = data.X_original;
    else
        X_values = zeros(1, length(feature_names)); % Fallback
    end
    
    % Recuperar Metadatos (R2, RBO, Runs)
    if isfield(data, 'lime_stats')
        r2_val = data.lime_stats.R2_mean;
    else
        r2_val = NaN;
    end
    
    if isfield(data, 'rbo_stats')
        rbo_val = data.rbo_stats.mean;
    elseif isfield(data, 'rbo_mean') % Compatibilidad antigua
        rbo_val = data.rbo_mean;
    else
        rbo_val = NaN;
    end
    
    num_runs = length(all_explanations);
    
    % Información temporal (para el título)
    if isfield(data, 'K_TARGET'), K = data.K_TARGET; else, K = 1; end
    Ts_sim = 60; t_sec = (K-1)*Ts_sim;
    d = floor(t_sec/86400)+1; 
    rem = mod(t_sec, 86400); h = floor(rem/3600); m = round((rem - h*3600)/60);
    
    % Valor Real
    real_val = 0;
    if isfield(data, 'estado') && isfield(data.estado, 'Y_target_real_vector')
        real_val = data.estado.Y_target_real_vector(t_idx);
    end
    
    % 2. Procesar Pesos (Media y Std)
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
    
    % 3. Etiquetas y Grupos
    plot_labels = cell(N_feat, 1);
    groups = zeros(N_feat, 1);
    
    for i = 1:N_feat
        raw = feature_names{i};
        % Identificar Agente
        if contains(raw,'MG1'), g=1; elseif contains(raw,'MG2'), g=2; elseif contains(raw,'MG3'), g=3; elseif contains(raw,'aq','IgnoreCase',true), g=4; else, g=4; end
        groups(i) = g;
        plot_labels{i} = parse_label_latex(raw, X_values(i));
    end
    
    % Ordenar
    [sorted_w, sort_idx] = sort(abs(avg_w), 'descend');
    sorted_labels = plot_labels(sort_idx);
    sorted_real_w = avg_w(sort_idx);
    sorted_std = std_w(sort_idx);
    
    % Porcentajes Interacción
    infl_agents = zeros(1,4);
    for g=1:4, infl_agents(g) = sum(abs(avg_w(groups==g))); end
    total = sum(infl_agents); if total<1e-9, total=1; end
    pct = (infl_agents/total)*100;
    
    % 4. Configurar Textos
    if is_pumping
        target_sym = sprintf('$Q_{p}^{%d}$', t_idx);
        plot_title = [s_title_base ' (Pumping)'];
        fname_suffix = '_PUMP';
    else
        target_sym = sprintf('$Q_{s}^{%d}$', t_idx);
        plot_title = s_title_base;
        fname_suffix = '';
    end
    
    % Nombres de Archivo de Salida (Sin extensión, la función añade .pdf)
    [~, name_core, ~] = fileparts(filename);
    f_rank = fullfile(output_dir, ['Ranking_' name_core]);
    f_int = fullfile(output_dir, ['Interaction_' name_core]);
    
    % 5. Generar Plots
    create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, plot_title, ...
        d, h, m, real_val, t_idx, target_sym, r2_val, rbo_val, num_runs, f_rank);
        
    create_interaction_plot(pct, colors, plot_title, ...
        d, h, m, real_val, t_idx, target_sym, r2_val, rbo_val, num_runs, f_int);
end


%% --- HELPER: LABEL PARSER (Latex) ---
function lbl = parse_label_latex(raw, val)
    is_mean = contains(raw, 'Mean_');
    is_max = contains(raw, 'Max_');
    is_std = contains(raw, 'Std_');
    
    core = regexprep(raw, '(Mean_|Max_|Std_|MG\d_)', '');
    
    if contains(core, 'SoC'), sym = 'SoC'; unit='\%'; val=val*100;
    elseif contains(core, 'tank') || contains(core, 'Estanque'), sym = 'V_{Tank}'; unit='L';
    elseif contains(core, 'P_dem'), sym = 'P_{L}'; unit='kW';
    elseif contains(core, 'P_gen'), sym = 'P_{G}'; unit='kW';
    elseif contains(core, 'Q_dem'), sym = 'Q_{L}'; unit='L/s';
    elseif contains(core, 'aq'), sym = 'EAW'; unit='L';
    else, sym = 'X'; unit=''; end
    
    % Decoradores matemáticos
    if is_mean, sym = ['\bar{' sym '}']; end
    if is_max, sym = ['\hat{' sym '}']; end
    if is_std, sym = ['\sigma(' sym ')']; end
    
    % Subíndice Agente
    if contains(raw, 'MG1'), sub='1'; elseif contains(raw, 'MG2'), sub='2'; elseif contains(raw, 'MG3'), sub='3'; else, sub=''; end
    if ~isempty(sub), sym = [sym '^{' sub '}']; end
    
    lbl = sprintf('$%s$ (%.1f%s)', sym, val, unit);
end


%% --- PLOT 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_txt, d, h, m, val, mg, target_sym, r2, rbo, runs, fname)
    N = min(length(weights), 20); % Top 20 features max
    weights = weights(1:N); errors = errors(1:N); labels = labels(1:N);
    
    fig = figure('Visible','off','Units','inches','Position',[0 0 7 6],'Color','w');
    
    % Colores barras (Positivo/Negativo)
    cdata = repmat([0.466 0.674 0.188], N, 1);
    cdata(weights < 0, :) = repmat([0.635 0.078 0.184], sum(weights<0), 1);
    
    b = barh(weights, 0.6, 'FaceColor','flat', 'CData',cdata); hold on;
    errorbar(weights, 1:N, errors, 'horizontal', 'k', 'LineStyle','none');
    
    set(gca, 'YTick', 1:N, 'YTickLabel', labels, 'YDir','reverse', ...
        'TickLabelInterpreter','latex', 'FontName','Times New Roman', 'FontSize',10);
    
    xlabel('Influence Weight', 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','bold');
    
    % TÍTULO CON MÉTRICAS
    stats_str = sprintf('$R^2=%.2f$ | RBO=%.2f | Runs=%d', r2, rbo, runs);
    full_title = {['\bf ' title_txt]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', mg, d, h, m, target_sym, val); ...
                  ['\rm\fontsize{9} ' stats_str]};
              
    title(full_title, 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',11);
    
    grid on; xline(0, 'k-');
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end


%% --- PLOT 2: INTERACTION ---
function create_interaction_plot(pct, colors, title_txt, d, h, m, val, mg, target_sym, r2, rbo, runs, fname)
    fig = figure('Visible','off','Units','inches','Position',[0 0 6 4.5],'Color','w');
    
    b = bar(1:4, pct, 0.6, 'FaceColor','flat'); b.CData = colors;
    
    ylabel('Influence Share [%]', 'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
    xticks(1:4); xticklabels({'MG1','MG2','MG3','Aquifer'}); ylim([0 100]); grid on;
    
    for i=1:4, text(i, pct(i)+3, sprintf('%.1f%%',pct(i)), 'HorizontalAlignment','center','FontSize',9); end
    
    stats_str = sprintf('$R^2=%.2f$ | RBO=%.2f | Runs=%d', r2, rbo, runs);
    full_title = {['\bf Interaction: ' title_txt]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | %s = %.2f', mg, d, h, m, target_sym, val); ...
                  ['\rm\fontsize{9} ' stats_str]};
              
    title(full_title, 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',11);
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end