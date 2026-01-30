%% --- File: plot_lime_results_3mg_MEAN.m ---
%
% VISUALIZATION SCRIPT FOR LIME (MEAN/AVERAGE VERSION)
%
% Generates plots for Q_t (Water Exchange) where input features
% representing predictions (P_gen, P_dem, Q_dem) are AVERAGES over the horizon.
%
% FIXES:
% - LaTeX Escape for % symbol in SoC (Critical Fix).
% - Robust case-insensitive name detection.
% - Agent-based grouping for Interaction Plots.
%
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
scenarios_list = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
targets_list = [1, 2, 3];

% Colors for Interaction Plot (Agent Colors)
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray

% Output Directories
dir_paper = 'figures_paper_mean';
dir_pres  = 'figures_presentation_mean';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING LIME PLOTS (MEAN FEATURE VERSION - FIXED) ---\n');

for s_idx = 1:length(scenarios_list)
    scn_name = scenarios_list{s_idx};
    
    switch scn_name
        case 'GlobalPeak', s_title = 'Scenario A: Global Peak Interaction';
        case 'Altruismo', s_title = 'Scenario B: Active Water Export';
        case 'DirectSatisfaction', s_title = 'Scenario C: Direct Demand Satisfaction';
        otherwise, s_title = scn_name;
    end
    
    for t_idx = targets_list
        
        % LOAD _MEAN FILE
        filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scn_name, t_idx);
        if ~exist(filename, 'file')
            fprintf('  [!] File not found: %s\n', filename);
            continue;
        end
        
        fprintf('  > Processing: %s...\n', filename);
        data = load(filename);
        
        all_explanations = data.all_explanations;
        feature_names = data.feature_names;
        X_values = data.estado.X_original;
        K_TARGET = data.K_TARGET;
        
        % Time info
        Ts_sim = 60; 
        t_seconds = (K_TARGET - 1) * Ts_sim;
        day_num = floor(t_seconds / 86400) + 1;
        rem_seconds = mod(t_seconds, 86400);
        hour_val = floor(rem_seconds / 3600);
        min_val = round((rem_seconds - hour_val*3600) / 60);
        
        real_Qt = data.estado.Y_target_real_vector(t_idx);
        
        % Weights processing
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
        
        % --- GENERATE LABELS & GROUPS ---
        plot_labels = cell(N_features, 1);
        groups_vec = zeros(N_features, 1); 
        
        for i = 1:N_features
            raw_name = feature_names{i};
            val = X_values(i);
            
            % A. Identify Owner (Agent Grouping)
            g_owner = 4; 
            if contains(raw_name, 'MG1', 'IgnoreCase', true), g_owner=1;
            elseif contains(raw_name, 'MG2', 'IgnoreCase', true), g_owner=2;
            elseif contains(raw_name, 'MG3', 'IgnoreCase', true), g_owner=3;
            end
            if contains(raw_name, 'aq', 'IgnoreCase', true), g_owner = 4; end
            
            groups_vec(i) = g_owner; 
            
            % B. Generate Robust Label
            plot_labels{i} = get_mean_latex_label(raw_name, val, g_owner);
        end
        
        % --- SORTING ---
        [sorted_w, sort_idx] = sort(abs(avg_weights), 'descend');
        sorted_labels = plot_labels(sort_idx);
        sorted_real_w = avg_weights(sort_idx);
        sorted_std = std_weights(sort_idx);
        
        % --- AGGREGATION ---
        influence_per_agent = zeros(1, 4);
        for g = 1:4
            idx_group = (groups_vec == g);
            influence_per_agent(g) = sum(abs(avg_weights(idx_group)));
        end
        total_infl = sum(influence_per_agent);
        if total_infl < 1e-9, total_infl = 1; end
        influence_pct = (influence_per_agent / total_infl) * 100;
        
        % --- PLOTTING ---
        
        % 1. RANKING PLOT
        fname_rank_pap = fullfile(dir_paper, sprintf('Ranking_%s_MG%d_MEAN_Paper', scn_name, t_idx));
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, s_title, ...
            day_num, hour_val, min_val, real_Qt, t_idx, 'paper', fname_rank_pap);
            
        fname_rank_pres = fullfile(dir_pres, sprintf('Ranking_%s_MG%d_MEAN_Slide', scn_name, t_idx));
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, s_title, ...
            day_num, hour_val, min_val, real_Qt, t_idx, 'presentation', fname_rank_pres);
            
        % 2. INTERACTION PLOT
        agent_colors = [color_mg1; color_mg2; color_mg3; color_aq];
        
        fname_int_pap = fullfile(dir_paper, sprintf('Interaction_%s_MG%d_MEAN_Paper', scn_name, t_idx));
        create_interaction_plot(influence_pct, agent_colors, s_title, ...
            day_num, hour_val, min_val, real_Qt, t_idx, 'paper', fname_int_pap);
            
        fname_int_pres = fullfile(dir_pres, sprintf('Interaction_%s_MG%d_MEAN_Slide', scn_name, t_idx));
        create_interaction_plot(influence_pct, agent_colors, s_title, ...
            day_num, hour_val, min_val, real_Qt, t_idx, 'presentation', fname_int_pres);
            
    end
end
fprintf('--- ALL MEAN PLOTS EXPORTED SUCCESSFULLY ---\n');


%% --- HELPER: LABEL PARSER (ROBUST + LATEX FIX) ---
function label = get_mean_latex_label(raw_name, val, g_owner)
    is_mean = contains(raw_name, 'Mean_', 'IgnoreCase', true);
    
    % Limpiamos el nombre para buscar el núcleo (Core Name)
    % Quitamos MGx_ y Mean_ para comparar solo la variable
    core_name = regexprep(raw_name, 'MG\d_', '', 'ignorecase');
    core_name = regexprep(core_name, 'Mean_', '', 'ignorecase');
    
    % Detección Robusta
    if contains(core_name, 'SoC', 'IgnoreCase', true)
        sym = 'SoC'; 
        % === CORRECCIÓN CRÍTICA ===
        % Usamos \\%% para que LaTeX renderice el símbolo %
        val_fmt = '%.1f\\%%'; 
        val = val * 100;
        
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
        sym = 'X'; val_fmt='%.2f'; % Default
    end
    
    % Notación Matemática
    if is_mean
        final_sym = sprintf('\\bar{%s}', sym);
    else
        final_sym = sym; 
    end
    
    val_str = sprintf(val_fmt, val);
    
    if g_owner < 4
        label = sprintf('$%s^{%d}$ (%s)', final_sym, g_owner, val_str);
    else
        label = sprintf('$%s$ (%s)', final_sym, val_str);
    end
end


%% --- PLOT 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_text, d, h, m, qt, mg, mode, fname)
    N = length(weights);
    if strcmp(mode, 'paper')
        fig_w=7; fig_h=6; font_ax=10; font_t=11; bar_w=0.6;
        pos_ax = [0.35 0.12 0.60 0.78];
    else
        fig_w=14; fig_h=8; font_ax=14; font_t=16; bar_w=0.7;
        pos_ax = [0.25 0.12 0.70 0.78];
    end
    
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
    
    if qt > 0, flow_str = 'Exporting'; else, flow_str = 'Importing'; end
    
    full_title = {['Mean-Feature Analysis: ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | $Q_{t}^{%d}=%.2f$ L/s (%s)', ...
                  mg, d, h, m, mg, qt, flow_str)};
              
    title(full_title, 'FontName','Times New Roman', 'FontSize',font_t, 'Interpreter','latex');
    
    xlim_val = max(abs(weights))*1.2; if xlim_val<1e-6, xlim_val=1; end
    xlim([-xlim_val, xlim_val]); xline(0,'k-'); grid on;
    
    ax.Position = pos_ax;
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end


%% --- PLOT 2: INTERACTION ---
function create_interaction_plot(pct, cmap, title_text, d, h, m, qt, mg, mode, fname)
    if strcmp(mode, 'paper')
        fig_w=6; fig_h=4.5; font_ax=10; font_t=11;
    else
        fig_w=12; fig_h=7; font_ax=14; font_t=16;
    end
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
    
    if qt > 0, flow_str = 'Exporting'; else, flow_str = 'Importing'; end
    
    full_title = {['Interaction (Mean Features): ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | $Q_{t}^{%d}=%.2f$ L/s (%s)', mg, d, h, m, mg, qt, flow_str)};
              
    title(full_title, 'FontName','Times New Roman','FontSize',font_t, 'Interpreter','latex');
    
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector'); close(fig);
end