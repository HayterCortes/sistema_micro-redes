%% --- File: plot_lime_results_3mg_multi.m ---
%
% MASTER VISUALIZATION SCRIPT FOR LIME RESULTS (Water Exchange Qt)
%
% Generates 2 types of plots for each scenario/agent:
% 1. RANKING PLOT: Detailed feature influence (Thesis Notation).
% 2. INTERACTION PLOT: Aggregated influence by Agent (MG1, MG2, MG3, Aquifer).
%
% UPDATES:
% - Auto-detection of Import/Export flow direction for title accuracy.
%
% Output: High-quality PDFs for IEEE Papers and Presentations.
%
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
scenarios_list = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
targets_list = [1, 2, 3];

% Simulation Parameters
Ts_sim = 60;   
Ts_mpc = 1800; 

% Agent Colors for Interaction Plot
color_mg1 = [0 0.4470 0.7410];      % Blue
color_mg2 = [0.8500 0.3250 0.0980]; % Orange
color_mg3 = [0.9290 0.6940 0.1250]; % Yellow
color_aq  = [0.5 0.5 0.5];          % Gray

% Output Directories
dir_paper = 'figures_paper';
dir_pres  = 'figures_presentation';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING LIME PLOTS (Ranking & Interaction) ---\n');

for s_idx = 1:length(scenarios_list)
    scn_name = scenarios_list{s_idx};
    
    % Define Clean Title
    switch scn_name
        case 'GlobalPeak', scn_title = 'Scenario A: Global Peak Interaction';
        case 'Altruismo', scn_title = 'Scenario B: Active Water Export'; 
        case 'DirectSatisfaction', scn_title = 'Scenario C: Direct Demand Satisfaction';
        otherwise, scn_title = scn_name;
    end
    
    for t_idx = targets_list
        % Load Result File
        filename = sprintf('lime_Scenario_%s_MG%d.mat', scn_name, t_idx);
        
        if ~exist(filename, 'file')
            continue; 
        end
        
        fprintf('  > Processing: %s...\n', filename);
        data = load(filename);
        
        % --- 1. DATA PREPARATION ---
        all_explanations = data.all_explanations;
        feature_names = data.feature_names;
        X_values = data.estado.X_original; 
        K_TARGET = data.K_TARGET;
        
        % Calculate Time (HH:MM)
        t_seconds = (K_TARGET - 1) * Ts_sim;
        day_num = floor(t_seconds / 86400) + 1;
        rem_seconds = mod(t_seconds, 86400);
        hour_val = floor(rem_seconds / 3600);
        min_val = round((rem_seconds - hour_val*3600) / 60);
        
        % Get Real Output
        real_Qt = data.estado.Y_target_real_vector(t_idx);
        
        % Process Weights
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
        
        % --- 2. GENERATE RANKING LABELS & GROUPS ---
        plot_labels = cell(N_features, 1);
        groups_vec = zeros(N_features, 1); % 1=MG1, 2=MG2, 3=MG3, 4=Aquifer
        
        for i = 1:N_features
            raw_name = feature_names{i};
            val = X_values(i); 
            
            % A. DETECT AGENT (Group)
            g = 4; % Default to Shared/Aquifer
            if contains(raw_name, 'MG1', 'IgnoreCase', true), g=1;
            elseif contains(raw_name, 'MG2', 'IgnoreCase', true), g=2;
            elseif contains(raw_name, 'MG3', 'IgnoreCase', true), g=3;
            end
            groups_vec(i) = g;
            
            % B. DETECT VARIABLE & FORMAT LABELS
            if contains(raw_name, 'SoC', 'IgnoreCase', true)
                sym = 'SoC'; val_str = sprintf('%.2f\\%%', val*100); 
            elseif contains(raw_name, 'tank', 'IgnoreCase', true) || contains(raw_name, 'Estanque', 'IgnoreCase', true)
                sym = 'V_{Tank}'; val_str = sprintf('%.0f L', val); 
            elseif contains(raw_name, 'P_dem', 'IgnoreCase', true)
                sym = '\hat{P}_{L}'; val_str = sprintf('%.1f kW', val);
            elseif contains(raw_name, 'P_gen', 'IgnoreCase', true)
                sym = '\hat{P}_{G}'; val_str = sprintf('%.1f kW', val);
            elseif contains(raw_name, 'Q_dem', 'IgnoreCase', true)
                sym = '\hat{Q}_{L}'; val_str = sprintf('%.2f L/s', val);
            elseif contains(raw_name, 'aq', 'IgnoreCase', true)
                sym = 'EAW'; val_str = sprintf('%.0f L', val); g = 4; groups_vec(i)=4;
            else
                clean = regexprep(raw_name, 'MG\d_', '');
                sym = strrep(clean, '_', '\_'); val_str = sprintf('%.2f', val);
            end
            
            if g < 4
                plot_labels{i} = sprintf('$%s^{%d}$ (%s)', sym, g, val_str);
            else
                plot_labels{i} = sprintf('$%s$ (%s)', sym, val_str);
            end
        end
        
        % --- 3. SORTING FOR RANKING ---
        [sorted_w, sort_idx] = sort(abs(avg_weights), 'descend');
        sorted_labels = plot_labels(sort_idx);
        sorted_real_w = avg_weights(sort_idx);
        sorted_std = std_weights(sort_idx);
        
        % --- 4. CALCULATE AGGREGATED INTERACTION ---
        influence_per_agent = zeros(1, 4);
        for g = 1:4
            idx_group = (groups_vec == g);
            influence_per_agent(g) = sum(abs(avg_weights(idx_group)));
        end
        total_infl = sum(influence_per_agent);
        if total_infl < 1e-9, total_infl = 1; end
        influence_pct = (influence_per_agent / total_infl) * 100;
        
        % --- 5. PLOTTING FUNCTION CALLS ---
        
        % A. RANKING PLOT (Paper & Pres)
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, ...
                            scn_title, day_num, hour_val, min_val, real_Qt, t_idx, ...
                            'paper', fullfile(dir_paper, sprintf('Ranking_%s_MG%d_Paper', scn_name, t_idx)));
                            
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, ...
                            scn_title, day_num, hour_val, min_val, real_Qt, t_idx, ...
                            'presentation', fullfile(dir_pres, sprintf('Ranking_%s_MG%d_Slide', scn_name, t_idx)));
        
        % B. INTERACTION PLOT (Paper & Pres)
        agent_colors = [color_mg1; color_mg2; color_mg3; color_aq];
        create_interaction_plot(influence_pct, agent_colors, ...
                                scn_title, day_num, hour_val, min_val, real_Qt, t_idx, ...
                                'paper', fullfile(dir_paper, sprintf('Interaction_%s_MG%d_Paper', scn_name, t_idx)));
                                
        create_interaction_plot(influence_pct, agent_colors, ...
                                scn_title, day_num, hour_val, min_val, real_Qt, t_idx, ...
                                'presentation', fullfile(dir_pres, sprintf('Interaction_%s_MG%d_Slide', scn_name, t_idx)));
    end
end
fprintf('--- ALL PLOTS (RANKING & INTERACTION) EXPORTED SUCCESSFULLY ---\n');


%% --- PLOTTING FUNCTION 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_text, day, hour, minute, qt_val, mg_idx, mode, filename)
    N = length(weights);
    if strcmp(mode, 'paper')
        fig_width = 7; fig_height = 6; font_ax = 10; font_tit = 11; bar_w = 0.6;
        pos_vec = [0.35 0.12 0.60 0.78]; 
    else
        fig_width = 14; fig_height = 8; font_ax = 14; font_tit = 16; bar_w = 0.7;
        pos_vec = [0.25 0.12 0.70 0.78];
    end
    
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    bar_colors = zeros(N, 3);
    for k = 1:N
        if weights(k) >= 0, bar_colors(k,:) = [0.4660 0.6740 0.1880]; % Green
        else, bar_colors(k,:) = [0.6350 0.0780 0.1840]; end % Red
    end
    
    b = barh(weights, bar_w, 'FaceColor', 'flat'); b.CData = bar_colors; hold on;
    errorbar(weights, 1:N, errors, 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 1.0);
    
    ax = gca;
    set(ax, 'YTick', 1:N, 'YTickLabel', labels, 'YDir', 'reverse', ...
        'TickLabelInterpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', font_ax);
    xlim_val = max(abs(weights) + errors) * 1.15; if xlim_val < 1e-6, xlim_val=1; end
    xlim([-xlim_val, xlim_val]);
    xline(0, 'k-', 'LineWidth', 1.2); grid on;
    xlabel('Average Influence', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
       
    % --- LOGIC FOR FLOW DIRECTION ---
    if qt_val > 0
        flow_desc = 'Export';
    else
        flow_desc = 'Import';
    end
    % -------------------------------
    
    full_title = {title_text; sprintf('MG%d | Day %d, %02d:%02d | $Q_{t}^{%d} = %.3f$ L/s (%s)', mg_idx, day, hour, minute, mg_idx, qt_val, flow_desc)};
    title(full_title, 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    ax.Position = pos_vec;
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end


%% --- PLOTTING FUNCTION 2: INTERACTION ---
function create_interaction_plot(pct_values, colors, title_text, day, hour, minute, qt_val, mg_idx, mode, filename)
    
    if strcmp(mode, 'paper')
        fig_width = 6; fig_height = 4.5; font_ax = 10; font_tit = 11;
    else
        fig_width = 12; fig_height = 7; font_ax = 14; font_tit = 16;
    end
    
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    b = bar(1:4, pct_values, 0.6);
    b.FaceColor = 'flat';
    b.CData = colors;
    
    ylabel('Relative Total Influence [%]', 'FontName', 'Times New Roman', 'FontSize', font_ax, 'FontWeight', 'bold');
    xticks(1:4);
    xticklabels({'Microgrid 1', 'Microgrid 2', 'Microgrid 3', 'Aquifer'});
    ax = gca;
    set(ax, 'FontName', 'Times New Roman', 'FontSize', font_ax);
    grid on; ylim([0 100]);
    
    for i=1:4
        text(i, pct_values(i)+3, sprintf('%.1f%%', pct_values(i)), ...
            'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', ...
            'FontSize', font_ax, 'FontWeight', 'bold');
    end
    
    % --- LOGIC FOR FLOW DIRECTION ---
    if qt_val > 0
        flow_desc = 'Export';
    else
        flow_desc = 'Import';
    end
    % -------------------------------
    
    full_title = {['Interaction Analysis: ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | $Q_{t}^{%d} = %.3f$ L/s (%s)', ...
                  mg_idx, day, hour, minute, mg_idx, qt_val, flow_desc)};
              
    title(full_title, 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end