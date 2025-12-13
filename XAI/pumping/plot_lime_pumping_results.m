%% --- File: plot_lime_pumping_results.m ---
%
% MASTER VISUALIZATION SCRIPT FOR LIME RESULTS (Pumping Q_p)
%
% Generates 2 types of plots for each scenario/agent:
% 1. RANKING PLOT: Detailed feature influence (Thesis Notation).
% 2. DRIVERS PLOT: Aggregated influence by Physical Driver Category:
%    [Energy Availability, Water Needs, Network Pressure, Aquifer]
%
% Output: High-quality PDFs for IEEE Papers and Presentations.
%
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
scenarios_list = {'EnergyEfficiency', 'AquiferConstraint'};
targets_list = [1, 2, 3];

% Simulation Parameters
Ts_sim = 60;   
Ts_mpc = 1800; 

% Driver Colors for Interaction Plot
c_energy = [0.9290 0.6940 0.1250]; % Yellow/Gold (Solar/Energy)
c_water  = [0.0000 0.4470 0.7410]; % Blue (Local Water Needs)
c_net    = [0.8500 0.3250 0.0980]; % Orange (Network/Neighbors)
c_aq     = [0.5000 0.5000 0.5000]; % Gray (Aquifer)

% Output Directories
dir_paper = 'figures_paper';
dir_pres  = 'figures_presentation';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING LIME PUMPING PLOTS (Thesis Style) ---\n');

for s_idx = 1:length(scenarios_list)
    scn_name = scenarios_list{s_idx};
    
    % Define Clean Title Translation
    switch scn_name
        case 'EnergyEfficiency'
            scn_title = 'Scenario A: Energy Efficiency (Solar Pumping)';
        case 'AquiferConstraint'
            scn_title = 'Scenario B: Aquifer Constraint Compliance';
        otherwise
            scn_title = scn_name;
    end
    
    for t_idx = targets_list
        % Load Result File
        filename = sprintf('lime_PUMP_%s_MG%d.mat', scn_name, t_idx);
        
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
        
        % Get Real Output Value (Q_p)
        % Note: Q_p is usually positive.
        real_Qp = data.estado.Y_target_real_vector(t_idx);
        
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
        
        % --- 2. GENERATE RANKING LABELS & CLASSIFY DRIVERS ---
        plot_labels = cell(N_features, 1);
        driver_groups = zeros(N_features, 1); 
        % 1=Energy, 2=Water(Local), 3=Network(Neighbors), 4=Aquifer
        
        for i = 1:N_features
            raw_name = feature_names{i};
            val = X_values(i); 
            
            % A. DETECT AGENT OWNER
            g_owner = 4; % Default to Aquifer/Shared
            if contains(raw_name, 'MG1', 'IgnoreCase', true), g_owner=1;
            elseif contains(raw_name, 'MG2', 'IgnoreCase', true), g_owner=2;
            elseif contains(raw_name, 'MG3', 'IgnoreCase', true), g_owner=3;
            end
            
            % B. CLASSIFY DRIVER TYPE
            if contains(raw_name, 'aq', 'IgnoreCase', true) || contains(raw_name, 'Acuifero', 'IgnoreCase', true)
                driver_groups(i) = 4; % Aquifer
            elseif g_owner == t_idx
                % It's LOCAL variable
                if contains(raw_name, 'P_gen', 'IgnoreCase', true) || ...
                   contains(raw_name, 'SoC', 'IgnoreCase', true) || ...
                   contains(raw_name, 'P_dem', 'IgnoreCase', true) % Elec Demand affects Energy availability
                    driver_groups(i) = 1; % Energy
                else
                    driver_groups(i) = 2; % Water Need (Tank, Demand)
                end
            else
                % It's NEIGHBOR variable
                driver_groups(i) = 3; % Network Pressure
            end
            
            % C. FORMAT LABELS (Thesis Notation)
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
                sym = 'EAW'; val_str = sprintf('%.0f L', val); g_owner = 4;
            else
                clean = regexprep(raw_name, 'MG\d_', '');
                sym = strrep(clean, '_', '\_'); val_str = sprintf('%.2f', val);
            end
            
            if g_owner < 4
                plot_labels{i} = sprintf('$%s^{%d}$ (%s)', sym, g_owner, val_str);
            else
                plot_labels{i} = sprintf('$%s$ (%s)', sym, val_str);
            end
        end
        
        % --- 3. SORTING FOR RANKING ---
        [sorted_w, sort_idx] = sort(abs(avg_weights), 'descend');
        sorted_labels = plot_labels(sort_idx);
        sorted_real_w = avg_weights(sort_idx);
        sorted_std = std_weights(sort_idx);
        
        % --- 4. CALCULATE AGGREGATED DRIVERS ---
        influence_drivers = zeros(1, 4);
        for g = 1:4
            idx_group = (driver_groups == g);
            influence_drivers(g) = sum(abs(avg_weights(idx_group)));
        end
        total_infl = sum(influence_drivers);
        if total_infl < 1e-9, total_infl = 1; end
        influence_pct = (influence_drivers ./ total_infl) * 100;
        
        % --- 5. PLOTTING ---
        
        % A. RANKING PLOT
        fname_rank_pap = fullfile(dir_paper, sprintf('LIME_PumpRank_%s_MG%d_Paper', scn_name, t_idx));
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, ...
                            scn_title, day_num, hour_val, min_val, real_Qp, t_idx, ...
                            'paper', fname_rank_pap);
                            
        fname_rank_pres = fullfile(dir_pres, sprintf('LIME_PumpRank_%s_MG%d_Slide', scn_name, t_idx));
        create_ranking_plot(sorted_real_w, sorted_std, sorted_labels, ...
                            scn_title, day_num, hour_val, min_val, real_Qp, t_idx, ...
                            'presentation', fname_rank_pres);
        
        % B. DRIVERS PLOT
        driver_colors = [c_energy; c_water; c_net; c_aq];
        fname_drv_pap = fullfile(dir_paper, sprintf('LIME_PumpDrivers_%s_MG%d_Paper', scn_name, t_idx));
        create_drivers_plot(influence_pct, driver_colors, ...
                            scn_title, day_num, hour_val, min_val, real_Qp, t_idx, ...
                            'paper', fname_drv_pap);
                            
        fname_drv_pres = fullfile(dir_pres, sprintf('LIME_PumpDrivers_%s_MG%d_Slide', scn_name, t_idx));
        create_drivers_plot(influence_pct, driver_colors, ...
                            scn_title, day_num, hour_val, min_val, real_Qp, t_idx, ...
                            'presentation', fname_drv_pres);
    end
end
fprintf('--- PUMPING PLOTS EXPORTED SUCCESSFULLY ---\n');


%% --- PLOTTING FUNCTION 1: RANKING ---
function create_ranking_plot(weights, errors, labels, title_text, day, hour, minute, qp_val, mg_idx, mode, filename)
    N = length(weights);
    if strcmp(mode, 'paper')
        fig_width = 7; fig_height = 6; font_ax = 10; font_tit = 11; bar_w = 0.6;
        pos_vec = [0.35 0.12 0.60 0.78]; 
    else
        fig_width = 14; fig_height = 8; font_ax = 14; font_tit = 16; bar_w = 0.7;
        pos_vec = [0.25 0.12 0.70 0.78];
    end
    
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    % Colors: Green for Positive Influence (Encourages Pumping), Red for Negative (Discourages)
    bar_colors = zeros(N, 3);
    for k = 1:N
        if weights(k) >= 0, bar_colors(k,:) = [0.4660 0.6740 0.1880]; 
        else, bar_colors(k,:) = [0.6350 0.0780 0.1840]; end
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
       
    full_title = {title_text; sprintf('MG%d | Day %d, %02d:%02d | $Q_{p}^{%d} = %.3f$ L/s (Pumping)', mg_idx, day, hour, minute, mg_idx, qp_val)};
    title(full_title, 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    
    ax.Position = pos_vec;
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end


%% --- PLOTTING FUNCTION 2: DRIVERS ---
function create_drivers_plot(pct_values, colors, title_text, day, hour, minute, qp_val, mg_idx, mode, filename)
    
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
    % Categories specific to Pumping Analysis
    xticklabels({'Energy (Local)', 'Water Needs', 'Neighbors', 'Aquifer'});
    
    ax = gca;
    set(ax, 'FontName', 'Times New Roman', 'FontSize', font_ax);
    grid on; ylim([0 100]);
    
    for i=1:4
        text(i, pct_values(i)+3, sprintf('%.1f%%', pct_values(i)), ...
            'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', ...
            'FontSize', font_ax, 'FontWeight', 'bold');
    end
    
    full_title = {['Drivers Analysis: ' title_text]; ...
                  sprintf('Target: MG%d | Day %d, %02d:%02d | $Q_{p}^{%d} = %.3f$ L/s', ...
                  mg_idx, day, hour, minute, mg_idx, qp_val)};
              
    title(full_title, 'FontName', 'Times New Roman', 'FontSize', font_tit, 'Interpreter', 'latex');
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end