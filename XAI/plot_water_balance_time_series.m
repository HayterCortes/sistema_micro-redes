%% --- File: plot_water_balance_time_series.m ---
%
% VISUALIZATION SCRIPT: WATER BALANCE TIME SERIES
%
% Generates a 3x2 grid figure showing the evolution of all water variables
% involved in the mass balance of the microgrid.
%
% Variables:
% 1. Tank Volume (V_tank)
% 2. Net Tank Flow (Q_tank)
% 3. Pumping (Q_p)
% 4. Water Consumption (Q_L)
% 5. Water Bought (Q_buy)
% 6. Water Exchange (Q_t)
%
% Features:
% - Marks the LIME analysis instant with a specific color per scenario.
% - Exports to 'figures_paper' and 'figures_presentation'.
% - Uses Thesis Notation and consistent styling.
%
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
scenarios_list = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
targets_list = [1, 2, 3];

% Simulation Parameters
Ts_sim = 60; % 1 minute step
T_sim_days = 7;
total_samples = T_sim_days * 24 * 60;

% Scenario Marker Colors (Distinctive)
color_map = containers.Map;
color_map('GlobalPeak') = [0.8500 0.3250 0.0980];       % Orange/Red
color_map('Altruismo') = [0.4660 0.6740 0.1880];        % Green
color_map('DirectSatisfaction') = [0.4940 0.1840 0.5560]; % Purple

% Output Directories
dir_paper = 'figures_paper';
dir_pres = 'figures_presentation';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING WATER BALANCE TIME SERIES PLOTS ---\n');

% --- 1. LOAD DATA ---
try
    % Load Simulation Results
    res = load('results_mpc/resultados_mpc_3mg_7dias.mat');
    % Load Profiles (for Demand Q_L)
    prof = load('utils/full_profiles_for_sim.mat');
    
    % Extract Data (Limit to simulation length if needed)
    N_limit = min(size(res.V_tank, 1), total_samples);
    
    V_data_all = res.V_tank(1:N_limit, :);
    Qp_data_all = res.Q_p(1:N_limit, :);
    Qt_data_all = res.Q_t(1:N_limit, :);
    Qbuy_data_all = res.Q_DNO(1:N_limit, :); % Variable is Q_DNO in struct
    QL_data_all = prof.Q_dem_sim(1:N_limit, :);
    
    % Time Vector (Days 1-8 for plot consistency)
    t_vector = (0:N_limit-1)' * Ts_sim / 86400; % 0 to 7
    t_plot = t_vector + 1; % Shift to Day 1 start
    
catch ME
    error('Error loading data: %s. Ensure results_mpc and utils exist.', ME.message);
end

% --- 2. PLOTTING LOOP ---
for mg_idx = targets_list
    
    % Pre-calculate Net Flow for this MG
    % Balance: Q_tank = In - Out = (Qp + Qbuy) - (Qt + QL)
    % Note: Qt > 0 is Export (Out), so it subtracts.
    Q_net_all = Qp_data_all(:, mg_idx) + Qbuy_data_all(:, mg_idx) ...
                - Qt_data_all(:, mg_idx) - QL_data_all(:, mg_idx);
    
    for s_idx = 1:length(scenarios_list)
        scn_name = scenarios_list{s_idx};
        
        % Load LIME file to find the specific instant
        lime_file = sprintf('lime_Scenario_%s_MG%d.mat', scn_name, mg_idx);
        if ~exist(lime_file, 'file')
            continue;
        end
        load(lime_file, 'K_TARGET');
        
        % Verify Index
        if K_TARGET > N_limit
            fprintf('Warning: K_TARGET %d out of bounds for %s. Skipping.\n', K_TARGET, lime_file);
            continue;
        end
        
        % Get values for the marker
        t_mark = t_plot(K_TARGET);
        
        % Prepare Data Structure for Plotting Function
        d.V = V_data_all(:, mg_idx);
        d.Qnet = Q_net_all;
        d.Qp = Qp_data_all(:, mg_idx);
        d.QL = QL_data_all(:, mg_idx);
        d.Qbuy = Qbuy_data_all(:, mg_idx);
        d.Qt = Qt_data_all(:, mg_idx);
        
        marker_color = color_map(scn_name);
        
        % Define Title
        switch scn_name
            case 'GlobalPeak', s_title = 'Global Peak Interaction';
            case 'Altruismo', s_title = 'Active Water Export';
            case 'DirectSatisfaction', s_title = 'Direct Demand Satisfaction';
            otherwise, s_title = scn_name;
        end
        
        % --- GENERATE PAPER VERSION ---
        fname_pap = fullfile(dir_paper, sprintf('Series_Water_%s_MG%d_Paper', scn_name, mg_idx));
        create_series_figure(t_plot, d, K_TARGET, t_mark, marker_color, ...
            mg_idx, s_title, 'paper', fname_pap);
            
        % --- GENERATE PRESENTATION VERSION ---
        fname_pres = fullfile(dir_pres, sprintf('Series_Water_%s_MG%d_Slide', scn_name, mg_idx));
        create_series_figure(t_plot, d, K_TARGET, t_mark, marker_color, ...
            mg_idx, s_title, 'presentation', fname_pres);
            
        fprintf('  > Generated: %s (MG%d)\n', scn_name, mg_idx);
    end
end
fprintf('--- ALL TIME SERIES PLOTS EXPORTED ---\n');


%% --- LOCAL FUNCTION: PLOTTING ENGINE ---
function create_series_figure(t, d, k_idx, t_mark, m_color, mg, scn_title, mode, filename)
    
    if strcmp(mode, 'paper')
        % IEEE Full Width Figure (Double Column)
        fig_width = 7.16; fig_height = 6.0; 
        font_sz = 9; title_sz = 11; line_w = 0.8; mark_sz = 15;
    else
        % Presentation (16:9)
        fig_width = 14; fig_height = 8; 
        font_sz = 12; title_sz = 16; line_w = 1.5; mark_sz = 40;
    end
    
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    % Layout: 3 Rows x 2 Columns
    % Order:
    % 1. Volume (V_tank)       2. Net Flow (Q_tank)
    % 3. Pumping (Q_p)         4. Demand (Q_L)
    % 5. Bought (Q_buy)        6. Exchange (Q_t)
    
    % --- SUBPLOT 1: TANK VOLUME ---
    ax1 = subplot(3, 2, 1);
    plot_signal(t, d.V, t_mark, d.V(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Tank Volume $V_{Tank}^{%d}$', mg), 'Volume [L]', font_sz);
    
    % --- SUBPLOT 2: NET FLOW ---
    ax2 = subplot(3, 2, 2);
    plot_signal(t, d.Qnet, t_mark, d.Qnet(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Net Tank Flow $Q_{Tank}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
    yline(0, 'k:', 'LineWidth', 0.5); % Zero ref
    
    % --- SUBPLOT 3: PUMPING ---
    ax3 = subplot(3, 2, 3);
    plot_signal(t, d.Qp, t_mark, d.Qp(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Pumping $Q_{p}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
        
    % --- SUBPLOT 4: DEMAND ---
    ax4 = subplot(3, 2, 4);
    plot_signal(t, d.QL, t_mark, d.QL(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Water Consumption ${Q}_{L}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
        
    % --- SUBPLOT 5: BOUGHT ---
    ax5 = subplot(3, 2, 5);
    plot_signal(t, d.Qbuy, t_mark, d.Qbuy(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Water Bought $Q_{buy}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
        
    % --- SUBPLOT 6: EXCHANGE ---
    ax6 = subplot(3, 2, 6);
    plot_signal(t, d.Qt, t_mark, d.Qt(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Water Exchange $Q_{t}^{%d}$ (+Export/-Import)', mg), 'Water Flow [L/s]', font_sz);
    yline(0, 'k:', 'LineWidth', 0.5);
    
    % Global Title (Using text annotation for centering)
    sgtitle(sprintf('Water Balance Dynamics - MG%d - %s', mg, scn_title), ...
        'FontName', 'Times New Roman', 'FontSize', title_sz, 'FontWeight', 'bold');
    
    % Adjust X-Axis Labels (Only bottom plots need Time label)
    xlabel(ax5, 'Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_sz);
    xlabel(ax6, 'Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_sz);
    
    % Link axes for zooming time
    linkaxes([ax1, ax2, ax3, ax4, ax5, ax6], 'x');
    xlim([1 8]);
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end

function plot_signal(t, y, t_m, y_m, col, lw, msz, tit, ylab, fsz)
    hold on;
    plot(t, y, '-', 'Color', [0 0.4470 0.7410], 'LineWidth', lw); % Blue line default
    % Marker
    scatter(t_m, y_m, msz, col, 'filled', 'MarkerEdgeColor', 'k');
    
    % Styling
    ylabel(ylab, 'FontName', 'Times New Roman', 'FontSize', fsz);
    title(tit, 'FontName', 'Times New Roman', 'FontSize', fsz+1, 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', fsz);
    grid on; box on;
    xlim([1 8]); xticks(1:7);
end