%% --- File: plot_pumping_balance_time_series.m ---
%
% VISUALIZATION SCRIPT: PUMPING ANALYSIS TIME SERIES
%
% Updates:
% - Corrected label for Renewable Generation (P_G) to be rigorous.
%
%--------------------------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
scenarios_list = {'EnergyEfficiency', 'AquiferConstraint'};
targets_list = [1, 2, 3];

% Simulation Parameters
Ts_sim = 60; 
T_sim_days = 7;
total_samples = T_sim_days * 24 * 60;

% Scenario Marker Colors
color_map = containers.Map;
color_map('EnergyEfficiency') = [0.9290 0.6940 0.1250];  % Gold
color_map('AquiferConstraint') = [0.4000 0.4000 0.4000]; % Gray

% Output Directories
dir_paper = 'figures_paper';
dir_pres = 'figures_presentation';
if ~exist(dir_paper, 'dir'), mkdir(dir_paper); end
if ~exist(dir_pres, 'dir'), mkdir(dir_pres); end

fprintf('--- GENERATING PUMPING TIME SERIES PLOTS (Rigorous Labels) ---\n');

% --- 1. LOAD DATA ---
try
    res = load('results_mpc/resultados_mpc_3mg_7dias.mat');
    prof = load('utils/full_profiles_for_sim.mat');
    
    N_limit = min(size(res.V_tank, 1), total_samples);
    
    % Microgrid Data
    Qp_data_all = res.Q_p(1:N_limit, :);
    V_data_all = res.V_tank(1:N_limit, :);
    SoC_data_all = res.SoC(1:N_limit, :);
    
    % --- CÁLCULO ROBUSTO DE DESCENSO (s) ---
    if isfield(res, 's')
        s_data_all = res.s(1:N_limit, :);
    elseif isfield(res, 's_pozo')
        s_data_all = res.s_pozo(1:N_limit, :);
    elseif isfield(res, 'h_p') && isfield(res, 'mg')
        h_p_data = res.h_p(1:N_limit, :);
        s_data_all = zeros(size(h_p_data));
        for i = 1:3
            s_data_all(:, i) = res.mg(i).h_p0 - h_p_data(:, i);
        end
        fprintf('  > Variable "s" calculada desde "h_p" y "h_p0".\n');
    else
        warning('No se pudo calcular el descenso (s). Usando ceros.');
        s_data_all = zeros(N_limit, 3);
    end
    
    PG_data_all = prof.P_gen_sim(1:N_limit, :);
    QL_data_all = prof.Q_dem_sim(1:N_limit, :);
    
    % Aquifer Data
    if isfield(res, 'EAW')
        EAW_data = res.EAW(1:N_limit);
    else
        EAW_0 = 1e6; Rp_day = 10000; Rp_step = Rp_day / (24*60);
        EAW_data = zeros(N_limit, 1); EAW_data(1) = EAW_0;
        total_Qp = sum(Qp_data_all, 2);
        for k = 1:N_limit-1
            EAW_data(k+1) = EAW_data(k) + Rp_step - total_Qp(k)*60;
        end
    end
    
    t_vector = (0:N_limit-1)' * Ts_sim / 86400; 
    t_plot = t_vector + 1; 
    
catch ME
    error('Error loading data: %s.', ME.message);
end

% --- 2. PLOTTING LOOP ---
for mg_idx = targets_list
    for s_idx = 1:length(scenarios_list)
        scn_name = scenarios_list{s_idx};
        
        % Load LIME file
        lime_file = sprintf('lime_PUMP_%s_MG%d.mat', scn_name, mg_idx);
        if ~exist(lime_file, 'file')
            lime_file_alt = sprintf('lime_Scenario_%s_MG%d.mat', scn_name, mg_idx); 
            if exist(lime_file_alt, 'file'), lime_file = lime_file_alt; else, continue; end
        end
        
        load(lime_file, 'K_TARGET');
        if K_TARGET > N_limit, continue; end
        
        t_mark = t_plot(K_TARGET);
        marker_color = color_map(scn_name);
        
        d.Qp = Qp_data_all(:, mg_idx);
        d.PG = PG_data_all(:, mg_idx);
        d.s  = s_data_all(:, mg_idx);
        d.SoC = SoC_data_all(:, mg_idx);
        d.QL = QL_data_all(:, mg_idx);
        d.V = V_data_all(:, mg_idx);
        d.EAW = EAW_data;
        
        switch scn_name
            case 'EnergyEfficiency'
                s_title = 'Energy Efficiency (Solar Pumping)'; plot_mode = 'solar';
            case 'AquiferConstraint'
                s_title = 'Aquifer Constraint Compliance'; plot_mode = 'drawdown';
            otherwise
                s_title = scn_name; plot_mode = 'solar';
        end
        
        fname_pap = fullfile(dir_paper, sprintf('Series_Pump_%s_MG%d_Paper', scn_name, mg_idx));
        create_pumping_series_figure(t_plot, d, K_TARGET, t_mark, marker_color, ...
            mg_idx, s_title, plot_mode, 'paper', fname_pap);
            
        fname_pres = fullfile(dir_pres, sprintf('Series_Pump_%s_MG%d_Slide', scn_name, mg_idx));
        create_pumping_series_figure(t_plot, d, K_TARGET, t_mark, marker_color, ...
            mg_idx, s_title, plot_mode, 'presentation', fname_pres);
            
        fprintf('  > Generated: %s (MG%d) [Mode: %s]\n', scn_name, mg_idx, plot_mode);
    end
end
fprintf('--- ALL PUMPING SERIES PLOTS EXPORTED ---\n');


%% --- LOCAL FUNCTION: PLOTTING ENGINE ---
function create_pumping_series_figure(t, d, k_idx, t_mark, m_color, mg, scn_title, plot_mode, export_mode, filename)
    
    if strcmp(export_mode, 'paper')
        fig_width = 7.16; fig_height = 6.0; 
        font_sz = 9; title_sz = 11; line_w = 0.8; mark_sz = 20;
    else
        fig_width = 14; fig_height = 8; 
        font_sz = 12; title_sz = 16; line_w = 1.5; mark_sz = 50;
    end
    
    fig = figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'Visible', 'off', 'Color', 'w');
    
    % --- 1. PUMPING ---
    ax1 = subplot(3, 2, 1);
    plot_signal(t, d.Qp, t_mark, d.Qp(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Pumping $Q_{p}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
    
    % --- 2. DYNAMIC SUBPLOT ---
    ax2 = subplot(3, 2, 2);
    if strcmp(plot_mode, 'solar')
        % --- CORRECCIÓN AQUÍ: NOMBRE Y SÍMBOLO RIGUROSO ---
        plot_signal_color(t, d.PG, t_mark, d.PG(k_idx), m_color, [0.929 0.694 0.125], line_w, mark_sz, ...
            sprintf('Predicted Renewable Generation $\\hat{P}_{G}^{%d}$', mg), 'Power [kW]', font_sz);
        % --------------------------------------------------
    else
        plot_signal_color(t, d.s, t_mark, d.s(k_idx), m_color, [0.635 0.078 0.184], line_w, mark_sz, ...
            sprintf('Well Drawdown $s^{%d}$', mg), 'Level [m]', font_sz);
    end
        
    % --- 3. BATTERY SOC ---
    ax3 = subplot(3, 2, 3);
    plot_signal_color(t, d.SoC*100, t_mark, d.SoC(k_idx)*100, m_color, [0.466 0.674 0.188], line_w, mark_sz, ...
        sprintf('Battery State of Charge $SoC^{%d}$', mg), 'SoC [%]', font_sz);
    yline(20, 'k:', 'LineWidth', 0.5); yline(80, 'k:', 'LineWidth', 0.5);
    
    % --- 4. WATER DEMAND ---
    ax4 = subplot(3, 2, 4);
    plot_signal(t, d.QL, t_mark, d.QL(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Water Consumption $Q_{L}^{%d}$', mg), 'Water Flow [L/s]', font_sz);
        
    % --- 5. TANK VOLUME ---
    ax5 = subplot(3, 2, 5);
    plot_signal(t, d.V, t_mark, d.V(k_idx), m_color, line_w, mark_sz, ...
        sprintf('Tank Volume $V_{Tank}^{%d}$', mg), 'Volume [L]', font_sz);
        
    % --- 6. AQUIFER LEVEL ---
    ax6 = subplot(3, 2, 6);
    plot_signal_color(t, d.EAW, t_mark, d.EAW(k_idx), m_color, [0.5 0.5 0.5], line_w, mark_sz, ...
        'Aquifer Level $EAW$', 'Volume [L]', font_sz);
    
    sgtitle(sprintf('Pumping Dynamics - MG%d - %s', mg, scn_title), ...
        'FontName', 'Times New Roman', 'FontSize', title_sz, 'FontWeight', 'bold');
    
    xlabel(ax5, 'Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_sz);
    xlabel(ax6, 'Time [Days]', 'FontName', 'Times New Roman', 'FontSize', font_sz);
    
    linkaxes([ax1, ax2, ax3, ax4, ax5, ax6], 'x');
    xlim([1 8]); xticks(1:7);
    
    exportgraphics(fig, [filename '.pdf'], 'ContentType', 'vector');
    close(fig);
end

function plot_signal(t, y, t_m, y_m, m_col, lw, msz, tit, ylab, fsz)
    hold on;
    plot(t, y, '-', 'Color', [0 0.4470 0.7410], 'LineWidth', lw); 
    scatter(t_m, y_m, msz, m_col, 'filled', 'MarkerEdgeColor', 'k');
    ylabel(ylab, 'FontName', 'Times New Roman', 'FontSize', fsz);
    title(tit, 'FontName', 'Times New Roman', 'FontSize', fsz+1, 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', fsz);
    grid on; box on; xlim([1 8]); xticks(1:7);
end

function plot_signal_color(t, y, t_m, y_m, m_col, l_col, lw, msz, tit, ylab, fsz)
    hold on;
    plot(t, y, '-', 'Color', l_col, 'LineWidth', lw); 
    scatter(t_m, y_m, msz, m_col, 'filled', 'MarkerEdgeColor', 'k');
    ylabel(ylab, 'FontName', 'Times New Roman', 'FontSize', fsz);
    title(tit, 'FontName', 'Times New Roman', 'FontSize', fsz+1, 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', fsz);
    grid on; box on; xlim([1 8]); xticks(1:7);
end