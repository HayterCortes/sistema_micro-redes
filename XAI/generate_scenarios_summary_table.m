%% --- File: generate_scenarios_summary_table.m ---
%
% SCRIPT DE AUDITORÍA: RESUMEN DE ESCENARIOS DETECTADOS
% Extrae tiempo, microrred y valor de la acción explicada por LIME.
%--------------------------------------------------------------------------
clear; clc;

% --- CONFIGURACIÓN DE CASOS SEGÚN EL PAPER ---
% {Nombre_Escenario, MG_ID, Tipo_Variable}
selected_cases = {
    'Global Peak Interaction (A)',   1, 'Qs', 'Scenario_GlobalPeak'; ...
    'Global Peak Interaction (A)',   2, 'Qs', 'Scenario_GlobalPeak'; ...
    'Active Water Export (B)',       1, 'Qs', 'Scenario_Altruismo'; ...
    'Direct Satisfaction (C)',       2, 'Qs', 'Scenario_DirectSatisfaction'; ...
    'Direct Satisfaction (C)',       3, 'Qs', 'Scenario_DirectSatisfaction'; ...
    'Coordinated Pumping (D)',       1, 'Qp', 'EnergyEfficiency'; ...
    'Coordinated Pumping (D)',       2, 'Qp', 'EnergyEfficiency'; ...
    'Coordinated Pumping (D)',       3, 'Qp', 'EnergyEfficiency'
};

% Inicialización de contenedores para la tabla
scenario_col = {};
mg_col = [];
day_col = {};
hour_col = {};
var_col = {};
val_col = [];

Ts_sim = 60; % 1 minuto

fprintf('--- GENERANDO RESUMEN OPERATIVO DE ESCENARIOS ---\n');

for i = 1:size(selected_cases, 1)
    scn_label = selected_cases{i, 1};
    t_idx     = selected_cases{i, 2};
    var_type  = selected_cases{i, 3};
    file_tag  = selected_cases{i, 4};
    
    % Determinar nombre de archivo
    if strcmp(var_type, 'Qs')
        filename = sprintf('lime_%s_MG%d_MEAN.mat', file_tag, t_idx);
        target_sym = sprintf('Q_s^%d', t_idx);
    else
        filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', file_tag, t_idx);
        target_sym = sprintf('Q_p^%d', t_idx);
    end

    if exist(filename, 'file')
        data = load(filename);
        
        % Procesamiento de tiempo
        k_target = data.K_TARGET;
        t_sec = (k_target - 1) * Ts_sim;
        day_num = floor(t_sec / 86400) + 1;
        rem_s = mod(t_sec, 86400);
        h = floor(rem_s / 3600);
        m = round((rem_s - h*3600) / 60);
        
        % Valor real de la acción
        real_val = data.estado.Y_target_real_vector(t_idx);
        
        % Almacenar datos
        scenario_col{end+1, 1} = scn_label;
        mg_col(end+1, 1)       = t_idx;
        day_col{end+1, 1}      = sprintf('Day %d', day_num);
        hour_col{end+1, 1}     = sprintf('%02d:%02d', h, m);
        var_col{end+1, 1}      = target_sym;
        val_col(end+1, 1)      = real_val;
    else
        fprintf('  [!] Archivo no encontrado: %s\n', filename);
    end
end

% Crear y mostrar la tabla
SummaryTable = table(scenario_col, mg_col, day_col, hour_col, var_col, val_col, ...
    'VariableNames', {'Scenario', 'MG', 'Day', 'Time', 'Target_Variable', 'Value_L_s'});

fprintf('\n');
disp(SummaryTable);

% Guardar en CSV para referencia externa si es necesario
% writetable(SummaryTable, 'Scenarios_Operational_Summary.csv');