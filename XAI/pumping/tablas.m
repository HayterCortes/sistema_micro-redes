%% Script: generar_tabla_Qp_Altruismo_MG1.m
% Auditoría numérica de pesos LIME para la variable de bombeo (Qp).
% Caso específico: Escenario B (Altruismo) - Microrred 1.
%--------------------------------------------------------------------------
clear; clc;

% Configuramos solo el escenario solicitado
scenarios = {'Altruismo'};
t_idx = 1; % Microrred 1
num_features = 16; 

fprintf('--- GENERANDO TABLA NUMÉRICA LIME: BOMBEO MG1 (ESCENARIO B) ---\n\n');

% Solo procesamos para la Microrred 1
weights_matrix = zeros(num_features, length(scenarios));
f_names = cell(num_features, 1);
found_any = false;

for s_idx = 1:length(scenarios)
    % Nombre del archivo generado por el script robusto de Altruismo
    filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scenarios{s_idx}, t_idx);
    
    if exist(filename, 'file')
        data = load(filename);
        f_names = data.feature_names';
        found_any = true;
        
        num_runs = length(data.all_explanations);
        temp_w = zeros(num_features, num_runs);
        
        for r = 1:num_runs
            expl_run = data.all_explanations{r};
            % Mapeo de pesos por nombre de característica
            map_w = containers.Map(expl_run(:,1), [expl_run{:,2}]);
            for f = 1:num_features
                temp_w(f, r) = map_w(data.feature_names{f});
            end
        end
        % Promediamos los pesos de todas las ejecuciones (NUM_RUNS)
        weights_matrix(:, s_idx) = mean(temp_w, 2);
    else
        weights_matrix(:, s_idx) = NaN;
    end
end

if found_any
    % Creación de la tabla MATLAB enfocada en Altruismo
    Table_Qp_Altruismo = table(f_names, weights_matrix(:,1), ...
        'VariableNames', {'Feature', 'Altruism_Weight'});
    
    % Ordenar por importancia absoluta para facilitar la interpretación del paper
    [~, idx_sort] = sort(abs(Table_Qp_Altruismo.Altruism_Weight), 'descend');
    Table_Qp_Altruismo = Table_Qp_Altruismo(idx_sort, :);

    % Salida por Command Window
    fprintf('### AUDITORÍA DE PESOS LIME: MG%d (ALTRUISMO) ###\n', t_idx);
    disp(Table_Qp_Altruismo);
    
    % Guardado en el Workspace
    assignin('base', 'Table_Qp_MG1_Altruismo', Table_Qp_Altruismo);
else
    fprintf('[!] Error: No se encontró el archivo %s.\n', filename);
end

fprintf('\n--- PROCESO FINALIZADO ---\n');