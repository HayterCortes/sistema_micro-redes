%% Script: generar_tablas_Qs.m
% Auditoría numérica de pesos LIME para la variable de intercambio (Qs).

clear; clc;
scenarios = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
mgs = [1, 2, 3];
num_features = 16; 

fprintf('--- GENERANDO TABLAS NUMÉRICAS LIME: INTERCAMBIO (Qs) ---\n\n');

for t_idx = mgs
    weights_matrix = zeros(num_features, length(scenarios));
    f_names = cell(num_features, 1);
    found_any = false;
    
    for s_idx = 1:length(scenarios)
        filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scenarios{s_idx}, t_idx);
        if exist(filename, 'file')
            data = load(filename);
            f_names = data.feature_names';
            found_any = true;
            
            num_runs = length(data.all_explanations);
            temp_w = zeros(num_features, num_runs);
            for r = 1:num_runs
                expl_run = data.all_explanations{r};
                map_w = containers.Map(expl_run(:,1), [expl_run{:,2}]);
                for f = 1:num_features
                    temp_w(f, r) = map_w(data.feature_names{f});
                end
            end
            weights_matrix(:, s_idx) = mean(temp_w, 2);
        else
            weights_matrix(:, s_idx) = NaN;
        end
    end
    
    if ~found_any, for i=1:num_features, f_names{i}=sprintf('Feature_%d',i); end; end
    
    % Creación de la tabla
    Table_Qs = table(f_names, weights_matrix(:,1), weights_matrix(:,2), weights_matrix(:,3), ...
        'VariableNames', {'Feature', 'GlobalPeak', 'Altruismo', 'DirectSatisfaction'});
    
    fprintf('### TABLA PESOS LIME: MG%d ###\n', t_idx);
    disp(Table_Qs);
    fprintf('\n');
    
    assignin('base', sprintf('Table_Qs_MG%d', t_idx), Table_Qs);
end