%% Script: generar_tablas_Qs.m
% Auditoría numérica de pesos LIME para la variable de intercambio (Qs).
%
% ACTUALIZACIÓN:
% - Incluye selección de TIPO_MODELO ('AR' o 'TS') para cargar el archivo correcto.
%--------------------------------------------------------------------------

clear; clc;

% --- CONFIGURACIÓN DE USUARIO ---
TIPO_MODELO = 'TS'; % <--- CAMBIAR A 'AR' O 'TS' SEGÚN LOS DATOS GENERADOS

scenarios = {'GlobalPeak', 'Altruismo', 'DirectSatisfaction'};
mgs = [1, 2, 3];
num_features = 16; 

fprintf('--- GENERANDO TABLAS NUMÉRICAS LIME (%s): INTERCAMBIO (Qs) ---\n\n', TIPO_MODELO);

for t_idx = mgs
    weights_matrix = zeros(num_features, length(scenarios));
    f_names = cell(num_features, 1);
    found_any = false;
    
    for s_idx = 1:length(scenarios)
        % Construcción del nombre de archivo incluyendo el TIPO_MODELO
        filename = sprintf('lime_Scenario_%s_%s_MG%d_MEAN.mat', scenarios{s_idx}, TIPO_MODELO, t_idx);
        
        if exist(filename, 'file')
            data = load(filename);
            f_names = data.feature_names';
            found_any = true;
            
            num_runs = length(data.all_explanations);
            temp_w = zeros(num_features, num_runs);
            
            for r = 1:num_runs
                expl_run = data.all_explanations{r};
                % Crear un mapa para asegurar que los pesos correspondan al feature correcto
                map_w = containers.Map(expl_run(:,1), [expl_run{:,2}]);
                for f = 1:num_features
                    if isKey(map_w, data.feature_names{f})
                        temp_w(f, r) = map_w(data.feature_names{f});
                    else
                        temp_w(f, r) = 0;
                    end
                end
            end
            % Promedio de las ejecuciones (Runs)
            weights_matrix(:, s_idx) = mean(temp_w, 2);
        else
            weights_matrix(:, s_idx) = NaN;
            fprintf('Advertencia: No se encontró %s\n', filename);
        end
    end
    
    % Si no se encontraron archivos, generar nombres genéricos para evitar error de tabla
    if ~found_any
        for i=1:num_features, f_names{i}=sprintf('Feature_%d',i); end
    end
    
    % Creación de la tabla
    Table_Qs = table(f_names, weights_matrix(:,1), weights_matrix(:,2), weights_matrix(:,3), ...
        'VariableNames', {'Feature', 'GlobalPeak', 'Altruismo', 'DirectSatisfaction'});
    
    fprintf('### TABLA PESOS LIME: MG%d (%s) ###\n', t_idx, TIPO_MODELO);
    disp(Table_Qs);
    fprintf('\n');
    
    % Guardar en el Workspace con nombre dinámico
    assignin('base', sprintf('Table_Qs_MG%d_%s', t_idx, TIPO_MODELO), Table_Qs);
end
fprintf('Proceso completado.\n');