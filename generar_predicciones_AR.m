% --- generar_predicciones_AR.m (Versión Corregida) ---
function [p_dem_pred, p_gen_pred, q_dem_pred] = generar_predicciones_AR(hist_data, N)
    % Genera predicciones a N pasos usando los modelos AR entrenados.
    
    % --- CAMBIO CLAVE ---
    % Se cargan los modelos directamente desde el archivo .mat
    load('models/modelos_prediccion_AR.mat', 'modelos');

    num_mg = size(hist_data.P_dem, 2);
    p_dem_pred = zeros(N, num_mg);
    p_gen_pred = zeros(N, num_mg);
    q_dem_pred = zeros(N, num_mg);
    
    tipos_de_senal = {'P_dem', 'P_gen', 'Q_dem'};
    datos_completos = {hist_data.P_dem, hist_data.P_gen, hist_data.Q_dem};
    predicciones_out = {p_dem_pred, p_gen_pred, q_dem_pred};
    
    for i = 1:num_mg
        for j = 1:length(tipos_de_senal)
            tipo_actual = tipos_de_senal{j};
            historia_actual_completa = datos_completos{j}(:, i);
            
            nombre_modelo = sprintf('mg%d_%s_ar', i, tipo_actual);
            modelo_actual = modelos.(nombre_modelo);
            theta = modelo_actual.theta;
            num_lags = modelo_actual.num_regresores;
            
            historia_recursiva = historia_actual_completa(end-num_lags+1:end)';
            
            predicciones_temp = zeros(N, 1);
            for step = 1:N
                entrada_actual = [1, historia_recursiva];
                predicciones_temp(step) = entrada_actual * theta;
                historia_recursiva = [historia_recursiva(2:end), predicciones_temp(step)];
            end
            
            predicciones_out{j}(:, i) = predicciones_temp;
        end
    end
    
    p_dem_pred = predicciones_out{1};
    p_gen_pred = predicciones_out{2};
    q_dem_pred = predicciones_out{3};
end