function [p_dem_pred, p_gen_pred, q_dem_pred] = generar_predicciones_TS(hist_data, N)
    % GENERAR_PREDICCIONES_TS (Versión Optimizada)
    % Realiza la predicción recursiva a N pasos utilizando modelos Takagi-Sugeno.
    % Optimización: Inlining de la inferencia difusa y vectorización de reglas.
    
    % Cargar modelos una sola vez (Persistencia)
    persistent modelos_ts
    if isempty(modelos_ts)
        if isfile('models/modelos_prediccion_TS.mat')
            datos_cargados = load('models/modelos_prediccion_TS.mat');
            modelos_ts = datos_cargados.modelos_ts;
        else
            error('No se encuentra models/modelos_prediccion_TS.mat');
        end
    end
    
    % Inicialización de salidas
    num_mg = size(hist_data.P_dem, 2);
    p_dem_pred = zeros(N, num_mg);
    p_gen_pred = zeros(N, num_mg);
    q_dem_pred = zeros(N, num_mg);
    
    % Estructuras para iterar
    tipos = {'P_dem', 'P_gen', 'Q_dem'};
    datos_hist = {hist_data.P_dem, hist_data.P_gen, hist_data.Q_dem};
    
    % Matrices temporales para almacenar resultados por tipo
    salidas_temp = cell(1, 3);
    
    % --- Bucle Principal ---
    for j = 1:3 % Por tipo de variable (Demanda, Gen, Agua)
        tipo = tipos{j};
        preds_matriz = zeros(N, num_mg); % Pre-alloc
        
        for i = 1:num_mg % Por micro-red
            historia = datos_hist{j}(:, i);
            nombre_mod = sprintf('mg%d_%s_ts', i, tipo);
            
            % Extraer parámetros del struct para acceso rápido (evita lookups en bucle)
            modelo = modelos_ts.(nombre_mod);
            centers = modelo.centers;   % [n_rules x n_lags]
            sigmas  = modelo.sigmas;    % [n_rules x n_lags]
            thetas  = modelo.thetas;    % [n_lags+1 x n_rules]
            lags    = modelo.num_regresores;
            n_rules = size(centers, 1);
            
            % Validación de historia
            if length(historia) < lags
                error('Historia insuficiente (MG%d %s). Req: %d, Disp: %d', i, tipo, lags, length(historia));
            end
            
            % Vector de entrada inicial (fila)
            curr_input = historia(end-lags+1:end)'; 
            
            % Bucle Recursivo de Predicción (Cuello de botella optimizado)
            predicciones_i = zeros(N, 1);
            
            for k = 1:N
                % --- INFERENCIA DIFUSA VECTORIZADA (High Performance) ---
                
                % 1. Cálculo de Pertenencia (Membership)
                % MATLAB R2016b+ soporta expansión implícita (Broadcasting)
                % (Rules x Lags) - (1 x Lags) operan elemento a elemento por filas
                diff_sq = (centers - curr_input).^2; 
                
                % Gaussiana multidimensional: exp(-dist^2 / 2sigma^2)
                mu_matrix = exp(-diff_sq ./ (2 * sigmas.^2));
                
                % Producto de las dimensiones (AND lógico difuso) -> Grado de activación w
                w = prod(mu_matrix, 2); % Vector columna [n_rules x 1]
                
                % 2. Normalización (Firing Strength)
                sum_w = sum(w);
                if sum_w < 1e-12, sum_w = 1e-12; end % Evitar NaN
                beta = w / sum_w; % [n_rules x 1]
                
                % 3. Evaluación de Consecuentes (Modelos Lineales)
                % Entrada aumentada con bias: [1, x1, x2...]
                X_aug = [1, curr_input]; 
                
                % Cálculo simultáneo de todas las reglas:
                % (1 x Lags+1) * (Lags+1 x Rules) = (1 x Rules)
                y_local = X_aug * thetas; 
                
                % 4. Defusificación (Promedio Ponderado)
                % Producto punto entre salidas locales y pesos normalizados
                pred_val = y_local * beta; 
                
                % --- ACTUALIZACIÓN RECURSIVA ---
                predicciones_i(k) = pred_val;
                
                % Desplazar ventana: eliminar el dato más viejo, agregar predicción
                curr_input = [curr_input(2:end), pred_val];
            end
            
            preds_matriz(:, i) = predicciones_i;
        end
        salidas_temp{j} = preds_matriz;
    end
    
    % Asignar a variables de salida
    p_dem_pred = salidas_temp{1};
    p_gen_pred = salidas_temp{2};
    q_dem_pred = salidas_temp{3};
end