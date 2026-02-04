function [p_dem_pred, p_gen_pred, q_dem_pred] = generar_predicciones_TS(hist_data, N)
    % GENERAR_PREDICCIONES_TS (Versión Robusta)
    % Realiza la predicción recursiva a N pasos utilizando modelos Takagi-Sugeno.
    % CORRECCIÓN: Búsqueda dinámica de la carpeta 'models'.
    
    persistent modelos_ts
    
    if isempty(modelos_ts)
        % --- LÓGICA DE BÚSQUEDA ROBUSTA ---
        fname = 'modelos_prediccion_TS.mat';
        
        % Posibles ubicaciones de la carpeta 'models'
        possible_paths = {
            fullfile('models', fname),           % Caso: Ejecución desde raíz
            fullfile('..', 'models', fname),     % Caso: Ejecución desde XAI
            fullfile('..', '..', 'models', fname) % Caso: Ejecución desde XAI/pumping
        };
        
        file_found = '';
        for i = 1:length(possible_paths)
            if isfile(possible_paths{i})
                file_found = possible_paths{i};
                break;
            end
        end
        
        if isempty(file_found)
            error('generar_predicciones_TS:NoModel', ...
                  'No se encuentra %s en ninguna ruta esperada.', fname);
        end
        
        % Cargar
        datos_cargados = load(file_found);
        modelos_ts = datos_cargados.modelos_ts;
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
            if ~isfield(modelos_ts, nombre_mod)
                 error('Modelo %s no encontrado en la estructura cargada.', nombre_mod);
            end
            
            modelo = modelos_ts.(nombre_mod);
            centers = modelo.centers;   % [n_rules x n_lags]
            sigmas  = modelo.sigmas;    % [n_rules x n_lags]
            thetas  = modelo.thetas;    % [n_lags+1 x n_rules]
            lags    = modelo.num_regresores;
            
            % Validación de historia
            if length(historia) < lags
                error('Historia insuficiente (MG%d %s). Req: %d, Disp: %d', i, tipo, lags, length(historia));
            end
            
            % Vector de entrada inicial (fila)
            curr_input = historia(end-lags+1:end)'; 
            
            % Bucle Recursivo de Predicción
            predicciones_i = zeros(N, 1);
            
            for k = 1:N
                % 1. Cálculo de Pertenencia (Membership)
                diff_sq = (centers - curr_input).^2; 
                mu_matrix = exp(-diff_sq ./ (2 * sigmas.^2));
                w = prod(mu_matrix, 2); % [n_rules x 1]
                
                % 2. Normalización
                sum_w = sum(w);
                if sum_w < 1e-12, sum_w = 1e-12; end 
                beta = w / sum_w; 
                
                % 3. Evaluación de Consecuentes
                X_aug = [1, curr_input]; 
                y_local = X_aug * thetas; 
                
                % 4. Defusificación
                pred_val = y_local * beta; 
                
                % --- ACTUALIZACIÓN RECURSIVA ---
                predicciones_i(k) = pred_val;
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