%% --- Archivo: calculate_lime_pumping_3mg_STANDARD_MOMENTS.m ---
%
% LIME STANDARD (Sin AE) para variable de BOMBEO (Q_p).
% Utiliza 34 variables (Base + Momentos).
% Perturbación: GAUSSIANA (Normal).
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_pumping_3mg_STANDARD_MOMENTS(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % --- 1. DEFINICIÓN DE LÍMITES FÍSICOS (BOUNDS) ---
    min_bounds = []; max_bounds = [];
    
    % A. Variables Base (16 vars: SoC, Vol Tanque, Vol Acuífero, etc.)
    for i = 1:3
        mg_p = params.mg(i);
        % Estructura local: [SoC, V_tank, P_dem, P_gen, Q_dem]
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        
        min_bounds = [min_bounds, min_b_local]; 
        max_bounds = [max_bounds, max_b_local]; 
    end
    % Variable 16: Volumen Acuífero
    min_bounds = [min_bounds, 0.0]; 
    max_bounds = [max_bounds, inf]; 
    
    % B. Variables Momentos (18 vars: Max y Std)
    % Todas son magnitudes físicas >= 0
    extra_vars = N_features - length(min_bounds);
    if extra_vars > 0
        min_bounds = [min_bounds, zeros(1, extra_vars)];
        max_bounds = [max_bounds, inf(1, extra_vars)];
    end
    
    % --- 2. PARÁMETROS LIME (GAUSSIANO) ---
    strength = 0.10;  % Intensidad de ruido (10%)
    num_synthetic = 1000; 
    kernel_width = 0.50 * sqrt(N_features);
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    % --- 3. BUCLE DE EJECUCIÓN ---
    for run_idx = 1:NUM_EXECUTIONS
        
        % A. Generación de Perturbaciones (Gaussiana)
        noise = randn(num_synthetic, N_features);
        
        X_row = X_original(:)';
        
        % Factor de seguridad para evitar ceros absolutos
        epsilon_base = 1e-4 * mean(abs(X_row(X_row > 1e-3))); 
        if isnan(epsilon_base) || epsilon_base == 0, epsilon_base = 1e-4; end
        
        % Aplicación del ruido (Multiplicativo + Aditivo pequeño)
        % X_pert = X + Strength * (X*Ruido + Epsilon*Ruido)
        scaled_noise = strength .* (bsxfun(@times, noise, X_row) + epsilon_base .* noise);
        X_samples = bsxfun(@plus, X_row, scaled_noise);
        
        % Clipping (Respetar física)
        X_samples = max(min(X_samples, max_bounds), min_bounds);
        
        % B. Estandarización Local
        mu_X = mean(X_samples, 1);
        sigma_X = std(X_samples, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0; % Evitar división por cero
        
        X_samples_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % C. Predicción (Usando Wrapper de BOMBEO MOMENTS)
        % Nota: No usamos AE, pasamos las muestras "físicas" directo al wrapper
        % Pero el wrapper espera recibir datos y des-estandarizarlos.
        % Así que le pasamos los datos estandarizados y sus mu/sigma.
        Y_raw = predict_fn_mpc_wrapper_pumping_MOMENTS( ...
            X_samples_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
        
        % Estandarizar salida Y
        mu_Y = mean(Y_raw);
        sigma_Y = std(Y_raw);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_labels_std = (Y_raw - mu_Y) / sigma_Y;
        
        % D. LIME Core (Regresión Ponderada)
        [weights, r2_val] = explain_instance_lime_ROBUST( ...
            X_samples_std, Y_labels_std, X_orig_std, kernel_width);
        
        r2_history(run_idx) = r2_val;
        
        % Formatear Explicación
        expl_list = cell(N_features, 2);
        for f = 1:N_features
            expl_list{f, 1} = feature_names{f};
            expl_list{f, 2} = weights(f);
        end
        all_explanations{run_idx} = expl_list;
    end
    
    lime_stats.R2_mean = mean(r2_history);
end