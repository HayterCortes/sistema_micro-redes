%% --- Archivo: calculate_lime_pumping_3mg_with_AE_MOMENTS.m ---
%
% Wrapper LIME para Q_p (Bombeo) + AE Denoising + MOMENTS FULL (34 vars).
%
% CAMBIOS CLAVE:
% 1. Carga el modelo autoencoder_manifold_MOMENTS.mat (ahora con 34 vars).
% 2. Ajusta los Bounds para 34 variables (Base + MaxP/Q + StdP/Q).
% 3. Usa Ruido GAUSSIANO (Normal) y llama al núcleo LIME ROBUSTO.
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_AE_MOMENTS(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
    % --- 1. CARGAR AUTOENCODER (MOMENTS) ---
    persistent ae_model
    if isempty(ae_model)
        possible_paths = {'models/autoencoder_manifold_MOMENTS.mat', ...
                          '../models/autoencoder_manifold_MOMENTS.mat', ...
                          '../../models/autoencoder_manifold_MOMENTS.mat'};
        found = '';
        for i=1:length(possible_paths)
            if isfile(possible_paths{i}), found = possible_paths{i}; break; end
        end
        if isempty(found), error('Modelo AE MOMENTS no encontrado. Entrénalo primero.'); end
        loaded = load(found);
        ae_model = loaded.model_data;
    end

    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original); % Ahora debería ser 34
    
    % --- 2. LÍMITES FÍSICOS (Adaptado para 34 vars) ---
    % Estructura esperada (34 vars):
    % [Base(16), Max_Pgen(3), Max_Pdem(3), Max_Qdem(3), Std_Pgen(3), Std_Pdem(3), Std_Qdem(3)]
    
    min_bounds = []; max_bounds = [];
    
    % A. Variables Base (16)
    for i = 1:3
        mg_p = params.mg(i);
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        min_bounds = [min_bounds, min_b_local]; 
        max_bounds = [max_bounds, max_b_local]; 
    end
    min_bounds = [min_bounds, 0.0]; max_bounds = [max_bounds, inf]; % V_aq
    
    % B. Variables Nuevas (18 vars: Max y Std de Pgen, Pdem, Qdem)
    % Todas son magnitudes físicas positivas (potencia o caudal) o desviaciones.
    % Límite inferior 0, superior inf.
    extra_vars = N_features - length(min_bounds);
    if extra_vars > 0
        min_bounds = [min_bounds, zeros(1, extra_vars)];
        max_bounds = [max_bounds, inf(1, extra_vars)];
    end
    
    % --- 3. PARÁMETROS LIME (Ajuste Matemático) ---
    % Recomendación Experto: Gaussiano + Radio controlado
    strength = 0.10;  % 10% de perturbación
    num_synthetic = 1000; 
    num_samples_lime = 1000; 
    kernel_width = 0.50 * sqrt(N_features); 
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    fprintf('Iniciando LIME Denoising (MOMENTS FULL + GAUSS)... ');
    
    for run_idx = 1:NUM_EXECUTIONS
        
        % --- A. GENERACIÓN (Ruido Normal) ---
        noise = randn(num_synthetic, N_features); 
        
        X_row = X_original(:)';
        
        % Truco de escala robusto
        epsilon_base = 1e-4 * mean(abs(X_row(X_row > 1e-3))); 
        if isnan(epsilon_base) || epsilon_base==0, epsilon_base = 1e-4; end
        
        scaled_noise = strength .* (bsxfun(@times, noise, X_row) + epsilon_base .* noise);
        X_train_noisy = bsxfun(@plus, X_row, scaled_noise);
        
        % --- B. PROYECCIÓN (AE Denoising) ---
        % 1. Normalizar
        X_norm = (X_train_noisy - ae_model.min_vals) ./ (ae_model.max_vals - ae_model.min_vals);
        
        % 2. Proyectar (Limpiar)
        X_projected_norm = predict(ae_model.net, X_norm')'; 
        
        % 3. Des-normalizar
        X_projected = X_projected_norm .* (ae_model.max_vals - ae_model.min_vals) + ae_model.min_vals;
        
        % 4. Clipping final
        X_samples = max(min(X_projected, max_bounds), min_bounds);
        
        % --- C. PREPARACIÓN ---
        mu_X = mean(X_samples, 1);
        sigma_X = std(X_samples, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        X_samples_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % --- D. PREDICCIÓN (MPC con MOMENTOS) ---
        % Llamamos al adaptador que usa el wrapper avanzado
        Y_raw = predict_fn_mpc_wrapper_pumping_adapter( ...
            X_samples_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        mu_Y = mean(Y_raw);
        sigma_Y = std(Y_raw);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_labels_std = (Y_raw - mu_Y) / sigma_Y;
        
        % --- E. LIME CORE (ROBUSTO) ---
        % Usamos la versión robusta con filtro de pesos pequeños
        [weights, r2_val] = explain_instance_lime_ROBUST( ...
            X_samples_std, Y_labels_std, X_orig_std, kernel_width);
        
        r2_history(run_idx) = r2_val;
        
        expl_list = cell(N_features, 2);
        for f = 1:N_features
            expl_list{f, 1} = feature_names{f};
            expl_list{f, 2} = weights(f);
        end
        all_explanations{run_idx} = expl_list;
    end
    
    lime_stats.R2_mean = mean(r2_history);
    lime_stats.R2_history = r2_history;
    lime_stats.R2_min = min(r2_history);
    fprintf('Done.\n');
end

% Helper local
function Y = predict_fn_mpc_wrapper_pumping_adapter(X_std, mu, sigma, const, ctrl, min_b, max_b, target)
    % IMPORTANTE: Debe llamar a la versión MOMENTS del wrapper de predicción
    % Asegúrate de tener ese archivo actualizado también.
    Y = predict_fn_mpc_wrapper_pumping_MOMENTS(X_std, mu, sigma, const, ctrl, min_b, max_b, target);
end