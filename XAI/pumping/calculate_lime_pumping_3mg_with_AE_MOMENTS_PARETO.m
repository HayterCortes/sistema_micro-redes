%% --- Archivo: calculate_lime_pumping_3mg_with_AE_MOMENTS_PARETO.m ---
%
% Wrapper LIME para Q_p (Bombeo) + AE Denoising + MOMENTS FULL (34 vars).
% VERSIÓN: Perturbación PARETO EXACTA (antes de entrar al AE).
%
% Lógica:
% 1. Genera ruido Pareto multiplicativo.
% 2. Pasa el ruido por el Autoencoder (Denoising) para proyectar al manifold.
% 3. Predice usando el wrapper avanzado de Bombeo (MOMENTS).
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_AE_MOMENTS_PARETO(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
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
        if isempty(found), error('Modelo AE MOMENTS no encontrado.'); end
        loaded = load(found);
        ae_model = loaded.model_data;
    end
    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original); % 34 vars
    
    % --- 2. LÍMITES FÍSICOS (Bounds 34 vars) ---
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
    
    % B. Variables Nuevas (Max/Std)
    extra_vars = N_features - length(min_bounds);
    if extra_vars > 0
        min_bounds = [min_bounds, zeros(1, extra_vars)];
        max_bounds = [max_bounds, inf(1, extra_vars)];
    end
    
    % --- 3. PARÁMETROS LIME (PARETO) ---
    % Ajustados para replicar la intensidad del paper
    alpha = 3.0;        % Forma de la cola
    strength = 0.20;    % Intensidad (20%)
    num_synthetic = 1000; 
    kernel_width = 0.50 * sqrt(N_features); 
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    fprintf('Iniciando LIME Denoising (MOMENTS + PARETO)... ');
    
    for run_idx = 1:NUM_EXECUTIONS
        
        X_row = X_original(:)';
        
        % --- A. GENERACIÓN (PARETO EXACTA) ---
        % Fórmula paper: x * (1 + strength * sign * rand^(-1/alpha))
        
        % 1. Magnitud Pareto
        pareto_raw = rand(num_synthetic, N_features).^(-1/alpha);
        
        % 2. Signo Aleatorio
        signs = sign(rand(num_synthetic, N_features) - 0.5);
        
        % 3. Ruido Base
        noise = pareto_raw .* signs;
        
        % 4. Aplicación Multiplicativa
        scaled_noise = strength .* bsxfun(@times, noise, X_row);
        X_train_noisy = bsxfun(@plus, X_row, scaled_noise);
        
        % --- B. PROYECCIÓN (AE Denoising) ---
        % El AE recibe los datos "sucios" (Pareto) y los devuelve al manifold válido
        
        % 1. Normalizar (Escala del AE [0,1])
        X_norm = (X_train_noisy - ae_model.min_vals) ./ (ae_model.max_vals - ae_model.min_vals);
        
        % 2. Proyectar (Limpiar)
        X_projected_norm = predict(ae_model.net, X_norm')'; 
        
        % 3. Des-normalizar
        X_projected = X_projected_norm .* (ae_model.max_vals - ae_model.min_vals) + ae_model.min_vals;
        
        % 4. Clipping final (Bounds físicos del MPC)
        X_samples = max(min(X_projected, max_bounds), min_bounds);
        
        % --- C. PREPARACIÓN ---
        mu_X = mean(X_samples, 1);
        sigma_X = std(X_samples, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        X_samples_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % --- D. PREDICCIÓN (Wrapper Pumping MOMENTS) ---
        % Usamos el wrapper avanzado que entiende Max/Std y devuelve Q_p
        Y_raw = predict_fn_mpc_wrapper_pumping_MOMENTS( ...
            X_samples_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        mu_Y = mean(Y_raw);
        sigma_Y = std(Y_raw);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_labels_std = (Y_raw - mu_Y) / sigma_Y;
        
        % --- E. LIME CORE (ROBUSTO) ---
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