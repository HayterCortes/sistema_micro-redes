%% --- Archivo: calculate_lime_stability_3mg_with_quality.m ---
%
% Wrapper LIME para Q_t (Intercambio) + Métricas de Calidad.
% MODIFICACIÓN: Soporta selección de perturbación (GAUSSIAN / PARETO).
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_stability_3mg_with_quality(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX, PERTURBATION_TYPE)
    
    if nargin < 6, PERTURBATION_TYPE = 'PARETO'; end % Default
    
    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % Construir Límites
    min_bounds = []; max_bounds = [];
    for i = 1:3
        mg_p = params.mg(i);
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        min_bounds = [min_bounds, min_b_local]; 
        max_bounds = [max_bounds, max_b_local]; 
    end
    min_bounds = [min_bounds, 0.0]; max_bounds = [max_bounds, inf];
    
    % Parámetros LIME
    num_synthetic = 1000; num_samples_lime = 1000; 
    
    % Ajuste de parámetros según tipo de perturbación
    if strcmp(PERTURBATION_TYPE, 'PARETO')
        alpha = 3.0; 
        strength = 0.2;
        kernel_width = 0.75 * sqrt(N_features);
    else % GAUSSIAN
        alpha = 0; % No usado
        strength = 0.1;
        kernel_width = 0.50 * sqrt(N_features);
    end
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    % fprintf('Iniciando LIME Q_t (%s)...\n', PERTURBATION_TYPE);
    
    for run_idx = 1:NUM_EXECUTIONS
        
        X_row = X_original(:)';
        
        % --- A. GENERACIÓN DE RUIDO ---
        if strcmp(PERTURBATION_TYPE, 'PARETO')
            % Lógica Pareto (Multiplicativa con cola pesada)
            pareto_noise = rand(num_synthetic, N_features).^(-1/alpha);
            signs = sign(rand(num_synthetic, N_features) - 0.5);
            noise = pareto_noise .* signs;
            scaled_noise = strength .* bsxfun(@times, noise, X_row);
            X_train = bsxfun(@plus, X_row, scaled_noise);
            
        else % GAUSSIAN
            % Lógica Gaussiana (Aditiva/Multiplicativa Mixta)
            noise = randn(num_synthetic, N_features);
            epsilon_base = 1e-4 * mean(abs(X_row(X_row > 1e-3))); 
            if isnan(epsilon_base) || epsilon_base==0, epsilon_base=1e-4; end
            
            % X_pert = X + strength * (X * noise + epsilon * noise)
            X_train = bsxfun(@plus, X_row, strength .* (bsxfun(@times, noise, X_row) + epsilon_base .* noise));
        end
        
        % Clipping
        X_train = max(min(X_train, max_bounds), min_bounds);
        
        % --- B. Scalers ---
        mu_X = mean(X_train, 1);
        sigma_X = std(X_train, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        X_train_std = bsxfun(@rdivide, bsxfun(@minus, X_train, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % --- C. Predicción ---
        Y_train = predict_fn_mpc_wrapper_3mg( ...
            X_train_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        mu_Y = mean(Y_train);
        sigma_Y = std(Y_train);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_train_std = (Y_train - mu_Y) / sigma_Y;
        
        % --- D. LIME Core (ROBUSTO) ---
        % Usamos explain_instance_lime_ROBUST (el universal) para ambos casos
        [weights, r2_val] = explain_instance_lime_ROBUST( ...
            X_train_std, Y_train_std, X_orig_std, kernel_width);
        
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
end