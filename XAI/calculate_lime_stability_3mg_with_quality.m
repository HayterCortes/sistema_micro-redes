%% --- Archivo: calculate_lime_stability_3mg_with_quality.m ---
%
% Wrapper LIME para Q_t (Intercambio) + Métricas de Calidad.
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_stability_3mg_with_quality(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
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
    alpha = 3.0; strength = 0.2; 
    num_synthetic = 1000; num_samples_lime = 1000; 
    kernel_width = 0.75 * sqrt(N_features);
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    fprintf('Iniciando LIME Q_t (con Calidad R2)...\n');
    
    for run_idx = 1:NUM_EXECUTIONS
        % A. Ruido Pareto
        pareto_noise = rand(num_synthetic, N_features).^(-1/alpha);
        signs = sign(rand(num_synthetic, N_features) - 0.5);
        noise = pareto_noise .* signs;
        
        X_row = X_original(:)';
        scaled_noise = strength .* bsxfun(@times, noise, X_row);
        X_train = bsxfun(@plus, X_row, scaled_noise);
        X_train = max(min(X_train, max_bounds), min_bounds);
        
        % B. Scalers
        mu_X = mean(X_train, 1);
        sigma_X = std(X_train, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        X_train_std = bsxfun(@rdivide, bsxfun(@minus, X_train, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % C. Función Predictora (Q_t Wrapper)
        predict_raw = @(X_s) predict_fn_mpc_wrapper_3mg( ...
            X_s, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        Y_train = feval(predict_raw, X_train_std);
        
        mu_Y = mean(Y_train);
        sigma_Y = std(Y_train);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        predict_final = @(X_s) (feval(predict_raw, X_s) - mu_Y) / sigma_Y;
        
        % D. LIME Core (NUEVO: Captura R2)
        [weights, r2_val] = explain_instance_lime_with_quality( ...
            X_orig_std, predict_final, num_samples_lime, kernel_width, alpha);
        
        % Almacenar métricas
        r2_history(run_idx) = r2_val;
        
        % Guardar explicación
        expl_list = cell(N_features, 2);
        for f = 1:N_features
            expl_list{f, 1} = feature_names{f};
            expl_list{f, 2} = weights(f);
        end
        all_explanations{run_idx} = expl_list;
    end
    
    % Empaquetar estadísticas
    lime_stats.R2_mean = mean(r2_history);
    lime_stats.R2_history = r2_history;
    lime_stats.R2_min = min(r2_history);
end