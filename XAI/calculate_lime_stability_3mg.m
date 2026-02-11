%% --- Archivo: calculate_lime_stability_3mg.m ---
%
% Ejecuta LIME para el sistema cooperativo 3-MG (Versión MEAN/Stability).
% Usa perturbación PARETO explícita (Alpha=3.0) y devuelve estadísticas R^2.
%
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_stability_3mg(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)

    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % Construir Límites (Bounds)
    min_bounds = []; max_bounds = [];
    for i = 1:3
        mg_p = params.mg(i);
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        min_bounds = [min_bounds, min_b_local]; 
        max_bounds = [max_bounds, max_b_local]; 
    end
    min_bounds = [min_bounds, 0.0]; max_bounds = [max_bounds, inf]; % V_aq
    
    % --- PARÁMETROS DEL PAPER (PARETO) ---
    alpha = 3.0; 
    strength = 0.2; 
    num_synthetic = 1000;
    kernel_width = 0.75 * sqrt(N_features);
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    % fprintf('Iniciando LIME 3-MG (Target: Q_t MG%d)...\n', TARGET_MG_IDX);

    for run_idx = 1:NUM_EXECUTIONS
        % t_start = tic;
        
        % A. GENERACIÓN DE RUIDO PARETO (Lógica Exacta del Paper)
        % Formula: x * (1 + strength * sign * rand^(-1/alpha))
        pareto_noise = rand(num_synthetic, N_features).^(-1/alpha);
        signs = sign(rand(num_synthetic, N_features) - 0.5);
        noise = pareto_noise .* signs;
        
        X_row = X_original(:)';
        scaled_noise = strength .* bsxfun(@times, noise, X_row);
        
        % Aplicar ruido
        X_train = bsxfun(@plus, X_row, scaled_noise);
        
        % Clipping (Respetar límites físicos)
        X_train = max(min(X_train, max_bounds), min_bounds);
        
        % B. Estandarización Local (Para LIME)
        mu_X = mean(X_train, 1);
        sigma_X = std(X_train, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        X_train_std = bsxfun(@rdivide, bsxfun(@minus, X_train, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % C. Predicción (Usando el Wrapper antiguo de 16 vars)
        % Nota: Calculamos Y explícitamente aquí para pasarlo al núcleo ROBUST
        Y_train = predict_fn_mpc_wrapper_3mg( ...
            X_train_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        mu_Y = mean(Y_train);
        sigma_Y = std(Y_train);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_train_std = (Y_train - mu_Y) / sigma_Y;
        
        % D. LIME Core (ROBUSTO - Calcula R2)
        % Usamos el mismo núcleo matemático que los scripts nuevos para consistencia
        [weights, r2_val] = explain_instance_lime_ROBUST( ...
            X_train_std, Y_train_std, X_orig_std, kernel_width);
            
        % E. Guardar Resultados
        r2_history(run_idx) = r2_val;
        
        expl_list = cell(N_features, 2);
        for f = 1:N_features
            expl_list{f, 1} = feature_names{f};
            expl_list{f, 2} = weights(f);
        end
        all_explanations{run_idx} = expl_list;
        
        % fprintf('  Run %d/%d completado. R2=%.4f\n', run_idx, NUM_EXECUTIONS, r2_val);
    end
    
    % F. Estadísticas Finales
    lime_stats.R2_mean = mean(r2_history);
    lime_stats.R2_history = r2_history;
end