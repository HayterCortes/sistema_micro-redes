%% --- Archivo: calculate_lime_pumping_3mg_with_AE.m ---
%
% Wrapper LIME para Q_p (Bombeo) + Autoencoder Denoising.
%
% CORRECCIÓN CRÍTICA: Ahora usa 'explain_instance_lime_FROM_SAMPLES'
% para aprovechar realmente los datos proyectados por el AE.
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_AE(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
    % --- 1. CARGAR AUTOENCODER ---
    persistent ae_model
    if isempty(ae_model)
        possible_paths = {'models/autoencoder_manifold.mat', '../models/autoencoder_manifold.mat', '../../models/autoencoder_manifold.mat'};
        found = '';
        for i=1:length(possible_paths)
            if isfile(possible_paths{i}), found = possible_paths{i}; break; end
        end
        if isempty(found), error('Modelo AE no encontrado.'); end
        loaded = load(found);
        ae_model = loaded.model_data;
    end

    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % Límites Físicos
    min_bounds = []; max_bounds = [];
    for i = 1:3
        mg_p = params.mg(i);
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        min_bounds = [min_bounds, min_b_local]; max_bounds = [max_bounds, max_b_local]; 
    end
    min_bounds = [min_bounds, 0.0]; max_bounds = [max_bounds, inf];
    
    % Parámetros
    alpha = 3.0; 
    strength = 0.25; 
    num_synthetic = 1000; 
    num_samples_lime = 1000; 
    kernel_width = 0.75 * sqrt(N_features);
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    fprintf('Iniciando LIME BOMBEO (AE Denoising CORRECTED)... ');
    
    for run_idx = 1:NUM_EXECUTIONS
        
        % --- A. GENERACIÓN Y PROYECCIÓN (AE Denoising) ---
        % 1. Ruido Bruto
        pareto_noise = rand(num_synthetic, N_features).^(-1/alpha);
        signs = sign(rand(num_synthetic, N_features) - 0.5);
        noise = pareto_noise .* signs;
        
        X_row = X_original(:)';
        scaled_noise = strength .* bsxfun(@times, noise, X_row);
        X_train_noisy = bsxfun(@plus, X_row, scaled_noise);
        
        % 2. Proyección al Manifold (Limpieza)
        X_norm = (X_train_noisy - ae_model.min_vals) ./ (ae_model.max_vals - ae_model.min_vals);
        X_projected_norm = predict(ae_model.net, X_norm')'; 
        X_projected = X_projected_norm .* (ae_model.max_vals - ae_model.min_vals) + ae_model.min_vals;
        
        % 3. Clipping final y asignación a X_samples
        X_samples = max(min(X_projected, max_bounds), min_bounds);
        
        % --- B. PREPARACIÓN DE DATOS ---
        mu_X = mean(X_samples, 1);
        sigma_X = std(X_samples, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0;
        
        % Estandarización
        X_samples_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X;
        
        % --- C. PREDICCIÓN (MPC) ---
        % IMPORTANTE: Calculamos Y aquí, usando los datos limpios (X_samples_std)
        Y_raw = predict_fn_mpc_wrapper_3mg_AE_adapter( ...
            X_samples_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
            
        mu_Y = mean(Y_raw);
        sigma_Y = std(Y_raw);
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_labels_std = (Y_raw - mu_Y) / sigma_Y;
        
        % --- D. LIME CORE (Nueva Función) ---
        % Pasamos los datos YA generados y YA limpiados.
        [weights, r2_val] = explain_instance_lime_FROM_SAMPLES( ...
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
function Y = predict_fn_mpc_wrapper_3mg_AE_adapter(X_std, mu, sigma, const, ctrl, min_b, max_b, target)
    Y = predict_fn_mpc_wrapper_pumping(X_std, mu, sigma, const, ctrl, min_b, max_b, target);
end