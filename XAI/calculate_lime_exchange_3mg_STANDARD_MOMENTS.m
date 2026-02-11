%% --- Archivo: calculate_lime_exchange_3mg_STANDARD_MOMENTS.m ---
% LIME STANDARD (Sin AE) para Q_t usando 34 variables.
function [lime_stats, all_explanations] = calculate_lime_exchange_3mg_STANDARD_MOMENTS(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % Parámetros Standard (Pareto es lo clásico, pero usaremos Gauss para ser justos en comparación)
    strength = 0.10; 
    num_samples = 1000;
    kernel_width = 0.75 * sqrt(N_features);
    
    % Bounds (Para clipping físico básico)
    min_b=[]; max_b=[];
    for i=1:3, mg=params.mg(i); min_b=[min_b, mg.SoC_min, 0,0,0,0]; max_b=[max_b, mg.SoC_max, mg.V_max, inf,inf,inf]; end
    min_b=[min_b, 0]; max_b=[max_b, inf];
    if N_features > 16, min_b=[min_b, zeros(1,N_features-16)]; max_b=[max_b, inf(1,N_features-16)]; end
    
    r2_hist = zeros(NUM_EXECUTIONS,1); all_explanations = cell(NUM_EXECUTIONS,1);
    
    for run = 1:NUM_EXECUTIONS
        % A. Generación (Perturbación Directa)
        noise = randn(num_samples, N_features);
        X_orig_row = X_original(:)';
        eps_base = 1e-4 * mean(abs(X_orig_row(X_orig_row>1e-3))); if isnan(eps_base), eps_base=1e-4; end
        
        % X_pert = X + noise (Sin proyección AE)
        X_samples = bsxfun(@plus, X_orig_row, strength .* (bsxfun(@times, noise, X_orig_row) + eps_base .* noise));
        X_samples = max(min(X_samples, max_b), min_b);
        
        % B. Estandarización Local para LIME
        mu_X = mean(X_samples, 1); sigma_X = std(X_samples, 0, 1); sigma_X(sigma_X<1e-10)=1;
        X_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_orig_row - mu_X) ./ sigma_X;
        
        % C. Predicción Standard
        Y_raw = predict_fn_mpc_wrapper_exchange_STANDARD(X_samples, estado.constants, controller_obj, TARGET_MG_IDX);
        
        mu_Y = mean(Y_raw); sigma_Y = std(Y_raw); if sigma_Y<1e-10, sigma_Y=1; end
        Y_std = (Y_raw - mu_Y) / sigma_Y;
        
        % D. LIME Core
        [w, r2] = explain_instance_lime_ROBUST(X_std, Y_std, X_orig_std, kernel_width);
        
        r2_hist(run) = r2;
        expl = cell(N_features, 2);
        for f=1:N_features, expl{f,1}=feature_names{f}; expl{f,2}=w(f); end
        all_explanations{run} = expl;
    end
    lime_stats.R2_mean = mean(r2_hist);
end