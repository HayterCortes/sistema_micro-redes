%% --- Archivo: explain_instance_lime_with_quality.m ---
%
% Implementación de LIME con cálculo de FIDELIDAD (R^2).
% 
% Entradas:
%   - X_original_std: Instancia estandarizada.
%   - predict_fn_handle: Función de predicción caja negra.
%   - num_samples: Número de perturbaciones.
%   - kernel_width: Ancho del kernel.
%   - alpha: Parámetro Pareto.
%
% Salidas:
%   - feature_weights: Coeficientes de la regresión lineal (importancia).
%   - r2_score: Coeficiente de determinación (Calidad de la explicación).
%--------------------------------------------------------------------------
function [feature_weights, r2_score] = explain_instance_lime_with_quality(X_original_std, predict_fn_handle, num_samples, kernel_width, alpha)
    
    N_features = length(X_original_std);
    
    % --- 1. Generación de Perturbaciones (Pareto) ---
    pareto_noise = rand(num_samples, N_features).^(-1/alpha);
    signs = sign(rand(num_samples, N_features) - 0.5);
    X_perturbed_std = pareto_noise .* signs; 
    
    % --- 2. Predicciones Caja Negra (Complex Model) ---
    Y_true = feval(predict_fn_handle, X_perturbed_std); % (Nx1)
    
    % --- 3. Pesos del Kernel (Proximidad) ---
    diffs_sq = X_perturbed_std.^2;
    distances = sqrt(sum(diffs_sq, 2));
    % Peso w_i para cada muestra
    weights = sqrt(exp(-(distances.^2) / (kernel_width^2))); 
    
    % --- 4. Regresión Lineal Ponderada (WLS) ---
    % Modelo Explicativo: Y ~ X * beta
    X_aug = [ones(num_samples, 1), X_perturbed_std]; 
    
    % Ajuste WLS
    beta = lscov(X_aug, Y_true, weights);
    
    feature_weights = beta(2:end); % Descartar intercepto para LIME
    
    % --- 5. CÁLCULO DE CALIDAD (R^2 Ponderado) ---
    % Predicciones del modelo lineal (Explicación)
    Y_pred_linear = X_aug * beta;
    
    % Media ponderada de Y_true
    sum_w = sum(weights);
    mean_y_w = sum(weights .* Y_true) / sum_w;
    
    % Suma de Errores Cuadráticos (SSE) Ponderada
    residuals = Y_true - Y_pred_linear;
    SSE_w = sum(weights .* (residuals.^2));
    
    % Suma Total de Cuadrados (SST) Ponderada
    SST_w = sum(weights .* ((Y_true - mean_y_w).^2));
    
    % R^2 Score
    if SST_w < 1e-10
        r2_score = 1.0; % Caso trivial (varianza cero)
    else
        r2_score = 1 - (SSE_w / SST_w);
    end
    
    % Protección numérica
    if r2_score < 0, r2_score = 0; end 
end