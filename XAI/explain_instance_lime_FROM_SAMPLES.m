%% --- Archivo: explain_instance_lime_FROM_SAMPLES.m ---
%
% Núcleo de LIME que recibe muestras PRE-GENERADAS.
% Ideal para variantes avanzadas (Latent LIME, Genetic LIME, etc.)
% donde la perturbación ocurre fuera de esta función.
%
% Entradas:
%   - X_samples_std: Matriz (NxM) de muestras perturbadas y estandarizadas.
%   - Y_labels_std:  Vector (Nx1) de predicciones normalizadas del modelo.
%   - X_origin_std:  Vector (1xM) del punto original estandarizado (referencia).
%   - kernel_width:  Ancho del kernel para ponderación.
%
% Salidas:
%   - feature_weights: Coeficientes beta.
%   - r2_score: Fidelidad local.
%--------------------------------------------------------------------------
function [feature_weights, r2_score] = explain_instance_lime_FROM_SAMPLES(X_samples_std, Y_labels_std, X_origin_std, kernel_width)

    num_samples = size(X_samples_std, 1);
    
    % --- 1. Calcular Distancias (Kernel) ---
    % Distancia Euclidiana entre cada muestra generada y el origen
    % Nota: X_origin_std es el centro.
    
    % Diferencia vectorizada
    diffs = bsxfun(@minus, X_samples_std, X_origin_std);
    dist_sq = sum(diffs.^2, 2);
    distances = sqrt(dist_sq);
    
    % Peso w_i
    weights = sqrt(exp(-(distances.^2) / (kernel_width^2)));
    
    % --- 2. Regresión Lineal Ponderada (WLS) ---
    % Modelo: Y ~ X * beta
    X_aug = [ones(num_samples, 1), X_samples_std];
    
    % Ajuste
    beta = lscov(X_aug, Y_labels_std, weights);
    feature_weights = beta(2:end); % Ignorar intercepto
    
    % --- 3. Cálculo de Fidelidad (R^2) ---
    Y_pred_linear = X_aug * beta;
    
    % Media ponderada real
    sum_w = sum(weights);
    mean_y_w = sum(weights .* Y_labels_std) / sum_w;
    
    % Sumas de cuadrados ponderadas
    residuals = Y_labels_std - Y_pred_linear;
    SSE_w = sum(weights .* (residuals.^2));
    SST_w = sum(weights .* ((Y_labels_std - mean_y_w).^2));
    
    if SST_w < 1e-10
        r2_score = 1.0;
    else
        r2_score = 1 - (SSE_w / SST_w);
    end
    
    % Protección numérica
    if r2_score < 0, r2_score = 0; end
end