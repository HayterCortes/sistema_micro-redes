%% --- Archivo: explain_instance_lime_ROBUST.m ---
%
% Núcleo de LIME ROBUSTO (Weighted Least Squares).
% Recibe muestras pre-generadas y calcula la regresión lineal local.
%
% MEJORA: Filtra pesos infinitesimales para "agudizar" la explicación 
% en zonas de picos y discontinuidades (Recomendación Matemática).
%
% Entradas:
%   - X_samples_std: Matriz (NxM) muestras estandarizadas.
%   - Y_labels_std:  Vector (Nx1) predicciones estandarizadas.
%   - X_origin_std:  Vector (1xM) origen estandarizado.
%   - kernel_width:  Ancho del kernel.
%
% Salidas:
%   - feature_weights: Coeficientes beta (importancia).
%   - r2_score: Fidelidad local (R^2 ponderado).
%--------------------------------------------------------------------------
function [feature_weights, r2_score] = explain_instance_lime_ROBUST(X_samples_std, Y_labels_std, X_origin_std, kernel_width)
    
    num_samples = size(X_samples_std, 1);
    
    % --- 1. Calcular Distancias (Kernel Exponencial) ---
    % Distancia Euclidiana al origen
    diffs = bsxfun(@minus, X_samples_std, X_origin_std);
    dist_sq = sum(diffs.^2, 2);
    distances = sqrt(dist_sq);
    
    % Calcular Pesos (Kernel Gaussiano)
    weights = sqrt(exp(-(distances.^2) / (kernel_width^2)));
    
    % --- MEJORA MATEMÁTICA: Hard Thresholding ---
    % Eliminar influencia de puntos muy lejanos (ruido numérico)
    weights(weights < 1e-6) = 0;
    
    % Protección: Si el kernel es muy estrecho y todo es 0, usar uniforme local
    if sum(weights) < 1e-10
        weights(:) = 1.0; 
    end
    
    % --- 2. Regresión Lineal Ponderada (WLS) ---
    % Modelo: Y = beta0 + beta1*X1 + ... + error
    X_aug = [ones(num_samples, 1), X_samples_std];
    
    % Ajuste (lscov maneja pesos cero ignorando esas filas correctamente)
    [beta, ~] = lscov(X_aug, Y_labels_std, weights);
    
    feature_weights = beta(2:end); % Descartar intercepto (beta0)
    
    % --- 3. Cálculo de Fidelidad (R^2 Ponderado) ---
    Y_pred_linear = X_aug * beta;
    
    % Media ponderada real (solo de los puntos activos)
    sum_w = sum(weights);
    mean_y_w = sum(weights .* Y_labels_std) / sum_w;
    
    % Sumas de cuadrados ponderadas
    residuals = Y_labels_std - Y_pred_linear;
    
    SSE_w = sum(weights .* (residuals.^2));
    SST_w = sum(weights .* ((Y_labels_std - mean_y_w).^2));
    
    % Cálculo R2 con protección de varianza cero
    if SST_w < 1e-10
        % Si la varianza ponderada es 0 (Y es constante en el vecindario),
        % el modelo lineal (plano) explica perfectamente esa constante.
        r2_score = 1.0;
    else
        r2_score = 1 - (SSE_w / SST_w);
    end
    
    % Protección final numérica
    if r2_score < 0, r2_score = 0; end
end