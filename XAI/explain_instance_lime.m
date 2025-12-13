%% --- Archivo: explain_instance_lime.m ---
%
% Implementación de LIME (Local Interpretable Model-agnostic Explanations)
% desde cero para un modelo de regresión.
% 
% ESTA FUNCIÓN ES REQUERIDA POR: calculate_lime_stability_3mg.m
%

function [feature_weights] = explain_instance_lime(X_original_std, predict_fn_handle, num_samples, kernel_width, alpha)
    
    % X_original_std: La instancia (fila) a explicar, ya estandarizada.
    % predict_fn_handle: Handle a la función que toma (N_samples, N_features)
    %                    y devuelve (N_samples, 1) de predicciones.
    % num_samples: Nro de perturbaciones a generar (LIME 'num_samples').
    % kernel_width: Ancho del kernel exponencial.
    % alpha: Parámetro de forma de la distribución de Pareto.
    
    N_features = length(X_original_std);
    
    % --- ESTRATEGIA DE PERTURBACIÓN (PARETO) ---
    % Generar perturbaciones de cola pesada para mejor estabilidad
    
    % 1. Generar magnitud (Pareto)
    pareto_noise = rand(num_samples, N_features).^(-1/alpha);
    
    % 2. Generar dirección (Signo aleatorio)
    signs = sign(rand(num_samples, N_features) - 0.5);
    
    % 3. Ruido final
    X_perturbed_std = pareto_noise .* signs; 
    
    % --- PASO 2: Obtener Predicciones de la Caja Negra ---
    % Se evalúa el modelo complejo (MPC Wrapper) con los datos perturbados
    Y_perturbed = feval(predict_fn_handle, X_perturbed_std); % (N_samples x 1)
    
    % --- PASO 3: Calcular Distancias y Pesos (Kernel) ---
    
    % 1. Calcular distancia Euclideana entre la instancia original y las perturbadas
    %    Dado que X_perturbed_std es ruido sumado a cero (centrado), 
    %    y X_original_std se asume el origen en el espacio latente local,
    %    la distancia es simplemente la norma del ruido.
    %    Sin embargo, para ser rigurosos con la formulación LIME:
    
    %    En esta implementación, X_perturbed_std ya representa la DEVIACIÓN
    %    desde el origen estandarizado.
    
    diffs_sq = X_perturbed_std.^2;
    distances_sq = sum(diffs_sq, 2); % (Mx1)
    distances = sqrt(distances_sq);
    
    % 2. Kernel exponencial para asignar pesos (importancia local)
    kernel_sigma = kernel_width;
    weights = sqrt(exp(-(distances.^2) / (kernel_sigma^2))); 
    
    % --- PASO 4: Resolver Regresión Lineal Ponderada (WLS) ---
    
    % Modelo lineal local: Y = X * beta
    % Agregamos columna de unos para el intercepto (bias)
    X_aug = [ones(num_samples, 1), X_perturbed_std]; 
    
    % Resolver A*x = b con pesos W (Weighted Least Squares)
    % MATLAB lscov maneja esto eficientemente
    beta = lscov(X_aug, Y_perturbed, weights);
    
    % beta(1) es el intercepto
    % beta(2:end) son los coeficientes de importancia (feature weights)
    feature_weights = beta(2:end);
    
end