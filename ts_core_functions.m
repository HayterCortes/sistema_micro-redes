% --- ts_core_functions.m ---
function funcs = ts_core_functions()
    funcs.train = @train_ts;
    funcs.eval = @eval_ts;
end

function model = train_ts(X, y, n_clusters)
    % Implementa la identificación del modelo TS (Sección 2.3.2)
    % 1. Clustering Difuso (Fuzzy C-Means) para las premisas
    options = [2.0, 100, 1e-5, 0]; % [exponent, max_iter, min_impro, display]
    [centers, U] = fcm(X, n_clusters, options);
    
    % 2. Estimación de parámetros de premisas (Gaussianas) [cite: 368]
    % Se estima sigma para cada cluster y cada dimensión de entrada
    [n_samples, n_inputs] = size(X);
    sigmas = zeros(n_clusters, n_inputs);
    
    for i = 1:n_clusters
        for j = 1:n_inputs
            % Desviación estándar ponderada por la pertenencia U
            diff = X(:, j) - centers(i, j);
            weights = U(i, :)'.^2; % Exponente difuso m=2
            sigmas(i, j) = sqrt(sum(weights .* (diff.^2)) / sum(weights));
            
            % Evitar sigma cero por estabilidad numérica
            if sigmas(i, j) < 1e-6, sigmas(i, j) = 1e-6; end
        end
    end
    
    % 3. Estimación de consecuencias (Modelos Lineales Locales) [cite: 385]
    % y_r = theta_0 + theta_1*x1 + ... (Ecuación 2.9)
    thetas = zeros(n_inputs + 1, n_clusters);
    X_aug = [ones(n_samples, 1), X]; % Matriz aumentada para el bias (theta_0)
    
    for i = 1:n_clusters
        % Mínimos Cuadrados Ponderados (WLS)
        % Matriz de pesos diagonal con las pertenencias
        W = diag(U(i, :));
        
        % Solución analítica: theta = (X'WX)^-1 * X'Wy
        % Usamos lscov o operador \ por eficiencia y estabilidad
        % theta_i = lscov(X_aug, y, U(i, :)'); 
        
        % Implementación manual robusta para WLS:
        % Se ponderan las filas de X e y por la raíz de los pesos
        sqrt_W = sqrt(U(i, :)');
        X_w = X_aug .* sqrt_W;
        y_w = y .* sqrt_W;
        
        thetas(:, i) = X_w \ y_w;
    end
    
    model.centers = centers;
    model.sigmas = sigmas;
    model.thetas = thetas;
    model.n_rules = n_clusters;
end

function y_pred = eval_ts(model, X_new)
    % Evalúa el modelo TS para nuevos datos (Ecuaciones 2.10 - 2.13)
    [n_samples, n_inputs] = size(X_new);
    n_rules = model.n_rules;
    
    beta = zeros(n_samples, n_rules);
    
    % 1. Calcular grado de activación (Firing Strength) - Eq 2.12 & 2.13
    for r = 1:n_rules
        % Función de pertenencia Gaussiana multidimensional (producto)
        mu_matrix = exp(-((X_new - model.centers(r, :)).^2) ./ (2 * model.sigmas(r, :).^2));
        beta(:, r) = prod(mu_matrix, 2);
    end
    
    % 2. Normalizar activaciones (Eq 2.11)
    sum_beta = sum(beta, 2);
    % Evitar división por cero
    sum_beta(sum_beta == 0) = 1e-12; 
    beta_norm = beta ./ sum_beta;
    
    % 3. Calcular salidas locales y salida global (Eq 2.10)
    X_aug = [ones(n_samples, 1), X_new];
    y_pred = zeros(n_samples, 1);
    
    for r = 1:n_rules
        y_local = X_aug * model.thetas(:, r);
        y_pred = y_pred + beta_norm(:, r) .* y_local;
    end
end