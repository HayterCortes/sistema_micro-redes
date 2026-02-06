%% --- Archivo: calculate_lime_pumping_3mg_LATENT.m ---
%
% Wrapper LIME para Q_p (Bombeo) usando PERTURBACIÓN LATENTE.
%
% METODOLOGÍA:
% 1. Codifica el estado actual al espacio latente (Z) usando el Autoencoder.
% 2. Perturba Z con ruido Gaussiano Isotrópico (variaciones intrínsecamente válidas).
% 3. Decodifica a X (proyectando sobre la variedad/manifold físico).
% 4. Usa 'explain_instance_lime_FROM_SAMPLES' para calcular R2 honesto.
%--------------------------------------------------------------------------
function [lime_stats, all_explanations] = calculate_lime_pumping_3mg_LATENT(estado, controller_obj, params, NUM_EXECUTIONS, TARGET_MG_IDX)
    
    % --- 1. CARGAR AUTOENCODER ---
    persistent ae_model
    if isempty(ae_model)
        % Búsqueda robusta del modelo entrenado
        possible_paths = { ...
            'models/autoencoder_manifold.mat', ...
            '../models/autoencoder_manifold.mat', ...
            '../../models/autoencoder_manifold.mat'};
        found = '';
        for i=1:length(possible_paths)
            if isfile(possible_paths{i}), found = possible_paths{i}; break; end
        end
        if isempty(found)
            error('Modelo AE no encontrado. Ejecuta primero train_manifold_autoencoder_UNIFIED.m');
        end
        loaded = load(found);
        ae_model = loaded.model_data;
    end

    X_original = estado.X_original; 
    feature_names = estado.feature_names;
    N_features = length(X_original);
    
    % --- 2. CONSTRUIR LÍMITES FÍSICOS (Hard Constraints) ---
    % Se usan solo como seguridad final post-decodificación
    min_bounds = []; max_bounds = [];
    for i = 1:3
        mg_p = params.mg(i);
        % Estructura del estado por MG: [SoC, V_tank, P_dem, P_gen, Q_dem]
        min_b_local = [mg_p.SoC_min, 0.0, 0.0, 0.0, 0.0];
        max_b_local = [mg_p.SoC_max, mg_p.V_max, inf, inf, inf];
        min_bounds = [min_bounds, min_b_local]; 
        max_bounds = [max_bounds, max_b_local]; 
    end
    % Variable global V_aq
    min_bounds = [min_bounds, 0.0]; 
    max_bounds = [max_bounds, inf];
    
    % --- 3. PARÁMETROS LIME LATENTE ---
    % Sigma aumentado a 0.5 para garantizar exploración real del vecindario
    % y evitar falsos positivos de R2=1.0 en zonas planas.
    latent_sigma = 0.5;  
    num_samples_lime = 1000; 
    kernel_width = 0.75 * sqrt(N_features);
    
    all_explanations = cell(NUM_EXECUTIONS, 1);
    r2_history = zeros(NUM_EXECUTIONS, 1);
    
    fprintf('Iniciando LIME BOMBEO (Latent Sampling)... ');
    
    for run_idx = 1:NUM_EXECUTIONS
        
        % --- A. PREPARAR DATO ORIGINAL ---
        X_row = X_original(:)';
        
        % Normalizar entrada para la red (Escala 0 a 1 según entrenamiento)
        % Nota: ae_model.min_vals y max_vals vienen del entrenamiento del AE
        X_norm = (X_row - ae_model.min_vals) ./ (ae_model.max_vals - ae_model.min_vals);
        
        % --- B. PERTURBACIÓN EN ESPACIO LATENTE (LA CLAVE) ---
        
        % 1. Encode (X -> Z)
        % Obtenemos la representación comprimida del estado actual
        Z_original = encode(ae_model.net, X_norm'); 
        
        % 2. Perturbar Z (Ruido Isotrópico en el Manifold)
        hidden_size = length(Z_original);
        noise_Z = randn(hidden_size, num_samples_lime) * latent_sigma;
        
        % Sumar ruido al vector latente
        Z_perturbed = bsxfun(@plus, Z_original, noise_Z);
        
        % 3. Decode (Z -> X)
        % Proyectar de vuelta al espacio físico siguiendo las geodésicas aprendidas
        X_projected_norm = decode(ae_model.net, Z_perturbed); 
        X_projected_norm = X_projected_norm'; % Transponer a (Samples x Features)
        
        % 4. Des-normalizar al mundo físico real
        X_samples = bsxfun(@plus, bsxfun(@times, X_projected_norm, (ae_model.max_vals - ae_model.min_vals)), ae_model.min_vals);
        
        % 5. Clipping de seguridad
        % El AE suele respetar la física, pero recortamos por si acaso hay desbordes numéricos
        X_samples = max(min(X_samples, max_bounds), min_bounds);
        
        % --- C. PREPARAR DATOS PARA LIME ---
        % Calculamos estadísticas de la nube generada para estandarizar
        mu_X = mean(X_samples, 1);
        sigma_X = std(X_samples, 0, 1);
        sigma_X(sigma_X < 1e-10) = 1.0; % Evitar división por cero
        
        % Estandarizar Muestras (X_samples_std) y el Origen (X_orig_std)
        % Esto es necesario para que el Kernel de distancia funcione bien
        X_samples_std = bsxfun(@rdivide, bsxfun(@minus, X_samples, mu_X), sigma_X);
        X_orig_std = (X_row - mu_X) ./ sigma_X; 
        
        % --- D. OBTENER PREDICCIONES (MPC BOMBEO) ---
        % Llamamos al adaptador que interroga al controlador MPC
        % IMPORTANTE: Pasamos las muestras generadas limpias
        Y_raw = predict_fn_mpc_wrapper_pumping_adapter( ...
            X_samples_std, mu_X, sigma_X, estado.constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX);
        
        % Normalizar etiquetas Y (Salida Q_p)
        mu_Y = mean(Y_raw);
        sigma_Y = std(Y_raw);
        
        % Debug opcional (descomentar si se requiere ver varianza de salida)
        if run_idx == 1, fprintf('[DEBUG Sigma_Y: %.4e] ', sigma_Y); end
        
        if sigma_Y < 1e-10, sigma_Y = 1.0; end
        
        Y_labels_std = (Y_raw - mu_Y) / sigma_Y;
        
        % --- E. LIME CORE (Correcto: FROM SAMPLES) ---
        % Usamos la función que SOLO calcula regresión, sin regenerar ruido.
        [weights, r2_val] = explain_instance_lime_FROM_SAMPLES( ...
            X_samples_std, Y_labels_std, X_orig_std, kernel_width);
        
        r2_history(run_idx) = r2_val;
        
        % Almacenar explicación de esta ejecución
        expl_list = cell(N_features, 2);
        for f = 1:N_features
            expl_list{f, 1} = feature_names{f};
            expl_list{f, 2} = weights(f);
        end
        all_explanations{run_idx} = expl_list;
    end
    
    % Empaquetar estadísticas finales
    lime_stats.R2_mean = mean(r2_history);
    lime_stats.R2_history = r2_history;
    lime_stats.R2_min = min(r2_history);
    fprintf('Done.\n');
end

% --- Helper Local para Adaptar Predictor de Bombeo ---
function Y = predict_fn_mpc_wrapper_pumping_adapter(X_std, mu, sigma, const, ctrl, min_b, max_b, target)
    % Esta función actúa como puente para llamar al predictor original
    % Asegúrate de que 'predict_fn_mpc_wrapper_pumping.m' esté en el path
    Y = predict_fn_mpc_wrapper_pumping(X_std, mu, sigma, const, ctrl, min_b, max_b, target);
end