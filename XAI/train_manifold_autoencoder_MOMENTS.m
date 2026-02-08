%% --- Archivo: train_manifold_autoencoder_MOMENTS.m ---
%
% Entrena un Autoencoder (AE) UNIFICADO con MOMENTOS ESTADÍSTICOS.
%
% NOVEDAD: Incluye Mean, Max y Std para TODAS las variables de predicción
% (P_gen, P_dem, Q_dem) en las 3 MGs. Vector total: 34 features.
%--------------------------------------------------------------------------
clear; clc; close all;

% --- 1. CONFIGURACIÓN ---
MODELOS_ENTRENAMIENTO = {'AR', 'TS'}; 
HIDDEN_SIZE = 18; % Aumentado a 18 para manejar adecuadamente las 34 features
fprintf('--- ENTRENANDO AUTOENCODER (AR + TS) - MOMENTS FULL VERSION (34 vars) ---\n');

X_train_total = [];

% --- 2. BUCLE DE CARGA DE DATOS ---
for m_idx = 1:length(MODELOS_ENTRENAMIENTO)
    tipo_curr = MODELOS_ENTRENAMIENTO{m_idx};
    fprintf('\n>>> Procesando datos del modelo: %s <<<\n', tipo_curr);
    
    try
        fname = sprintf('resultados_mpc_%s_3mg_7dias.mat', tipo_curr);
        % Búsqueda robusta de archivo
        possible_paths = {fullfile('..', '..', 'results_mpc'), fullfile('..', 'results_mpc'), 'results_mpc', '.'};
        for i = 1:length(possible_paths)
            p = fullfile(possible_paths{i}, fname);
            if isfile(p), fname = p; break; end
        end
        
        if ~isfile(fname), warning('No encontrado: %s', tipo_curr); continue; end
        
        results = load(fname); 
        fprintf('  Archivo cargado: %s\n', fname);
        
        steps_sim = 1 : 12 : length(results.SoC); 
        
        h_wait = waitbar(0, ['Procesando ' tipo_curr '...']);
        
        X_batch = [];
        for k = steps_sim
            try
                [estado, ~] = reconstruct_state_matlab_3mg(k, tipo_curr);
                
                % --- FEATURE ENGINEERING (MOMENTOS COMPLETOS) ---
                P_dem = estado.constants.p_dem_pred_full; 
                P_gen = estado.constants.p_gen_pred_full; 
                Q_dem = estado.constants.q_dem_pred_full; 
                
                % 1. Medias (Reemplazarán los valores base correspondientes)
                m_P_dem = mean(P_dem, 1); 
                m_P_gen = mean(P_gen, 1); 
                m_Q_dem = mean(Q_dem, 1);
                
                % 2. Máximos (Picos de potencia y caudal)
                max_P_gen = max(P_gen, [], 1);
                max_P_dem = max(P_dem, [], 1);
                max_Q_dem = max(Q_dem, [], 1);
                
                % 3. Desviaciones Estándar (Variabilidad/Intermitencia)
                std_P_gen = std(P_gen, 0, 1);
                std_P_dem = std(P_dem, 0, 1);
                std_Q_dem = std(Q_dem, 0, 1);
                
                % Construir Vector Base (Mean)
                x_vec = estado.X_original;
                base_idx = [3, 8, 13];
                for m = 1:3
                    bi = base_idx(m);
                    x_vec(bi) = m_P_dem(m); x_vec(bi+1) = m_P_gen(m); x_vec(bi+2) = m_Q_dem(m);
                end
                
                % Construir Vector Extendido (34 variables)
                % Orden: [Base(16), Max_Pgen(3), Max_Pdem(3), Max_Qdem(3), Std_Pgen(3), Std_Pdem(3), Std_Qdem(3)]
                x_extended = [x_vec, max_P_gen, max_P_dem, max_Q_dem, std_P_gen, std_P_dem, std_Q_dem];
                
                X_batch = [X_batch; x_extended];
            catch
                continue; 
            end
            if mod(k, 100) == 0, waitbar(k/length(results.SoC), h_wait); end
        end
        close(h_wait);
        
        X_train_total = [X_train_total; X_batch];
        fprintf('  + %d muestras añadidas. Dimensión actual de matriz: [%d, %d]\n', size(X_batch, 1), size(X_train_total, 1), size(X_train_total, 2));
        
        % Guardar nombres de features extendidos (Solo la primera vez)
        if m_idx == 1 || ~exist('feature_names_saved', 'var')
            base_names = estado.feature_names;
            
            % Generar nombres nuevos para los momentos (Total 18 nombres nuevos)
            new_names = {};
            for i=1:3, new_names{end+1} = sprintf('Max_P_gen_MG%d', i); end
            for i=1:3, new_names{end+1} = sprintf('Max_P_dem_MG%d', i); end
            for i=1:3, new_names{end+1} = sprintf('Max_Q_dem_MG%d', i); end
            
            for i=1:3, new_names{end+1} = sprintf('Std_P_gen_MG%d', i); end
            for i=1:3, new_names{end+1} = sprintf('Std_P_dem_MG%d', i); end
            for i=1:3, new_names{end+1} = sprintf('Std_Q_dem_MG%d', i); end
            
            feature_names_saved = [base_names, new_names];
        end
        
    catch ME
        warning('Error procesando %s: %s', tipo_curr, ME.message);
    end
end

if isempty(X_train_total), error('No hay datos.'); end

% --- 3. NORMALIZACIÓN ---
min_vals = min(X_train_total, [], 1);
max_vals = max(X_train_total, [], 1);

% Evitar división por cero si alguna variable es constante
max_vals(max_vals == min_vals) = max_vals(max_vals == min_vals) + 1.0;

X_train_norm = (X_train_total - min_vals) ./ (max_vals - min_vals);
X_train_norm = X_train_norm'; 

% --- 4. ENTRENAMIENTO ---
fprintf('\nEntrenando Autoencoder (Hidden Size: %d, Features: %d)...\n', HIDDEN_SIZE, size(X_train_norm, 1));
autoenc = trainAutoencoder(X_train_norm, HIDDEN_SIZE, ...
    'MaxEpochs', 600, ...        
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'ScaleData', false); 

% --- 5. VALIDACIÓN ---
X_rec = predict(autoenc, X_train_norm);
mse = mean(mean((X_train_norm - X_rec).^2));
fprintf('MSE Reconstrucción: %.6f\n', mse);

% --- 6. GUARDAR ---
model_data.net = autoenc;
model_data.min_vals = min_vals;
model_data.max_vals = max_vals;
model_data.feature_names = feature_names_saved;

target_dir = 'models';
if exist('../models', 'dir'), target_dir = '../models'; end
if exist('../../models', 'dir'), target_dir = '../../models'; end
if ~exist(target_dir, 'dir'), mkdir(target_dir); end

save(fullfile(target_dir, 'autoencoder_manifold_MOMENTS.mat'), 'model_data');
fprintf('--> Modelo guardado en: autoencoder_manifold_MOMENTS.mat\n');