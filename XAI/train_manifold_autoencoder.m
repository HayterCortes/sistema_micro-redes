%% --- Archivo: train_manifold_autoencoder.m ---
%
% Entrena un Autoencoder (AE) UNIFICADO para aprender la variedad física.
%
% CORRECCIÓN FINAL:
% 1. Carga variables correctamente desde la estructura .mat.
% 2. Guarda el modelo en la carpeta 'models' correcta (buscando ../ o ../../).
%--------------------------------------------------------------------------
clear; clc; close all;

% --- 1. CONFIGURACIÓN ---
MODELOS_ENTRENAMIENTO = {'AR', 'TS'}; 
HIDDEN_SIZE = 10; 

fprintf('--- ENTRENANDO AUTOENCODER UNIFICADO (AR + TS) ---\n');

X_train_total = [];

% --- 2. BUCLE DE CARGA DE DATOS ---
for m_idx = 1:length(MODELOS_ENTRENAMIENTO)
    tipo_curr = MODELOS_ENTRENAMIENTO{m_idx};
    fprintf('\n>>> Procesando datos del modelo: %s <<<\n', tipo_curr);
    
    try
        % Rutas posibles
        fname = sprintf('../results_mpc/resultados_mpc_%s_3mg_7dias.mat', tipo_curr);
        if ~isfile(fname), fname = sprintf('resultados_mpc_%s_3mg_7dias.mat', tipo_curr); end
        if ~isfile(fname), fname = sprintf('results_mpc/resultados_mpc_%s_3mg_7dias.mat', tipo_curr); end
        
        if ~isfile(fname)
            warning('No se encuentra archivo para %s. Saltando...', tipo_curr);
            continue;
        end
        
        % Carga directa
        results = load(fname); 
        fprintf('  Archivo cargado: %s\n', fname);
        
        if ~isfield(results, 'SoC')
            warning('Variable SoC no encontrada en %s', fname);
            continue;
        end
        
        % Submuestreo
        steps_sim = 1 : 12 : length(results.SoC); 
        
        fprintf('  Extrayendo características...\n');
        h_wait = waitbar(0, ['Procesando ' tipo_curr '...']);
        
        X_batch = [];
        for k = steps_sim
            try
                [estado, ~] = reconstruct_state_matlab_3mg(k, tipo_curr);
                
                % Feature Engineering (Mean)
                P_dem = estado.constants.p_dem_pred_full; 
                P_gen = estado.constants.p_gen_pred_full; 
                Q_dem = estado.constants.q_dem_pred_full; 
                m_P_dem = mean(P_dem, 1); m_P_gen = mean(P_gen, 1); m_Q_dem = mean(Q_dem, 1);
                
                x_vec = estado.X_original;
                base_idx = [3, 8, 13];
                for m = 1:3
                    bi = base_idx(m);
                    x_vec(bi) = m_P_dem(m); x_vec(bi+1) = m_P_gen(m); x_vec(bi+2) = m_Q_dem(m);
                end
                X_batch = [X_batch; x_vec];
            catch
                continue; 
            end
            if mod(k, 100) == 0, waitbar(k/length(results.SoC), h_wait); end
        end
        close(h_wait);
        
        X_train_total = [X_train_total; X_batch];
        fprintf('  + %d muestras añadidas.\n', size(X_batch, 1));
        
        if m_idx == 1 || ~exist('feature_names_saved', 'var')
            feature_names_saved = estado.feature_names;
        end
        
    catch ME
        warning('Error procesando %s: %s', tipo_curr, ME.message);
    end
end

if isempty(X_train_total), error('No hay datos para entrenar.'); end

% --- 3. NORMALIZACIÓN ---
min_vals = min(X_train_total, [], 1);
max_vals = max(X_train_total, [], 1);
max_vals(max_vals == min_vals) = max_vals(max_vals == min_vals) + 1.0;

X_train_norm = (X_train_total - min_vals) ./ (max_vals - min_vals);
X_train_norm = X_train_norm'; 

% --- 4. ENTRENAMIENTO ---
fprintf('\nEntrenando Autoencoder Unificado...\n');
autoenc = trainAutoencoder(X_train_norm, HIDDEN_SIZE, ...
    'MaxEpochs', 500, ...        
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'ScaleData', false); 

% --- 5. VALIDACIÓN ---
X_reconstructed = predict(autoenc, X_train_norm);
mse_error = mean(mean((X_train_norm - X_reconstructed).^2));
fprintf('Error de reconstrucción (MSE): %.6f\n', mse_error);

% --- 6. GUARDAR (Ruta Robusta) ---
model_data.net = autoenc;
model_data.min_vals = min_vals;
model_data.max_vals = max_vals;
model_data.feature_names = feature_names_saved;
model_data.trained_on = MODELOS_ENTRENAMIENTO;

% Buscar la carpeta 'models' correcta (subiendo niveles si es necesario)
target_dir = 'models'; % Default local
if exist('../models', 'dir'), target_dir = '../models'; end
if exist('../../models', 'dir'), target_dir = '../../models'; end

% Crear si no existe en la ruta detectada
if ~exist(target_dir, 'dir')
    mkdir(target_dir);
end

outfile = fullfile(target_dir, 'autoencoder_manifold.mat');
save(outfile, 'model_data');

fprintf('--> Modelo UNIFICADO guardado en: %s\n', outfile);
fprintf('LISTO. Ahora ejecuta el script de LIME con AE.\n');