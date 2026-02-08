%% --- Archivo: predict_fn_mpc_wrapper_pumping_MOMENTS.m ---
%
% Wrapper MPC Avanzado que reconstruye perfiles usando MOMENTOS (Mean, Max, Std).
% Permite que LIME explique decisiones basadas en PICOS y VARIABILIDAD.
%
% Entradas:
%   - muestras_std: Matriz (Nx34) estandarizada.
%   - mu, sigma: Vectores de media y desviación para des-estandarizar.
%   - constants: Estructura con perfiles base full.
%   - controller_obj: Objeto optimizador YALMIP.
%
% Lógica de Reconstrucción (Z-Score Inversa):
%   X_new(t) = Mean_target + (X_base(t) - Mean_base) * (Std_target / Std_base)
%--------------------------------------------------------------------------
function [predicciones_Y] = predict_fn_mpc_wrapper_pumping_MOMENTS(muestras_std, mu, sigma, constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX)

    % 1. Des-estandarizar
    % Recuperamos los valores físicos reales (incluyendo Max y Std)
    muestras_originales = bsxfun(@plus, bsxfun(@times, muestras_std, sigma), mu);
    
    % Clipping de seguridad
    muestras_originales = bsxfun(@min, muestras_originales, max_bounds);
    muestras_originales = bsxfun(@max, muestras_originales, min_bounds);
    
    N_samples = size(muestras_originales, 1);
    predicciones_Y = zeros(N_samples, 1); 
    
    % 2. Cargar Perfiles Base (Predicciones originales del MPC)
    p_dem_base_full = constants.p_dem_pred_full; 
    p_gen_base_full = constants.p_gen_pred_full;
    q_dem_base_full = constants.q_dem_pred_full;
    
    % 3. Pre-calcular estadísticas de la base para referencia
    % Necesitamos saber cuál era la media y desviación original para poder escalar
    
    % Medias Base
    mu_gen_base = mean(p_gen_base_full, 1); 
    mu_dem_base = mean(p_dem_base_full, 1);
    mu_q_base   = mean(q_dem_base_full, 1);
    
    % Desviaciones Base
    std_gen_base = std(p_gen_base_full, 0, 1);
    std_dem_base = std(p_dem_base_full, 0, 1);
    std_q_base   = std(q_dem_base_full, 0, 1);
    
    % Evitar división por cero si el perfil base es plano (noche/cero)
    std_gen_base(std_gen_base < 1e-6) = inf; 
    std_dem_base(std_dem_base < 1e-6) = inf;
    std_q_base(std_q_base < 1e-6)     = inf;

    % 4. Mapa de Índices (Mapping para 34 variables)
    % Estructura del vector X (34 vars):
    % [ 1-5 (MG1 Base), 6-10 (MG2 Base), 11-15 (MG3 Base), 16 (V_aq) ] -> 16 vars
    % [ 17-19 (Max_Pgen), 20-22 (Max_Pdem), 23-25 (Max_Qdem) ] -> 9 Maxs
    % [ 26-28 (Std_Pgen), 29-31 (Std_Pdem), 32-34 (Std_Qdem) ] -> 9 Stds
    
    % Índices MEDIAS (Están en el bloque base 1-16)
    idx_mean_pdem = [3, 8, 13];   % MG1, MG2, MG3
    idx_mean_pgen = [4, 9, 14];
    idx_mean_qdem = [5, 10, 15];
    
    % Índices STDS (Están en el bloque final 26-34)
    idx_std_pgen = [26, 27, 28];
    idx_std_pdem = [29, 30, 31];
    idx_std_qdem = [32, 33, 34];
    
    % Estados Iniciales
    idx_soc = [1, 6, 11];
    idx_tank = [2, 7, 12];
    idx_vaq = 16;

    % 5. Bucle Paralelo de Predicción
    % Usamos parfor para velocidad, ya que el MPC es costoso
    parfor i = 1:N_samples
        x_p = muestras_originales(i, :);
        
        % A. Reconstruir Estados Iniciales
        soC_0_vec   = x_p(idx_soc);
        v_tank_0_vec = x_p(idx_tank);
        v_aq_0_val  = x_p(idx_vaq);
        
        % B. Reconstruir Perfiles Temporales (Moment Matching)
        p_dem_k = zeros(size(p_dem_base_full));
        p_gen_k = zeros(size(p_gen_base_full));
        q_dem_k = zeros(size(q_dem_base_full));
        
        for mg = 1:3
            % --- 1. RECONSTRUCCIÓN SOLAR (P_GEN) ---
            target_mean = x_p(idx_mean_pgen(mg));
            target_std  = x_p(idx_std_pgen(mg));
            
            if isinf(std_gen_base(mg)) 
                p_gen_k(:, mg) = max(0, target_mean); 
            else
                shape = p_gen_base_full(:, mg) - mu_gen_base(mg);
                scale_factor = target_std / std_gen_base(mg);
                p_gen_k(:, mg) = max(0, target_mean + shape * scale_factor);
            end
            
            % --- 2. RECONSTRUCCIÓN DEMANDA ELÉCTRICA (P_DEM) ---
            target_mean_dem = x_p(idx_mean_pdem(mg));
            target_std_dem  = x_p(idx_std_pdem(mg));
            
            if isinf(std_dem_base(mg))
                p_dem_k(:, mg) = max(0, target_mean_dem);
            else
                shape_dem = p_dem_base_full(:, mg) - mu_dem_base(mg);
                scale_factor_dem = target_std_dem / std_dem_base(mg);
                p_dem_k(:, mg) = max(0, target_mean_dem + shape_dem * scale_factor_dem);
            end
            
            % --- 3. RECONSTRUCCIÓN DEMANDA AGUA (Q_DEM) ---
            % Crucial para capturar picos de consumo directo en MG1
            target_mean_q = x_p(idx_mean_qdem(mg));
            target_std_q  = x_p(idx_std_qdem(mg));
            
            if isinf(std_q_base(mg))
                q_dem_k(:, mg) = max(0, target_mean_q);
            else
                shape_q = q_dem_base_full(:, mg) - mu_q_base(mg);
                scale_factor_q = target_std_q / std_q_base(mg);
                q_dem_k(:, mg) = max(0, target_mean_q + shape_q * scale_factor_q);
            end
        end
        
        % C. Ejecutar Controlador MPC
        Inputs_k = { ...
            soC_0_vec, v_tank_0_vec, v_aq_0_val, ...
            p_dem_k, p_gen_k, q_dem_k, ...
            constants.q_p_hist_0, constants.p_mgref_hist_0, ...
            constants.k_mpc_actual, constants.Q_p_hist_mpc ...
        };
        
        try
            [sol_out, status] = controller_obj(Inputs_k);
            
            if status == 0
                % Extraer variable de decisión (Q_p es el output 2)
                q_p_res = sol_out{2}; 
                predicciones_Y(i) = q_p_res(TARGET_MG_IDX); 
            else
                % Fallo del solver (infeasible)
                predicciones_Y(i) = 0; 
            end
        catch
            predicciones_Y(i) = 0;
        end
    end
end