%% --- Archivo: predict_fn_mpc_wrapper_3mg.m ---
%
% Wrapper para conectar LIME con el Controlador MPC Cooperativo (3-MG).
%
% ** ACTUALIZACIÓN: Añadido argumento 'TARGET_MG_IDX' para seleccionar
%    qué variable de salida explicar (Q_t de MG1, MG2 o MG3).
%--------------------------------------------------------------------------
function [predicciones_Y] = predict_fn_mpc_wrapper_3mg(muestras_std, mu, sigma, constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX)

    N_samples = size(muestras_std, 1);
    predicciones_Y = zeros(N_samples, 1); 

    % --- 1. Des-estandarizar y Clipping ---
    muestras_originales = bsxfun(@plus, bsxfun(@times, muestras_std, sigma), mu);
    muestras_originales = bsxfun(@min, muestras_originales, max_bounds);
    muestras_originales = bsxfun(@max, muestras_originales, min_bounds);

    % --- 2. Extraer Constantes ---
    p_dem_base = constants.p_dem_pred_full; 
    p_gen_base = constants.p_gen_pred_full;
    q_dem_base = constants.q_dem_pred_full;
    
    % --- 3. Loop de Predicción ---
    parfor i = 1:N_samples
        x_p = muestras_originales(i, :);
        
        % Reconstruir Estados
        soC_0_vec   = [x_p(1), x_p(6), x_p(11)];
        v_tank_0_vec = [x_p(2), x_p(7), x_p(12)];
        v_aq_0_val  = x_p(16);
        
        % Reconstruir Pronósticos Escalados
        p_dem_k = zeros(size(p_dem_base));
        p_gen_k = zeros(size(p_gen_base));
        q_dem_k = zeros(size(q_dem_base));
        
        idx_base = [3, 4, 5;  8, 9, 10;  13, 14, 15]; 
        
        for mg_idx = 1:3
            % P_dem
            val_pert = x_p(idx_base(mg_idx, 1));
            val_base = p_dem_base(1, mg_idx);
            scale = 1.0;
            if abs(val_base) > 1e-6, scale = val_pert / val_base; end
            p_dem_k(:, mg_idx) = max(p_dem_base(:, mg_idx) * scale, 0);
            
            % P_gen
            val_pert = x_p(idx_base(mg_idx, 2));
            val_base = p_gen_base(1, mg_idx);
            scale = 1.0;
            if abs(val_base) > 1e-6, scale = val_pert / val_base; end
            p_gen_k(:, mg_idx) = max(p_gen_base(:, mg_idx) * scale, 0);
            
            % Q_dem
            val_pert = x_p(idx_base(mg_idx, 3));
            val_base = q_dem_base(1, mg_idx);
            scale = 1.0;
            if abs(val_base) > 1e-6, scale = val_pert / val_base; end
            q_dem_k(:, mg_idx) = max(q_dem_base(:, mg_idx) * scale, 0);
        end
        
        % Ejecutar Controlador
        Inputs_k = { ...
            soC_0_vec, v_tank_0_vec, v_aq_0_val, ...
            p_dem_k, p_gen_k, q_dem_k, ...
            constants.q_p_hist_0, constants.p_mgref_hist_0, ...
            constants.k_mpc_actual, constants.Q_p_hist_mpc ...
        };
        
        try
            [sol_out, status] = controller_obj(Inputs_k);
            
            if status == 0
                % Outputs = {P_mgref, Q_p, Q_buy, Q_t, ...}
                q_t_res = sol_out{4}; 
                
                % ** SELECCIÓN DINÁMICA DEL TARGET **
                predicciones_Y(i) = q_t_res(TARGET_MG_IDX); 
            else
                predicciones_Y(i) = 0; 
            end
        catch
            predicciones_Y(i) = 0;
        end
    end
end