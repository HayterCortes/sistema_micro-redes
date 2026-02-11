%% --- Archivo: predict_fn_mpc_wrapper_exchange_MOMENTS.m ---
%
% Wrapper MPC para explicar INTERCAMBIO HÍDRICO (Q_t) usando MOMENTOS.
% Reconstruye perfiles complejos (Picos/Variabilidad) y devuelve la decisión de intercambio.
%--------------------------------------------------------------------------
function [predicciones_Y] = predict_fn_mpc_wrapper_exchange_MOMENTS(muestras_std, mu, sigma, constants, controller_obj, min_bounds, max_bounds, TARGET_MG_IDX)

    % 1. Des-estandarizar y Clipping
    muestras_originales = bsxfun(@plus, bsxfun(@times, muestras_std, sigma), mu);
    muestras_originales = bsxfun(@min, muestras_originales, max_bounds);
    muestras_originales = bsxfun(@max, muestras_originales, min_bounds);
    
    N_samples = size(muestras_originales, 1);
    predicciones_Y = zeros(N_samples, 1); 
    
    % 2. Cargar Base y Estadísticas
    p_dem_base_full = constants.p_dem_pred_full; 
    p_gen_base_full = constants.p_gen_pred_full;
    q_dem_base_full = constants.q_dem_pred_full;
    
    mu_gen_base = mean(p_gen_base_full, 1); std_gen_base = std(p_gen_base_full, 0, 1); std_gen_base(std_gen_base<1e-6)=inf;
    mu_dem_base = mean(p_dem_base_full, 1); std_dem_base = std(p_dem_base_full, 0, 1); std_dem_base(std_dem_base<1e-6)=inf;
    mu_q_base   = mean(q_dem_base_full, 1); std_q_base   = std(q_dem_base_full, 0, 1); std_q_base(std_q_base<1e-6)=inf;

    % Índices (Mapeo de 34 vars)
    % Base(16) + MaxPgen(3)+MaxPdem(3)+MaxQdem(3) + StdPgen(3)+StdPdem(3)+StdQdem(3)
    idx_soc=[1,6,11]; idx_tank=[2,7,12]; idx_vaq=16;
    idx_mean_pdem=[3,8,13]; idx_mean_pgen=[4,9,14]; idx_mean_qdem=[5,10,15];
    
    % Asumiendo que las Std están al final (posiciones 26 a 34)
    idx_std_pgen=[26,27,28]; idx_std_pdem=[29,30,31]; idx_std_qdem=[32,33,34];

    % 3. Bucle de Predicción
    parfor i = 1:N_samples
        x_p = muestras_originales(i, :);
        
        % A. Reconstruir Estados
        soC_0_vec=x_p(idx_soc); v_tank_0_vec=x_p(idx_tank); v_aq_0_val=x_p(idx_vaq);
        
        % B. Reconstruir Perfiles (Z-Score Logic)
        p_dem_k = zeros(size(p_dem_base_full)); 
        p_gen_k = zeros(size(p_gen_base_full)); 
        q_dem_k = zeros(size(q_dem_base_full));
        
        for mg = 1:3
            % P_GEN
            tm=x_p(idx_mean_pgen(mg)); ts=x_p(idx_std_pgen(mg));
            if isinf(std_gen_base(mg)), p_gen_k(:,mg)=max(0,tm); else, p_gen_k(:,mg)=max(0,tm+(p_gen_base_full(:,mg)-mu_gen_base(mg))*(ts/std_gen_base(mg))); end
            
            % P_DEM
            tm=x_p(idx_mean_pdem(mg)); ts=x_p(idx_std_pdem(mg));
            if isinf(std_dem_base(mg)), p_dem_k(:,mg)=max(0,tm); else, p_dem_k(:,mg)=max(0,tm+(p_dem_base_full(:,mg)-mu_dem_base(mg))*(ts/std_dem_base(mg))); end
            
            % Q_DEM
            tm=x_p(idx_mean_qdem(mg)); ts=x_p(idx_std_qdem(mg));
            if isinf(std_q_base(mg)), q_dem_k(:,mg)=max(0,tm); else, q_dem_k(:,mg)=max(0,tm+(q_dem_base_full(:,mg)-mu_q_base(mg))*(ts/std_q_base(mg))); end
        end
        
        % C. Ejecutar MPC
        Inputs_k = {soC_0_vec, v_tank_0_vec, v_aq_0_val, p_dem_k, p_gen_k, q_dem_k, ...
                    constants.q_p_hist_0, constants.p_mgref_hist_0, constants.k_mpc_actual, constants.Q_p_hist_mpc};
        
        try
            [sol_out, status] = controller_obj(Inputs_k);
            if status == 0
                % *** CRÍTICO: OUTPUT 4 es Q_t (Intercambio) ***
                q_t_res = sol_out{4}; 
                predicciones_Y(i) = q_t_res(TARGET_MG_IDX); 
            else
                predicciones_Y(i) = 0; 
            end
        catch
            predicciones_Y(i) = 0;
        end
    end
end