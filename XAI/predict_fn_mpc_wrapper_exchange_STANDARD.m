%% --- Archivo: predict_fn_mpc_wrapper_exchange_STANDARD.m ---
% Wrapper simple para LIME Standard (Sin AE).
% Recibe muestras perturbadas (físicas) y devuelve Q_t.
%--------------------------------------------------------------------------
function [predicciones_Y] = predict_fn_mpc_wrapper_exchange_STANDARD(X_samples_phys, constants, controller_obj, TARGET_MG_IDX)

    N_samples = size(X_samples_phys, 1);
    predicciones_Y = zeros(N_samples, 1);
    
    % Cargar Bases
    p_dem_base_full = constants.p_dem_pred_full; 
    p_gen_base_full = constants.p_gen_pred_full;
    q_dem_base_full = constants.q_dem_pred_full;
    
    % Estadísticas Base
    mu_gen = mean(p_gen_base_full,1); std_gen = std(p_gen_base_full,0,1); std_gen(std_gen<1e-6)=inf;
    mu_dem = mean(p_dem_base_full,1); std_dem = std(p_dem_base_full,0,1); std_dem(std_dem<1e-6)=inf;
    mu_q   = mean(q_dem_base_full,1); std_q   = std(q_dem_base_full,0,1); std_q(std_q<1e-6)=inf;
    
    % Índices 34 vars
    idx_soc=[1,6,11]; idx_tank=[2,7,12]; idx_vaq=16;
    idx_m_pd=[3,8,13]; idx_m_pg=[4,9,14]; idx_m_qd=[5,10,15];
    idx_s_pg=[26,27,28]; idx_s_pd=[29,30,31]; idx_s_qd=[32,33,34];
    
    parfor i = 1:N_samples
        x = X_samples_phys(i, :);
        
        % A. Estados
        soC_0 = x(idx_soc); V_tank_0 = x(idx_tank); V_aq_0 = x(idx_vaq);
        
        % B. Perfiles (Z-Score)
        p_g = zeros(size(p_gen_base_full)); p_d = zeros(size(p_dem_base_full)); q_d = zeros(size(q_dem_base_full));
        
        for m=1:3
            % P_Gen
            tm=x(idx_m_pg(m)); ts=x(idx_s_pg(m));
            if isinf(std_gen(m)), p_g(:,m)=max(0,tm); else, p_g(:,m)=max(0,tm+(p_gen_base_full(:,m)-mu_gen(m))*(ts/std_gen(m))); end
            
            % P_Dem
            tm=x(idx_m_pd(m)); ts=x(idx_s_pd(m));
            if isinf(std_dem(m)), p_d(:,m)=max(0,tm); else, p_d(:,m)=max(0,tm+(p_dem_base_full(:,m)-mu_dem(m))*(ts/std_dem(m))); end
            
            % Q_Dem
            tm=x(idx_m_qd(m)); ts=x(idx_s_qd(m));
            if isinf(std_q(m)), q_d(:,m)=max(0,tm); else, q_d(:,m)=max(0,tm+(q_dem_base_full(:,m)-mu_q(m))*(ts/std_q(m))); end
        end
        
        % C. MPC
        Inputs = {soC_0, V_tank_0, V_aq_0, p_d, p_g, q_d, ...
                  constants.q_p_hist_0, constants.p_mgref_hist_0, constants.k_mpc_actual, constants.Q_p_hist_mpc};
        try
            [sol, stat] = controller_obj(Inputs);
            if stat==0, res=sol{4}; predicciones_Y(i)=res(TARGET_MG_IDX); else, predicciones_Y(i)=0; end
        catch, predicciones_Y(i)=0; end
    end
end