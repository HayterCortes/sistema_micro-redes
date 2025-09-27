% cargar_y_preparar_datos.m
function [P_dem_sim, P_gen_sim, Q_dem_sim, P_dem_train, P_gen_train, Q_dem_train, P_dem_val, P_gen_val, Q_dem_val, P_dem_test, P_gen_test, Q_dem_test] = cargar_y_preparar_datos(dias_sim, paso_mpc, train_ratio, val_ratio)
    % Carga, pre-procesa y divide cronológicamente los datos en Entrenamiento, Validación y Prueba.
    
    fprintf('--- Iniciando carga y pre-procesamiento de datos ---\n');
    addpath('data');

    % --- Carga de datos crudos (raw) ---
    d_mr1_raw = load('data/winter_30D.mat').winter_30D;
    g_mr1_raw = load('data/pv_wint.mat').pv_wint * 22;
    d_mr2_raw = load('data/winter_60D.mat').winter_60D;
    g_mr2_raw = load('data/wind_inv.mat').wind_inv * 8.49;
    d_mr3_raw = load('data/School_inv.mat').School_inv * 0.45;
    g_mr3_raw = load('data/pv_wint.mat').pv_wint * 30 + load('data/wind_inv.mat').wind_inv * 5;
    h_mr1_raw = load('data/Dwellings30Water.mat').Dwellings30Water;
    h_mr2_raw = load('data/Dwellings60Water.mat').Dwellings60Water;
    h_mr3_raw = load('data/SchoolWater.mat').SchoolWater;

    %% --- 1. Procesar Datos para MODELADO (submuestreados) ---
    fprintf('Submuestreando datos para modelado (paso = %d min)...\n', paso_mpc);
    d1_ts = submuestreo_max(d_mr1_raw, paso_mpc); d2_ts = submuestreo_max(d_mr2_raw, paso_mpc); d3_ts = submuestreo_max(d_mr3_raw, paso_mpc);
    g1_ts = submuestreo_max(g_mr1_raw, paso_mpc); g2_ts = submuestreo_max(g_mr2_raw, paso_mpc); g3_ts = submuestreo_max(g_mr3_raw, paso_mpc);
    h1_ts = submuestreo_max(h_mr1_raw, paso_mpc); h2_ts = submuestreo_max(h_mr2_raw, paso_mpc); h3_ts = submuestreo_max(h_mr3_raw, paso_mpc);

    min_len_ts = min([length(d1_ts), length(d2_ts), length(d3_ts), length(g1_ts), length(g2_ts), length(g3_ts), length(h1_ts), length(h2_ts), length(h3_ts)]);
    
    P_dem_full = [d1_ts(1:min_len_ts), d2_ts(1:min_len_ts), d3_ts(1:min_len_ts)];
    P_gen_full = [g1_ts(1:min_len_ts), g2_ts(1:min_len_ts), g3_ts(1:min_len_ts)];
    Q_dem_full = [h1_ts(1:min_len_ts), h2_ts(1:min_len_ts), h3_ts(1:min_len_ts)];

    % --- División Cronológica en 3 partes ---
    train_end_idx = floor(train_ratio * min_len_ts);
    val_end_idx = floor((train_ratio + val_ratio) * min_len_ts);
    fprintf('Dividiendo datos de modelado: %d (Train), %d (Val), %d (Test).\n', train_end_idx, val_end_idx - train_end_idx, min_len_ts - val_end_idx);
    
    P_dem_train = P_dem_full(1:train_end_idx, :); P_dem_val = P_dem_full(train_end_idx+1:val_end_idx, :); P_dem_test = P_dem_full(val_end_idx+1:end, :);
    P_gen_train = P_gen_full(1:train_end_idx, :); P_gen_val = P_gen_full(train_end_idx+1:val_end_idx, :); P_gen_test = P_gen_full(val_end_idx+1:end, :);
    Q_dem_train = Q_dem_full(1:train_end_idx, :); Q_dem_val = Q_dem_full(train_end_idx+1:val_end_idx, :); Q_dem_test = Q_dem_full(val_end_idx+1:end, :);
    
    %% --- 2. Preparar Datos para SIMULACIÓN (alta resolución, del conjunto de prueba) ---
    fprintf('Preparando datos para simulación (%d días) del conjunto de prueba...\n', dias_sim);
    
    min_len_raw = min([length(d_mr1_raw), length(g_mr1_raw), length(d_mr2_raw), length(g_mr2_raw), length(d_mr3_raw), length(g_mr3_raw), length(h_mr1_raw), length(h_mr2_raw), length(h_mr3_raw)]);
    
    val_start_idx_raw = floor(train_ratio * min_len_raw) + 1;
    test_start_idx_raw = floor((train_ratio + val_ratio) * min_len_raw) + 1;
    
    sim_start_idx = test_start_idx_raw;
    muestras_sim = dias_sim * 24 * 60;
    sim_end_idx = sim_start_idx + muestras_sim - 1;
    
    if sim_end_idx > min_len_raw
        error('No hay suficientes datos en el conjunto de prueba para una simulación de %d días.', dias_sim);
    end
    
    P_dem_sim = [d_mr1_raw(sim_start_idx:sim_end_idx), d_mr2_raw(sim_start_idx:sim_end_idx), d_mr3_raw(sim_start_idx:sim_end_idx)];
    P_gen_sim = [g_mr1_raw(sim_start_idx:sim_end_idx), g_mr2_raw(sim_start_idx:sim_end_idx), g_mr3_raw(sim_start_idx:sim_end_idx)];
    Q_dem_sim = [h_mr1_raw(sim_start_idx:sim_end_idx), h_mr2_raw(sim_start_idx:sim_end_idx), h_mr3_raw(sim_start_idx:sim_end_idx)];
    
    fprintf('--- Carga y pre-procesamiento completado ---\n\n');
end

function datos_submuestreados = submuestreo_max(datos, tamano_ventana)
    num_ventanas = floor(length(datos) / tamano_ventana);
    longitud_ajustada = num_ventanas * tamano_ventana;
    datos_ajustados = datos(1:longitud_ajustada);
    matriz_datos = reshape(datos_ajustados, tamano_ventana, num_ventanas);
    datos_submuestreados = max(matriz_datos, [], 1)';
end