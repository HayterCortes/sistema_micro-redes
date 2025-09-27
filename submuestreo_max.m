function datos_submuestreados = submuestreo_max(datos, tamano_ventana)
    % SUBMUESTREO_MAX: Submuestrea una o varias series temporales (por columnas)
    % tomando el valor máximo de cada ventana.
    
    if isvector(datos)
        datos = datos(:); % Asegurar que sea un vector columna
    end
    num_series = size(datos, 2);
    num_ventanas = floor(size(datos, 1) / tamano_ventana);
    datos_submuestreados = zeros(num_ventanas, num_series);

    for i = 1:num_series
        serie_actual = datos(:, i);
        if num_ventanas == 0
            datos_submuestreados = [];
            return;
        end
        longitud_ajustada = num_ventanas * tamano_ventana;
        datos_ajustados = serie_actual(1:longitud_ajustada);
        matriz_datos = reshape(datos_ajustados, tamano_ventana, num_ventanas);
        datos_submuestreados(:, i) = max(matriz_datos, [], 1)';
    end
end