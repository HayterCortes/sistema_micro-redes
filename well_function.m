function W = well_function(u)
    % Calcula la función de pozo de Theis W(u) utilizando la 
    % función de integral exponencial E1(u) de MATLAB.
    %
    % Esta función implementa la Ecuación (4.17) de la tesis de 
    % Jiménez, L. (2024).
    %
    % Entradas:
    %   u: Argumento adimensional de la función.
    %
    % Salidas:
    %   W: Valor de la función de pozo.

    W = expint(u);
end