function plot_resultados(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Ts)
    % Función para visualizar y guardar resultados para una presentación técnica.
    
    %% --- 1. Definición de Estilos y Parámetros de Visualización ---
    t = (0:size(SoC, 1)-1) * Ts / 3600; % Vector de tiempo en horas
    leyendas = {mg.nombre};
    
    % Estilos consistentes para todos los gráficos
    fontSizeTitle = 14;
    fontSizeLabels = 12;
    lineWidth = 2.0;
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250]; % Colores por defecto de MATLAB

    % Crear carpetas para guardar los resultados si no existen
    if ~exist('results/individual', 'dir'), mkdir('results/individual'); end
    if ~exist('results/paired', 'dir'), mkdir('results/paired'); end
    
    fprintf('Generando gráficos para la presentación...\n');

    %% --- 2. Generación de Gráficos Individuales ---
    % Cada gráfico se guarda como un archivo de imagen vectorial (.eps) y .png

    % Gráfico 1: Estado de Carga Baterías
    fig1 = figure('Name', 'SoC Baterías');
    plot(t, SoC * 100, 'LineWidth', lineWidth);
    title('Estado de Carga de las Baterías', 'FontSize', fontSizeTitle);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    ylabel('SoC [%]', 'FontSize', fontSizeLabels);
    ylim([0 100]); grid on;
    legend(leyendas, 'Location', 'best');
    set(gca, 'FontSize', fontSizeLabels);
    print(fig1, 'results/individual/SoC_baterias', '-depsc'); % .eps para LaTeX
    saveas(fig1, 'results/individual/SoC_baterias.png');

    % Gráfico 2: Volumen de Estanques
    fig2 = figure('Name', 'Volumen Estanques');
    plot(t, V_tank / 1000, 'LineWidth', lineWidth); % Convertir a m³
    title('Volumen de Agua en Estanques', 'FontSize', fontSizeTitle);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    ylabel('Volumen [m^3]', 'FontSize', fontSizeLabels); % Unidad cambiada
    grid on;
    legend(leyendas, 'Location', 'best');
    set(gca, 'FontSize', fontSizeLabels);
    print(fig2, 'results/individual/Volumen_estanques', '-depsc');
    saveas(fig2, 'results/individual/Volumen_estanques.png');

    % Gráfico 3: Volumen del Acuífero
    fig3 = figure('Name', 'Volumen Acuífero');
    plot(t, V_aq / 1000, 'k', 'LineWidth', lineWidth); % Convertir a m³
    title('Volumen del Acuífero Compartido', 'FontSize', fontSizeTitle);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    ylabel('Volumen [m^3]', 'FontSize', fontSizeLabels); % Unidad cambiada
    grid on;
    set(gca, 'FontSize', fontSizeLabels);
    print(fig3, 'results/individual/Volumen_acuifero', '-depsc');
    saveas(fig3, 'results/individual/Volumen_acuifero.png');

    %% --- 3. Generación de Gráficos Comparativos (Pares) ---
    % Mostrar relaciones causa-efecto en una misma diapositiva.
    
    % Par 1: Dinámica de Bombeo (Caudal vs. Potencia)
    fig_pair1 = figure('Name', 'Dinámica de Bombeo');
    subplot(2,1,1); % Gráfico superior
    plot(t, Q_p, 'LineWidth', lineWidth-0.5);
    title('Dinámica de Bombeo', 'FontSize', fontSizeTitle);
    ylabel('Caudal [L/s]', 'FontSize', fontSizeLabels);
    grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);
    
    subplot(2,1,2); % Gráfico inferior
    plot(t, P_pump, 'LineWidth', lineWidth-0.5);
    ylabel('Potencia [kW]', 'FontSize', fontSizeLabels);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);
    print(fig_pair1, 'results/paired/Dinamica_bombeo', '-depsc');
    saveas(fig_pair1, 'results/paired/Dinamica_bombeo.png');

    % Par 2: Interacción con la Red (Compra de Agua y Energía)
    fig_pair2 = figure('Name', 'Interacción con DNO');
    subplot(2,1,1);
    plot(t, Q_DNO, 'LineWidth', lineWidth-0.5);
    title('Interacción con la Red Externa (DNO)', 'FontSize', fontSizeTitle);
    ylabel('Agua Comprada [L/s]', 'FontSize', fontSizeLabels);
    grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);

    subplot(2,1,2);
    plot(t, P_grid, 'LineWidth', lineWidth-0.5);
    ylabel('Energía Comprada [kW]', 'FontSize', fontSizeLabels);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);
    print(fig_pair2, 'results/paired/Interaccion_DNO', '-depsc');
    saveas(fig_pair2, 'results/paired/Interaccion_DNO.png');
    
    % Par 3: Estados de Almacenamiento (Energía vs. Agua)
    fig_pair3 = figure('Name', 'Estados de Almacenamiento');
    subplot(2,1,1);
    plot(t, SoC * 100, 'LineWidth', lineWidth-0.5);
    title('Estados de los Sistemas de Almacenamiento', 'FontSize', fontSizeTitle);
    ylabel('SoC Baterías [%]', 'FontSize', fontSizeLabels);
    ylim([0 100]); grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);
    
    subplot(2,1,2);
    plot(t, V_tank / 1000, 'LineWidth', lineWidth-0.5); % m³
    ylabel('Volumen Estanques [m^3]', 'FontSize', fontSizeLabels);
    xlabel('Tiempo [horas]', 'FontSize', fontSizeLabels);
    grid on; legend(leyendas);
    set(gca, 'FontSize', fontSizeLabels);
    print(fig_pair3, 'results/paired/Almacenamiento', '-depsc');
    saveas(fig_pair3, 'results/paired/Almacenamiento.png');

    close all;
end