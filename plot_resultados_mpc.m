% plot_resultados_mpc.m (Versión Final con Exportación a .eps)
function plot_resultados_mpc(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Q_t)
    
    %% --- 1. Definición de Estilos y Parámetros ---
    t = (0:size(SoC, 1)-1) * mg(1).Ts_sim / 3600; 
    leyendas = {mg.nombre};
    fontSizeTitle = 14;
    fontSizeLabels = 12;
    lineWidth = 1.5;
    
    if ~exist('results_mpc', 'dir'), mkdir('results_mpc'); end
    if ~exist('results_mpc/individual', 'dir'), mkdir('results_mpc/individual'); end
    if ~exist('results_mpc/paired', 'dir'), mkdir('results_mpc/paired'); end
    
    fprintf('Generando y exportando gráficos en formato PNG y EPS...\n');
    
    %% --- 2. Generación de Gráficos ---

    % Gráfico 1: Estado de Carga (SoC) de Baterías
    fig1 = figure('Name', 'SoC Baterías (MPC)');
    plot(t, SoC * 100, 'LineWidth', lineWidth);
    title('Estado de Carga de las Baterías (Control MPC)');
    xlabel('Tiempo [horas]'); ylabel('SoC [%]'); ylim([0 100]); grid on;
    legend(leyendas, 'Location', 'best'); set(gca, 'FontSize', fontSizeLabels);
    saveas(fig1, 'results_mpc/individual/SoC_baterias_mpc.png');
    print(fig1, 'results_mpc/individual/SoC_baterias_mpc', '-depsc'); % Exportar a .eps

    % Gráfico 2: Volumen en Estanques
    fig2 = figure('Name', 'Volumen Estanques (MPC)');
    plot(t, V_tank / 1000, 'LineWidth', lineWidth);
    title('Volumen de Agua en Estanques (Control MPC)');
    xlabel('Tiempo [horas]'); ylabel('Volumen [m^3]'); grid on;
    legend(leyendas, 'Location', 'best'); set(gca, 'FontSize', fontSizeLabels);
    saveas(fig2, 'results_mpc/individual/Volumen_estanques_mpc.png');
    print(fig2, 'results_mpc/individual/Volumen_estanques_mpc', '-depsc'); % Exportar a .eps

    % Gráfico 3: Volumen del Acuífero
    fig3 = figure('Name', 'Volumen Acuífero (MPC)');
    plot(t, V_aq / 1000, 'k', 'LineWidth', lineWidth);
    title('Volumen del Acuífero Compartido (Control MPC)');
    xlabel('Tiempo [horas]'); ylabel('Volumen [m^3]'); grid on;
    set(gca, 'FontSize', fontSizeLabels);
    saveas(fig3, 'results_mpc/individual/Volumen_acuifero_mpc.png');
    print(fig3, 'results_mpc/individual/Volumen_acuifero_mpc', '-depsc'); % Exportar a .eps

    % Gráfico 4: Dinámica de Bombeo
    fig4 = figure('Name', 'Dinámica de Bombeo (MPC)');
    subplot(2,1,1);
    plot(t, Q_p, 'LineWidth', lineWidth);
    title('Dinámica de Bombeo (Control MPC)');
    ylabel('Caudal Bombeado [L/s]');
    grid on; legend(leyendas); set(gca, 'FontSize', fontSizeLabels);
    subplot(2,1,2);
    plot(t, P_pump, 'LineWidth', lineWidth);
    ylabel('Potencia de Bomba [kW]'); xlabel('Tiempo [horas]');
    grid on; legend(leyendas); set(gca, 'FontSize', fontSizeLabels);
    saveas(fig4, 'results_mpc/paired/Dinamica_bombeo_mpc.png');
    print(fig4, 'results_mpc/paired/Dinamica_bombeo_mpc', '-depsc'); % Exportar a .eps

    % Gráfico 5: Interacción con DNO
    fig5 = figure('Name', 'Interacción con DNO (MPC)');
    subplot(2,1,1);
    plot(t, P_grid, 'LineWidth', lineWidth);
    title('Interacción con la Red Externa (Control MPC)');
    ylabel('Energía Comprada [kW]');
    grid on; legend(leyendas); set(gca, 'FontSize', fontSizeLabels);
    subplot(2,1,2);
    plot(t, Q_DNO, 'LineWidth', lineWidth);
    ylabel('Agua Comprada [L/s]'); xlabel('Tiempo [horas]');
    grid on; legend(leyendas); set(gca, 'FontSize', fontSizeLabels);
    saveas(fig5, 'results_mpc/paired/Interaccion_DNO_mpc.png');
    print(fig5, 'results_mpc/paired/Interaccion_DNO_mpc', '-depsc'); % Exportar a .eps
    
    % Gráfico 6: Cooperación Hídrica
    fig6 = figure('Name', 'Cooperación Hídrica (MPC)');
    plot(t, Q_t, 'LineWidth', lineWidth);
    hold on; yline(0, 'k--', 'LineWidth', 1);
    title('Cooperación Hídrica: Intercambio de Agua (Q_t)');
    xlabel('Tiempo [horas]'); ylabel('Caudal de Intercambio [L/s]');
    grid on; legend(leyendas, 'Location', 'best'); set(gca, 'FontSize', fontSizeLabels);
    saveas(fig6, 'results_mpc/individual/Cooperacion_hidrica_mpc.png');
    print(fig6, 'results_mpc/individual/Cooperacion_hidrica_mpc', '-depsc'); % Exportar a .eps
    
    close all;
    fprintf('Exportación completada.\n');
end