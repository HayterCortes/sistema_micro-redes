function plot_resultados_mpc(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Q_t, h_p)
    
    %% --- 1. Definición de Estilos y Parámetros ---
    t = (0:size(SoC, 1)-1) * mg(1).Ts_sim / 3600; 
    n_mg = length(mg);
    leyendas = {mg.nombre};
    fontSizeLabels = 12;
    lineWidth = 1.5;
    
    colores = [0, 0.4470, 0.7410;   % Azul
               0.8500, 0.3250, 0.0980; % Naranja
               0.9290, 0.6940, 0.1250];% Amarillo
    if ~exist('results_mpc', 'dir'), mkdir('results_mpc'); end
    
    fprintf('Generando y exportando gráficos en formato PNG, EPS y FIG...\n');
    
    %% --- 2. Generación de Gráficos (Formato Corregido) ---
    
    % Gráfico 1: Estado de Carga (SoC) de Baterías
    fig1 = figure('Name', 'SoC Baterías (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Estado de Carga de las Baterías ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, SoC(:, i) * 100, 'LineWidth', lineWidth, 'Color', colores(i,:));
        hold on;
        yline(mg(i).SoC_min * 100, 'r--', 'LineWidth', 1);
        yline(mg(i).SoC_max * 100, 'r--', 'LineWidth', 1);
        hold off;
        title(leyendas{i});
        ylabel('SoC [%]');
        ylim([0 100]);
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename1 = 'results_mpc/SoC_baterias_mpc';
    saveas(fig1, [filename1 '.png']);
    print(fig1, filename1, '-depsc');
    savefig(fig1, [filename1 '.fig']);
    
    % Gráfico 2: Volumen en Estanques
    fig2 = figure('Name', 'Volumen Estanques (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Volumen de Agua en Estanques ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, V_tank(:, i) / 1000, 'LineWidth', lineWidth, 'Color', colores(i,:));
        hold on;
        yline(mg(i).V_max / 1000, 'r--', 'LineWidth', 1);
        hold off;
        title(leyendas{i});
        ylabel('Volumen [m^3]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename2 = 'results_mpc/Volumen_estanques_mpc';
    saveas(fig2, [filename2 '.png']);
    print(fig2, filename2, '-depsc');
    savefig(fig2, [filename2 '.fig']);

    % Gráfico 3: Volumen del Acuífero
    fig3 = figure('Name', 'Volumen Acuífero (MPC)');
    plot(t, V_aq / 1000, 'k', 'LineWidth', lineWidth);
    title('Volumen del Acuífero Compartido ');
    xlabel('Tiempo [horas]'); ylabel('Volumen [m^3]'); grid on;
    set(gca, 'FontSize', fontSizeLabels);
    filename3 = 'results_mpc/Volumen_acuifero_mpc';
    saveas(fig3, [filename3 '.png']);
    print(fig3, filename3, '-depsc');
    savefig(fig3, [filename3 '.fig']);

    % Gráfico 4: Caudal de Bombeo
    fig4 = figure('Name', 'Caudal de Bombeo (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Caudal Extraído por Bombas Eléctricas ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, Q_p(:, i), 'LineWidth', lineWidth, 'Color', colores(i,:));
        title(leyendas{i});
        ylabel('Caudal [L/s]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename4 = 'results_mpc/Caudal_bombeo_mpc';
    saveas(fig4, [filename4 '.png']);
    print(fig4, filename4, '-depsc');
    savefig(fig4, [filename4 '.fig']);

    % Gráfico 5: Potencia de Bombeo
    fig5 = figure('Name', 'Potencia de Bombeo (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Potencia Consumida por Bombas Eléctricas ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, P_pump(:, i), 'LineWidth', lineWidth, 'Color', colores(i,:));
        title(leyendas{i});
        ylabel('Potencia [kW]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename5 = 'results_mpc/Potencia_bombeo_mpc';
    saveas(fig5, [filename5 '.png']);
    print(fig5, filename5, '-depsc');
    savefig(fig5, [filename5 '.fig']);
    
    % Gráfico 6: Energía Comprada al DNO
    fig6 = figure('Name', 'Energía Comprada (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Potencia Eléctrica Intercambiada con la Red ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, P_grid(:, i), 'LineWidth', lineWidth, 'Color', colores(i,:));
        title(leyendas{i});
        ylabel('Potencia [kW]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename6 = 'results_mpc/Energia_comprada_mpc';
    saveas(fig6, [filename6 '.png']);
    print(fig6, filename6, '-depsc');
    savefig(fig6, [filename6 '.fig']);
    
    % Gráfico 7: Agua Comprada al DNO
    fig7 = figure('Name', 'Agua Comprada (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Caudal de Agua Comprado a la Red ');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, Q_DNO(:, i), 'LineWidth', lineWidth, 'Color', colores(i,:));
        title(leyendas{i});
        ylabel('Caudal [L/s]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename7 = 'results_mpc/Agua_comprada_mpc';
    saveas(fig7, [filename7 '.png']);
    print(fig7, filename7, '-depsc');
    savefig(fig7, [filename7 '.fig']);
    
    % Gráfico 8: Cooperación Hídrica (Intercambio Qt)
    fig8 = figure('Name', 'Cooperación Hídrica (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Cooperación Hídrica: Intercambio de Agua (Qt)');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        plot(t, Q_t(:, i), 'LineWidth', lineWidth, 'Color', colores(i,:));
        hold on; yline(0, 'k--', 'LineWidth', 1); hold off;
        title(leyendas{i});
        ylabel('Caudal [L/s]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename8 = 'results_mpc/Cooperacion_hidrica_mpc';
    saveas(fig8, [filename8 '.png']);
    print(fig8, filename8, '-depsc');
    savefig(fig8, [filename8 '.fig']);
    
    %% --- 3. NUEVOS GRÁFICOS SUGERIDOS ---
    
    % NUEVO Gráfico 9: Descenso del Nivel del Pozo (Drawdown)
    fig9 = figure('Name', 'Descenso Pozo (MPC)', 'Position', [100, 100, 800, 700]);
    sgtitle('Descenso del Nivel del Pozo (Drawdown)');
    for i = 1:n_mg
        subplot(n_mg, 1, i);
        s_pozo = h_p(:, i) - mg(i).h_p0; % s = h_p(t) - h_p(0)
        plot(t, s_pozo, 'LineWidth', lineWidth, 'Color', colores(i,:));
        hold on;
        % Usamos mg(1).s_max porque es un parámetro global e igual para todas las MG
        yline(mg(1).s_max, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Límite s_{max}');
        hold off;
        title(leyendas{i});
        ylabel('Descenso [m]');
        grid on;
        set(gca, 'FontSize', fontSizeLabels);
        xlim([t(1) t(end)]);
    end
    xlabel('Tiempo [horas]');
    filename9 = 'results_mpc/Descenso_pozo_mpc';
    saveas(fig9, [filename9 '.png']);
    print(fig9, filename9, '-depsc');
    savefig(fig9, [filename9 '.fig']);

    % NUEVO Gráfico 10: Recursos Totales Comprados al DNO
    fig10 = figure('Name', 'Recursos Totales DNO (MPC)', 'Position', [100, 100, 800, 600]);
    sgtitle('Recursos Totales Comprados a la Red Externa (DNO)');
    
    % Subplot para Potencia Eléctrica
    subplot(2, 1, 1);
    P_DNO_total = sum(P_grid, 2);
    plot(t, P_DNO_total, 'k', 'LineWidth', lineWidth);
    hold on;
    yline(mg(1).P_grid_max, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Límite P_{grid,max}');
    hold off;
    title('Potencia Eléctrica Total');
    ylabel('Potencia [kW]');
    grid on;
    set(gca, 'FontSize', fontSizeLabels);
    xlim([t(1) t(end)]);
    
    % Subplot para Caudal de Agua
    subplot(2, 1, 2);
    Q_DNO_total = sum(Q_DNO, 2);
    plot(t, Q_DNO_total, 'b', 'LineWidth', lineWidth);
    title('Caudal de Agua Total');
    ylabel('Caudal [L/s]');
    grid on;
    set(gca, 'FontSize', fontSizeLabels);
    xlim([t(1) t(end)]);
    xlabel('Tiempo [horas]');
    filename10 = 'results_mpc/Recursos_totales_DNO_mpc';
    saveas(fig10, [filename10 '.png']);
    print(fig10, filename10, '-depsc');
    savefig(fig10, [filename10 '.fig']);

    % NUEVO Gráfico 11: Costo de Operación Acumulado
    fig11 = figure('Name', 'Costo Acumulado (MPC)');
    C_p = 110; % [CLP/kWh]
    C_q = 644; % [CLP/m^3]
    
    costo_energia_inst = (sum(P_grid, 2) * (mg(1).Ts_sim / 3600)) * C_p;
    costo_agua_inst = (sum(Q_DNO, 2) * mg(1).Ts_sim / 1000) * C_q;
    costo_total_acumulado = cumsum(costo_energia_inst + costo_agua_inst);
    
    plot(t, costo_total_acumulado, 'Color', [0.4940 0.1840 0.5560], 'LineWidth', lineWidth+0.5);
    title('Costo de Operación Acumulado (Energía + Agua)');
    xlabel('Tiempo [horas]');
    ylabel('Costo Acumulado [CLP]');
    grid on;
    set(gca, 'FontSize', fontSizeLabels);
    xlim([t(1) t(end)]);
    
    filename11 = 'results_mpc/Costo_operacion_acumulado_mpc';
    saveas(fig11, [filename11 '.png']);
    print(fig11, filename11, '-depsc');
    savefig(fig11, [filename11 '.fig']);
    
    close all;
    fprintf('Exportación completada.\n');
end