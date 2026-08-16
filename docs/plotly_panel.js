/* ================================================================
   Radar de Valorización Urbana — Panel de Visualizaciones Plotly
   Módulo: plotly_panel.js
   ================================================================ */

const URBAN_COLORS = {
    teal:      '#2a9d8f',
    tealLight: '#5cb8b0',
    tealDark:  '#1d6b62',
    orange:    '#f4823c',
    orangeLight:'#f9b87a',
    orangeDark:'#c45f1a',
    gray:      '#4a4f5c',
    grayLight: '#7a8090',
    grayDark:  '#353944',
    success:   '#4ade80',
    danger:    '#ef4444',
    warning:   '#fbbf24',
    bg:        '#1e2026',
    text:      '#e8eaed',
    grid:      '#353944'
};

const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        color: URBAN_COLORS.text,
        family: 'Segoe UI, system-ui, sans-serif',
        size: 12
    },
    margin: { t: 40, r: 20, b: 50, l: 60 }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
    displaylogo: false
};

/* ================================================================
   renderShapWaterfall — Gráfico Waterfall para valores SHAP
   ================================================================ */
function renderShapWaterfall(elementId, shapData) {
    const el = document.getElementById(elementId);
    if (!el || !shapData) return;

    const features = shapData.features || [];
    const values   = shapData.values || [];
    const baseValue = shapData.base_value || 0;
    const prediction = shapData.prediction || 0;

    if (features.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos SHAP disponibles</p>';
        return;
    }

    const cumulative = [];
    let running = baseValue;
    cumulative.push(baseValue);
    for (let i = 0; i < values.length; i++) {
        running += values[i];
        cumulative.push(running);
    }

    const traces = [];

    for (let i = 0; i < features.length; i++) {
        const isPositive = values[i] >= 0;
        const yStart = cumulative[i];
        const yEnd = cumulative[i + 1];

        traces.push({
            type: 'bar',
            x: [features[i]],
            y: [Math.abs(yEnd - yStart)],
            base: Math.min(yStart, yEnd),
            marker: {
                color: isPositive ? URBAN_COLORS.success : URBAN_COLORS.danger,
                opacity: 0.85
            },
            text: [`${isPositive ? '+' : ''}${values[i].toFixed(4)}`],
            textposition: 'outside',
            textfont: { size: 10, color: URBAN_COLORS.grayLight },
            name: features[i],
            showlegend: false,
            hovertemplate: `<b>${features[i]}</b><br>SHAP: ${values[i].toFixed(4)}<br>Contribución: ${((values[i] / Math.abs(prediction - baseValue || 1)) * 100).toFixed(1)}%<extra></extra>`
        });
    }

    traces.push({
        type: 'bar',
        x: ['Valor Base'],
        y: [baseValue],
        marker: { color: URBAN_COLORS.gray },
        text: [baseValue.toFixed(4)],
        textposition: 'outside',
        textfont: { size: 10, color: URBAN_COLORS.grayLight },
        name: 'Base',
        showlegend: false
    });

    traces.push({
        type: 'bar',
        x: ['Predicción'],
        y: [prediction],
        marker: { color: URBAN_COLORS.orange },
        text: [prediction.toFixed(4)],
        textposition: 'outside',
        textfont: { size: 10, color: URBAN_COLORS.orangeLight },
        name: 'Predicción',
        showlegend: false
    });

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Contribución SHAP por Característica',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            tickangle: -35,
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        yaxis: {
            title: 'Valor SHAP',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        barmode: 'stack',
        bargap: 0.3
    };

    Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderValuationChart — Gráfico de barras comparando valorizaciones
   ================================================================ */
function renderValuationChart(elementId, data) {
    const el = document.getElementById(elementId);
    if (!el || !data) return;

    const cellIds = data.cell_ids || [];
    const values  = data.values || [];
    const horizon = data.horizon || 12;

    if (cellIds.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Seleccione celdas para comparar</p>';
        return;
    }

    const barColors = values.map(v => {
        if (v > 0.15) return URBAN_COLORS.success;
        if (v > 0.05) return URBAN_COLORS.teal;
        if (v > 0) return URBAN_COLORS.warning;
        return URBAN_COLORS.danger;
    });

    const trace = {
        type: 'bar',
        x: cellIds.map(c => `Celda ${c}`),
        y: values.map(v => (v * 100).toFixed(2)),
        marker: {
            color: barColors,
            line: { color: URBAN_COLORS.tealDark, width: 1 }
        },
        text: values.map(v => `${(v * 100).toFixed(2)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 11 },
        hovertemplate: '<b>Celda %{x}</b><br>Valorización: %{y}%<br>Horizonte: ' + horizon + ' meses<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: `Predicción de Valorización (Horizonte: ${horizon} meses)`,
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Celda',
            gridcolor: URBAN_COLORS.grid,
            tickangle: cellIds.length > 6 ? -35 : 0
        },
        yaxis: {
            title: 'Valorización Predicha (%)',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        bargap: 0.35
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderScenarioChart — Comparación de escenario base vs simulado
   ================================================================ */
function renderScenarioChart(elementId, baseline, scenario) {
    const el = document.getElementById(elementId);
    if (!el || !baseline || !scenario) return;

    const cellIds = baseline.cell_ids || [];
    const baseVals = baseline.values || [];
    const scenVals = scenario.values || [];

    if (cellIds.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Ejecute la simulación para ver resultados</p>';
        return;
    }

    const traceBase = {
        type: 'bar',
        name: 'Escenario Base',
        x: cellIds.map(c => `Celda ${c}`),
        y: baseVals.map(v => (v * 100).toFixed(2)),
        marker: { color: URBAN_COLORS.teal, opacity: 0.8 },
        text: baseVals.map(v => `${(v * 100).toFixed(1)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.tealLight, size: 9 },
        hovertemplate: '<b>Base — Celda %{x}</b><br>Valorización: %{y}%<extra></extra>'
    };

    const traceScenario = {
        type: 'bar',
        name: 'Escenario Simulado',
        x: cellIds.map(c => `Celda ${c}`),
        y: scenVals.map(v => (v * 100).toFixed(2)),
        marker: { color: URBAN_COLORS.orange, opacity: 0.85 },
        text: scenVals.map(v => `${(v * 100).toFixed(1)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.orangeLight, size: 9 },
        hovertemplate: '<b>Simulado — Celda %{x}</b><br>Valorización: %{y}%<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Comparación: Escenario Base vs Simulado',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Celda',
            gridcolor: URBAN_COLORS.grid,
            tickangle: cellIds.length > 6 ? -35 : 0
        },
        yaxis: {
            title: 'Valorización Predicha (%)',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        barmode: 'group',
        bargroupgap: 0.15,
        legend: {
            orientation: 'h',
            y: -0.3,
            font: { color: URBAN_COLORS.grayLight }
        }
    };

    Plotly.newPlot(el, [traceBase, traceScenario], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderFeatureImportance — Gráfico de barras horizontales
   ================================================================ */
function renderFeatureImportance(elementId, importanceData) {
    const el = document.getElementById(elementId);
    if (!el || !importanceData) return;

    let features = importanceData.features || [];
    let importances = importanceData.importances || [];

    if (features.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos de importancia disponibles</p>';
        return;
    }

    const paired = features.map((f, i) => ({ feature: f, importance: importances[i] }));
    paired.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));

    features = paired.map(p => p.feature);
    importances = paired.map(p => p.importance);

    const colors = importances.map(v => v >= 0 ? URBAN_COLORS.teal : URBAN_COLORS.orange);

    const trace = {
        type: 'bar',
        orientation: 'h',
        y: features,
        x: importances.map(v => Math.abs(v).toFixed(4)),
        marker: {
            color: colors,
            line: { color: URBAN_COLORS.grayDark, width: 1 }
        },
        text: importances.map(v => {
            const sign = v >= 0 ? '+' : '';
            return `${sign}${(v * 100).toFixed(2)}%`;
        }),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 10 },
        hovertemplate: '<b>%{y}</b><br>Importancia: %{x}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Importancia de Características',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Importancia Relativa',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        yaxis: {
            gridcolor: URBAN_COLORS.grid,
            autorange: 'reversed'
        },
        bargap: 0.4,
        margin: { ...PLOTLY_LAYOUT.margin, l: 140 }
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderIndexDistribution — Distribución del Índice (dona/barras)
   ================================================================ */
function renderIndexDistribution(elementId, distribution) {
    const el = document.getElementById(elementId);
    if (!el || !distribution) return;

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);

    if (values.reduce((a, b) => a + b, 0) === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos de distribución</p>';
        return;
    }

    const trace = {
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.45,
        marker: {
            colors: [URBAN_COLORS.danger, URBAN_COLORS.warning, URBAN_COLORS.teal, URBAN_COLORS.success],
            line: { color: URBAN_COLORS.bg, width: 2 }
        },
        textinfo: 'label+value',
        textfont: { size: 11, color: URBAN_COLORS.text },
        hovertemplate: '<b>%{label}</b><br>Celdas: %{value} (%{percent})<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Celdas por rango del Índice',
            font: { color: URBAN_COLORS.tealLight, size: 13 }
        },
        showlegend: true,
        legend: { orientation: 'h', y: -0.15, font: { size: 10, color: URBAN_COLORS.grayLight } }
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderCityComparison — Índice por ciudad (barras)
   ================================================================ */
function renderCityComparison(elementId, cityAverages, globalAverage) {
    const el = document.getElementById(elementId);
    if (!el || !cityAverages) return;

    const cities = Object.keys(cityAverages);
    const values = Object.values(cityAverages);

    if (cities.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos por ciudad</p>';
        return;
    }

    const trace = {
        type: 'bar',
        x: cities,
        y: values,
        marker: { color: [URBAN_COLORS.orange, URBAN_COLORS.teal] },
        text: values.map(v => v.toFixed(1)),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 12 },
        hovertemplate: '<b>%{x}</b><br>Índice promedio: %{y:.1f}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: `Índice de Inteligencia Urbana por Ciudad (global: ${globalAverage.toFixed(1)})`,
            font: { color: URBAN_COLORS.tealLight, size: 13 }
        },
        yaxis: {
            title: 'Índice (0-100)',
            gridcolor: URBAN_COLORS.grid,
            range: [0, 100],
            zerolinecolor: URBAN_COLORS.grid
        },
        xaxis: { gridcolor: URBAN_COLORS.grid },
        bargap: 0.4
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderTopCells — Top 10 celdas por valorización (barras h)
   ================================================================ */
function renderTopCells(elementId, topCells) {
    const el = document.getElementById(elementId);
    if (!el || !topCells || topCells.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos de top celdas</p>';
        return;
    }

    const shortIds = topCells.map(c => c.cell_id.slice(0, 8) + '…');
    const values = topCells.map(c => (c.annualized_valuation * 100).toFixed(2));

    const trace = {
        type: 'bar',
        orientation: 'h',
        y: shortIds,
        x: values,
        marker: {
            color: topCells.map(c => c.city === 'Quito' ? URBAN_COLORS.orange : URBAN_COLORS.teal),
            line: { color: URBAN_COLORS.grayDark, width: 1 }
        },
        text: values.map(v => v + '%'),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 10 },
        hovertemplate: '<b>%{y}</b><br>Valorización anual: %{x}%<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Top 10 Celdas — Valorización Anualizada',
            font: { color: URBAN_COLORS.tealLight, size: 13 }
        },
        xaxis: { title: 'Valorización anual (%)', gridcolor: URBAN_COLORS.grid },
        yaxis: { autorange: 'reversed', gridcolor: URBAN_COLORS.grid },
        bargap: 0.35,
        margin: { ...PLOTLY_LAYOUT.margin, l: 90 }
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderPriceTrend — Serie histórica de precios (línea)
   ================================================================ */
function renderPriceTrend(elementId, trendData) {
    const el = document.getElementById(elementId);
    if (!el || !trendData || !trendData.series || trendData.series.length === 0) {
        if (el) el.innerHTML = '';
        return;
    }

    const years = trendData.series.map(s => s.year);
    const prices = trendData.series.map(s => s.avg_price);

    const trace = {
        type: 'scatter',
        mode: 'lines+markers',
        x: years,
        y: prices,
        line: { color: URBAN_COLORS.orange, width: 2.5, shape: 'spline' },
        marker: { size: 7, color: URBAN_COLORS.orange, line: { color: '#fff', width: 1 } },
        fill: 'tozeroy',
        fillcolor: 'rgba(244,130,60,0.12)',
        hovertemplate: '<b>%{x}</b><br>Precio prom.: $%{y:,.0f}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: `Evolución de precios (tendencia ${(trendData.price_trend * 100).toFixed(1)}% anual)`,
            font: { color: URBAN_COLORS.tealLight, size: 12 }
        },
        xaxis: { title: 'Año', gridcolor: URBAN_COLORS.grid, dtick: 1 },
        yaxis: {
            title: 'Precio promedio (USD)',
            gridcolor: URBAN_COLORS.grid,
            tickformat: '$,.0f'
        },
        margin: { t: 40, r: 16, b: 40, l: 60 }
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderCityStatsTable — Estadísticas por ciudad
   ================================================================ */
function renderCityStatsTable(elementId, cityStats) {
    const el = document.getElementById(elementId);
    if (!el || !cityStats || cityStats.length === 0) return;

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Ciudad</th>
                    <th>Celdas</th>
                    <th>Transacciones</th>
                    <th>Precio Promedio</th>
                    <th>Valorización Prom.</th>
                </tr>
            </thead>
            <tbody>`;

    for (const stat of cityStats) {
        html += `
                <tr>
                    <td><b>${stat.city}</b></td>
                    <td>${stat.cells}</td>
                    <td>${stat.transactions.toLocaleString()}</td>
                    <td>$${Number(stat.avg_price).toLocaleString('en-US', { maximumFractionDigits: 0 })}</td>
                    <td class="${stat.avg_valuation >= 0 ? 'shap-bar-pos' : 'shap-bar-neg'}">${(stat.avg_valuation * 100).toFixed(2)}%</td>
                </tr>`;
    }

    html += '</tbody></table>';
    el.innerHTML = html;
}
