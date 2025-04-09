<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Metrics for Fewshot Learning</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        .chart-container {
            display: flex;
            flex-direction: row;
            width: 100%;
            height: 500px;
        }
        .chart-wrapper {
            flex: 1;
            margin: 10px;
        }
        h2 {
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .legend-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 0 10px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <h2>Performance Metrics for Fewshot Learning on Multiple Tasks</h2>
    <div class="chart-container">
        <div class="chart-wrapper">
            <canvas id="f1Chart"></canvas>
        </div>
        <div class="chart-wrapper">
            <canvas id="aurocChart"></canvas>
        </div>
        <div class="chart-wrapper">
            <canvas id="auprcChart"></canvas>
        </div>
    </div>
    <div class="legend-container" id="legendContainer"></div>

    <script>
        // Dataset based on the provided JSON
        const data = {
            "f1": {
                "ed_disposition": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.6217749705598973, 0.5944299390774587, 0.6172845163359758, 0.6371492020389142, 0.6543770133320793, 0.6956205492790859, 0.7034644270370669, 0.9467932759038533, 0.9973820317987357, 0.9992958948536315, 1.0]
                },
                "home": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.6119954588275079, 0.6127587459145382, 0.6084315393026364, 0.6117681201480084, 0.6121600386800435, 0.611707905853953, 0.6302921523880014, 0.6379901477832512, 0.6648530766868026, 0.6835268956055042, 0.685747497669895]
                },
                "icu": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.3976533137776148, 0.38450129773822767, 0.3867876721794813, 0.39053092501368364, 0.3906501633460636, 0.39045214597199907, 0.37298231979794055, 0.3987376129263342, 0.4606088623294622, 0.46832266325224076, 0.46760743801652893]
                },
                "mortality": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.00299625468164794, 0.0044943820224719105, 0.004514672686230248, 0.006535947712418301, 0.0014781966001478197, 0.004542013626040878, 0.05417533432392273, 0.1014096301465457, 0.17984536082474228, 0.2396638655462185, 0.2384726368159204]
                }
            },
            "auroc": {
                "ed_disposition": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.8331751387000096, 0.8524402214246325, 0.8758914554486945, 0.8877036114771677, 0.9029597010033809, 0.9228071224714497, 0.982449764349422, 0.9972445859114419, 0.9999162814069511, 0.999999941389758, 1.0]
                },
                "home": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.7320197607959118, 0.7325578220316381, 0.7300885127569929, 0.7288772764664462, 0.7303480688130366, 0.7313692001174366, 0.7562719616014966, 0.7708798210039168, 0.7920888797471994, 0.8092640380910886, 0.8111844459757406]
                },
                "icu": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.7499439929574466, 0.7427725680822402, 0.7454948883613645, 0.7445050831888946, 0.7487157584469772, 0.7451332121670347, 0.755794602569352, 0.7749220879419599, 0.7958997759909787, 0.8080998954986865, 0.8139794415736089]
                },
                "mortality": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.807724514060694, 0.8020296670162207, 0.8140812777053323, 0.8145272709706288, 0.8151568241326344, 0.8107938308299635, 0.8284775943112, 0.8339475231216982, 0.8550682097763662, 0.8716714988752812, 0.8732150167887051]
                }
            },
            "auprc": {
                "ed_disposition": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.6235169874234565, 0.6438753510849058, 0.681218580514696, 0.7046478360153747, 0.7360227450213864, 0.7837265151395221, 0.9403942155184136, 0.991488579834929, 0.9998934585528677, 0.9999998461899737, 1.0]
                },
                "home": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.6347392408563749, 0.6387365540522296, 0.6357871788786316, 0.6323539598463719, 0.6343968141203948, 0.6405503383053318, 0.6631523541460649, 0.6668069890386276, 0.6928801835450464, 0.7124913754381411, 0.7098074936295129]
                },
                "icu": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.3917643102023639, 0.37943290615216746, 0.3805038362379721, 0.3940411118552274, 0.3908305957063051, 0.39052283003627336, 0.3949519986995886, 0.4130756171129976, 0.4386675861091139, 0.4512105141952312, 0.44945370606420926]
                },
                "mortality": {
                    "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
                    "means": [0.1641633401886139, 0.16572714673446454, 0.16255615378152563, 0.14684580342083065, 0.17062475100626856, 0.15852797154283538, 0.19137984397398386, 0.19903435661034694, 0.24363885284931047, 0.258833804037808, 0.25457642404627745]
                }
            }
        };

        // Define colors and metrics
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'];
        const outcomes = ['ed_disposition', 'home', 'icu', 'mortality'];
        const outcomeTitles = ['ED Disposition', 'Home', 'ICU', 'Mortality'];
        const metrics = ['f1', 'auroc', 'auprc'];
        const metricTitles = ['F1', 'AUROC', 'AUPRC'];

        // Apply increments for specific outcomes as in the Python code
        const applyIncrements = (outcome, values) => {
            if (outcome !== 'ed_disposition') {
                const increments = {
                    128: 0.02, 256: 0.04, 512: 0.06, 1024: 0.08, 'all': 0.083
                };
                const xValues = data[metrics[0]][outcome].x_values;
                const result = [...values];
                
                xValues.forEach((shot, idx) => {
                    if (increments[shot]) {
                        result[idx] += increments[shot];
                    }
                });
                return result;
            }
            return values;
        };

        // Create charts for each metric
        metrics.forEach((metric, metricIdx) => {
            const chartData = {
                labels: data[metric][outcomes[0]].x_values,
                datasets: outcomes.map((outcome, outcomeIdx) => {
                    const adjustedMeans = applyIncrements(outcome, data[metric][outcome].means);
                    return {
                        label: outcomeTitles[outcomeIdx],
                        data: adjustedMeans,
                        borderColor: colors[outcomeIdx],
                        backgroundColor: colors[outcomeIdx],
                        pointStyle: ['circle', 'rect', 'triangle', 'rectRot'][outcomeIdx],
                        pointRadius: 6,
                        borderWidth: 2,
                        tension: 0.1
                    };
                })
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: metricTitles[metricIdx],
                            font: {
                                size: 18
                            }
                        }
                    },
                    scales: {
                        y: {
                            min: 0,
                            max: 1.1,
                            ticks: {
                                stepSize: 0.2,
                                font: {
                                    size: 14
                                }
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        },
                        x: {
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45,
                                font: {
                                    size: 14
                                }
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        }
                    }
                }
            };

            const ctx = document.getElementById(`${metric}Chart`).getContext('2d');
            new Chart(ctx, config);
        });

        // Create a custom legend
        const legendContainer = document.getElementById('legendContainer');
        outcomes.forEach((outcome, idx) => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            
            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = colors[idx];
            
            const label = document.createElement('span');
            label.textContent = outcomeTitles[idx];
            
            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legendContainer.appendChild(legendItem);
        });
    </script>
</body>
</html>
