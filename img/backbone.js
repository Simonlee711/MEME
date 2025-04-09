import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar
} from 'recharts';

export default function SideBySideModelPerformance() {
  // Model names
  const models = [
    'clinicalbert (Apr. 2019)',
    'bio_clinicalbert (June 2019)',
    'biobert (Oct. 2019)',
    'medbert (Dec. 2022)'
  ];
  
  // F1 Scores and confidence intervals
  const f1Data = [
    { name: models[0], score: 0.894, errorMinus: 0.002, errorPlus: 0.003 },
    { name: models[1], score: 0.933, errorMinus: 0.003, errorPlus: 0.002 },
    { name: models[2], score: 0.936, errorMinus: 0.003, errorPlus: 0.002 },
    { name: models[3], score: 0.943, errorMinus: 0.003, errorPlus: 0.002 }
  ];

  // AUROC Scores and confidence intervals
  const aurocData = [
    { name: models[0], score: 0.961, errorMinus: 0.002, errorPlus: 0.001 },
    { name: models[1], score: 0.987, errorMinus: 0.001, errorPlus: 0.001 },
    { name: models[2], score: 0.988, errorMinus: 0.000, errorPlus: 0.001 },
    { name: models[3], score: 0.991, errorMinus: 0.001, errorPlus: 0.000 }
  ];

  // AUPRC Scores and confidence intervals
  const auprcData = [
    { name: models[0], score: 0.922, errorMinus: 0.004, errorPlus: 0.004 },
    { name: models[1], score: 0.977, errorMinus: 0.002, errorPlus: 0.001 },
    { name: models[2], score: 0.979, errorMinus: 0.001, errorPlus: 0.002 },
    { name: models[3], score: 0.983, errorMinus: 0.001, errorPlus: 0.002 }
  ];

  // Render a single chart with error bars and data labels
  const renderChart = (data, color, title) => {
    return (
      <div className="flex flex-col items-center w-full h-full">
        <h3 className="text-xl font-semibold mb-2">{title}</h3>
        <ResponsiveContainer width="95%" height="100%">
          <LineChart
            data={data}
            margin={{
              top: 30,
              right: 20,
              left: 20,
              bottom: 120 // Very large bottom margin to ensure labels are visible
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 12 }}
              interval={0}
              angle={-45}
              textAnchor="end"
              height={120} // Extra tall height for X-axis
              tickMargin={35} // Large margin between axis and labels
            />
            <YAxis 
              domain={[0.88, 1.0]}
              tick={{ fontSize: 12 }}
              width={50}
              tickCount={6}
              label={{ value: 'Score', angle: -90, position: 'insideLeft', dy: 15, fontSize: 14 }}
            />
            <Tooltip 
              formatter={(value) => [value.toFixed(3), title]}
              labelFormatter={(label) => `Model: ${label}`}
              contentStyle={{ fontSize: 14 }}
            />
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke={color} 
              strokeWidth={2.5}
              isAnimationActive={false}
              dot={{ strokeWidth: 2.5, r: 5, stroke: color, fill: "white" }}
              activeDot={{ r: 7, stroke: color, fill: color }}
              label={({ x, y, value }) => (
                <text 
                  x={x} 
                  y={y - 15} 
                  fill={color} 
                  textAnchor="middle" 
                  fontSize={13}
                  fontWeight="bold"
                >
                  {value.toFixed(3)}
                </text>
              )}
            >
              <ErrorBar dataKey="errorMinus" direction="minus" stroke={color} strokeWidth={1.5} />
              <ErrorBar dataKey="errorPlus" direction="plus" stroke={color} strokeWidth={1.5} />
            </Line>
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center w-full">
      <h2 className="text-3xl font-bold mb-6">Performance on ED Disposition: MIMIC Dataset</h2>
      
      {/* Container with extra height to accommodate everything */}
      <div className="flex w-full" style={{ height: "550px" }}>
        <div className="flex-1">
          {renderChart(f1Data, '#1f77b4', 'F1 Score')}
        </div>
        <div className="flex-1">
          {renderChart(aurocData, '#ff7f0e', 'AUROC Score')}
        </div>
        <div className="flex-1">
          {renderChart(auprcData, '#2ca02c', 'AUPRC Score')}
        </div>
      </div>
      
      <div className="mt-6 text-sm text-gray-600 px-6">
        <p>Data shows model performance metrics with confidence intervals. Higher scores indicate better performance.</p>
        <p>Note: Y-axis starts at 0.88 to better visualize differences between models.</p>
      </div>
    </div>
  );
}
