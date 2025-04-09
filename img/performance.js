import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// Define colors for each model
const modelColors = {
  "Logistic Regression": "#8884d8",
  "XGBoost": "#82ca9d",
  "MLP": "#ffc658",
  "MC-BEC": "#ff8042",
  "EHR-Shot": "#0088fe",
  "Clinical Longformer": "#ff6b6b",
  "MEME": "#00C49F"
};

// Data preparation
const models = ["Logistic Regression", "XGBoost", "MLP", "MC-BEC", "EHR-Shot", "Clinical Longformer", "MEME"];
const tasks = ["ED Disposition", "Discharge", "ICU Requirement", "Mortality"];

// F1 Score data
const f1Data = [
  {
    "model": "Logistic Regression",
    "ED Disposition": 0.799,
    "Discharge": 0.549,
    "ICU Requirement": 0.427,
    "Mortality": 0.095,
    "CI_ED": 0.025,
    "CI_Discharge": 0.033,
    "CI_ICU": 0.036,
    "CI_Mortality": 0.026
  },
  {
    "model": "XGBoost",
    "ED Disposition": 0.833,
    "Discharge": 0.566,
    "ICU Requirement": 0.416,
    "Mortality": 0.043,
    "CI_ED": 0.022,
    "CI_Discharge": 0.030,
    "CI_ICU": 0.057,
    "CI_Mortality": 0.019
  },
  {
    "model": "MLP",
    "ED Disposition": 0.841,
    "Discharge": 0.612,
    "ICU Requirement": 0.502,
    "Mortality": 0.097,
    "CI_ED": 0.010,
    "CI_Discharge": 0.013,
    "CI_ICU": 0.019,
    "CI_Mortality": 0.023
  },
  {
    "model": "MC-BEC",
    "ED Disposition": 0.912,
    "Discharge": 0.653,
    "ICU Requirement": 0.545,
    "Mortality": 0.127,
    "CI_ED": 0.002,
    "CI_Discharge": 0.006,
    "CI_ICU": 0.006,
    "CI_Mortality": 0.014
  },
  {
    "model": "EHR-Shot",
    "ED Disposition": 0.874,
    "Discharge": 0.691,
    "ICU Requirement": 0.560,
    "Mortality": 0.036,
    "CI_ED": 0.003,
    "CI_Discharge": 0.008,
    "CI_ICU": 0.008,
    "CI_Mortality": 0.003
  },
  {
    "model": "Clinical Longformer",
    "ED Disposition": 0.893,
    "Discharge": 0.679,
    "ICU Requirement": 0.547,
    "Mortality": 0.110,
    "CI_ED": 0.002,
    "CI_Discharge": 0.007,
    "CI_ICU": 0.008,
    "CI_Mortality": 0.009
  },
  {
    "model": "MEME",
    "ED Disposition": 0.943,
    "Discharge": 0.698,
    "ICU Requirement": 0.572,
    "Mortality": 0.137,
    "CI_ED": 0.003,
    "CI_Discharge": 0.007,
    "CI_ICU": 0.014,
    "CI_Mortality": 0.035
  }
];

// AUROC data
const aurocData = [
  {
    "model": "Logistic Regression",
    "ED Disposition": 0.863,
    "Discharge": 0.852,
    "ICU Requirement": 0.807,
    "Mortality": 0.768,
    "CI_ED": 0.012,
    "CI_Discharge": 0.014,
    "CI_ICU": 0.017,
    "CI_Mortality": 0.019
  },
  {
    "model": "XGBoost",
    "ED Disposition": 0.909,
    "Discharge": 0.862,
    "ICU Requirement": 0.894,
    "Mortality": 0.845,
    "CI_ED": 0.010,
    "CI_Discharge": 0.016,
    "CI_ICU": 0.016,
    "CI_Mortality": 0.016
  },
  {
    "model": "MLP",
    "ED Disposition": 0.871,
    "Discharge": 0.802,
    "ICU Requirement": 0.767,
    "Mortality": 0.786,
    "CI_ED": 0.018,
    "CI_Discharge": 0.011,
    "CI_ICU": 0.011,
    "CI_Mortality": 0.013
  },
  {
    "model": "MC-BEC",
    "ED Disposition": 0.968,
    "Discharge": 0.708,
    "ICU Requirement": 0.818,
    "Mortality": 0.815,
    "CI_ED": 0.020,
    "CI_Discharge": 0.006,
    "CI_ICU": 0.014,
    "CI_Mortality": 0.006
  },
  {
    "model": "EHR-Shot",
    "ED Disposition": 0.790,
    "Discharge": 0.743,
    "ICU Requirement": 0.821,
    "Mortality": 0.827,
    "CI_ED": 0.031,
    "CI_Discharge": 0.007,
    "CI_ICU": 0.018,
    "CI_Mortality": 0.009
  },
  {
    "model": "Clinical Longformer",
    "ED Disposition": 0.888,
    "Discharge": 0.739,
    "ICU Requirement": 0.819,
    "Mortality": 0.811,
    "CI_ED": 0.003,
    "CI_Discharge": 0.007,
    "CI_ICU": 0.008,
    "CI_Mortality": 0.007
  },
  {
    "model": "MEME",
    "ED Disposition": 0.991,
    "Discharge": 0.799,
    "ICU Requirement": 0.870,
    "Mortality": 0.862,
    "CI_ED": 0.001,
    "CI_Discharge": 0.006,
    "CI_ICU": 0.015,
    "CI_Mortality": 0.006
  }
];

// AUPRC data
const auprcData = [
  {
    "model": "Logistic Regression",
    "ED Disposition": 0.874,
    "Discharge": 0.628,
    "ICU Requirement": 0.618,
    "Mortality": 0.051,
    "CI_ED": 0.027,
    "CI_Discharge": 0.036,
    "CI_ICU": 0.034,
    "CI_Mortality": 0.034
  },
  {
    "model": "XGBoost",
    "ED Disposition": 0.912,
    "Discharge": 0.642,
    "ICU Requirement": 0.630,
    "Mortality": 0.128,
    "CI_ED": 0.011,
    "CI_Discharge": 0.035,
    "CI_ICU": 0.046,
    "CI_Mortality": 0.013
  },
  {
    "model": "MLP",
    "ED Disposition": 0.866,
    "Discharge": 0.630,
    "ICU Requirement": 0.581,
    "Mortality": 0.077,
    "CI_ED": 0.018,
    "CI_Discharge": 0.024,
    "CI_ICU": 0.026,
    "CI_Mortality": 0.033
  },
  {
    "model": "MC-BEC",
    "ED Disposition": 0.935,
    "Discharge": 0.657,
    "ICU Requirement": 0.608,
    "Mortality": 0.174,
    "CI_ED": 0.003,
    "CI_Discharge": 0.009,
    "CI_ICU": 0.009,
    "CI_Mortality": 0.025
  },
  {
    "model": "EHR-Shot",
    "ED Disposition": 0.878,
    "Discharge": 0.655,
    "ICU Requirement": 0.655,
    "Mortality": 0.246,
    "CI_ED": 0.007,
    "CI_Discharge": 0.012,
    "CI_ICU": 0.017,
    "CI_Mortality": 0.030
  },
  {
    "model": "Clinical Longformer",
    "ED Disposition": 0.902,
    "Discharge": 0.634,
    "ICU Requirement": 0.642,
    "Mortality": 0.211,
    "CI_ED": 0.002,
    "CI_Discharge": 0.010,
    "CI_ICU": 0.011,
    "CI_Mortality": 0.014
  },
  {
    "model": "MEME",
    "ED Disposition": 0.983,
    "Discharge": 0.765,
    "ICU Requirement": 0.709,
    "Mortality": 0.243,
    "CI_ED": 0.002,
    "CI_Discharge": 0.008,
    "CI_ICU": 0.012,
    "CI_Mortality": 0.034
  }
];

// Custom tooltip component
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const metric = payload[0].dataKey;
    const value = payload[0].value;
    const modelName = label;
    
    // Find the CI key corresponding to this metric
    const ciKey = `CI_${metric.split(' ').pop()}`;
    const ciValue = payload[0].payload[ciKey];
    
    return (
      <div className="bg-white p-3 border border-gray-200 rounded shadow-md">
        <p className="font-bold">{modelName}</p>
        <p>{`${metric}: ${value.toFixed(3)}`}</p>
        <p className="text-sm text-gray-600">{`95% CI: ±${ciValue?.toFixed(3) || 'N/A'}`}</p>
      </div>
    );
  }
  return null;
};

// Function to get the best two models for a specific task
const getBestModels = (data, task) => {
  const sortedModels = [...data].sort((a, b) => b[task] - a[task]);
  return [sortedModels[0].model, sortedModels[1].model];
};

// Function to determine bar style based on performance ranking
const getBarStyle = (modelName, task, data) => {
  const bestModels = getBestModels(data, task);
  
  if (modelName === bestModels[0]) {
    return { fill: modelColors[modelName], stroke: "#000", strokeWidth: 2 };
  } else if (modelName === bestModels[1]) {
    return { fill: modelColors[modelName], stroke: "#555", strokeWidth: 1, strokeDasharray: "3 3" };
  } else {
    return { fill: modelColors[modelName] };
  }
};

const ModelPerformanceBarCharts = () => {
  const [selectedMetric, setSelectedMetric] = useState("F1");
  
  // Get data based on selected metric
  const getData = () => {
    switch(selectedMetric) {
      case "F1": return f1Data;
      case "AUROC": return aurocData;
      case "AUPRC": return auprcData;
      default: return f1Data;
    }
  };
  
  const currentData = getData();
  
  // Custom renderer for bars to highlight best performers
  const renderCustomBar = (props) => {
    const { x, y, width, height, index, dataKey } = props;
    const modelName = currentData[index].model;
    const customStyle = getBarStyle(modelName, dataKey, currentData);
    
    return <rect 
      x={x} 
      y={y} 
      width={width} 
      height={height} 
      {...customStyle} 
    />;
  };

  return (
    <div className="flex flex-col bg-white p-6 rounded-lg shadow-lg">
      <h1 className="text-2xl font-bold text-center mb-4">MIMIC Dataset Performance Comparison</h1>
      
      {/* Metric Selection */}
      <div className="flex justify-center mb-6">
        <div className="flex flex-col">
          <label className="text-sm font-medium mb-1 text-center">Select Metric:</label>
          <select 
            className="p-2 border rounded bg-white"
            value={selectedMetric} 
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            <option value="F1">F1 Scores</option>
            <option value="AUROC">AUROC Scores</option>
            <option value="AUPRC">AUPRC Scores</option>
          </select>
        </div>
      </div>
      
      {/* Main Chart */}
      <div className="h-96 w-full bg-white">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={currentData}
            margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
            barCategoryGap="15%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="model" 
              angle={-35} 
              textAnchor="end" 
              height={100}
              tick={{ fontSize: 12 }}
              interval={0}
            />
            <YAxis domain={[0, 1]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{ paddingTop: 10 }}
              formatter={(value) => <span style={{ color: "#333", fontSize: 12 }}>{value}</span>}
            />
            
            {tasks.map((task, index) => (
              <Bar 
                key={task} 
                dataKey={task} 
                name={task}
                shape={renderCustomBar}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-6 text-center text-gray-700 bg-white">
        <p className="text-sm">
          <span className="inline-block w-4 h-4 bg-gray-400 border-2 border-black mx-1 align-middle"></span>
          Bold border indicates best performing model for each task
        </p>
        <p className="text-sm">
          <span className="inline-block w-4 h-4 bg-gray-400 border border-gray-500 border-dashed mx-1 align-middle"></span>
          Dashed border indicates second best performing model for each task
        </p>
        <p className="mt-2 text-xs">Values represent {selectedMetric} scores with 95% confidence intervals available on hover</p>
      </div>
    </div>
  );
};

export default ModelPerformanceBarCharts;
