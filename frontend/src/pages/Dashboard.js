import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { Activity, TrendingUp, AlertTriangle, Shield, Zap, Database } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function Dashboard() {
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [flaggedTransactions, setFlaggedTransactions] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [formData, setFormData] = useState({
    amount: '',
    time: ''
  });

  useEffect(() => {
    checkHealth();
    loadAnalytics();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setHealthStatus(response.data);
      
      if (!response.data.model_loaded) {
        toast.warning('Model not trained yet', {
          description: 'Run the training script: python -m app.ml.train'
        });
      }
    } catch (error) {
      console.error('Health check failed:', error);
      toast.error('API connection failed');
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await axios.get(`${API}/analytics/risk-distribution`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredicting(true);

    try {
      const payload = {
        amount: parseFloat(formData.amount),
        time: parseFloat(formData.time),
        // Add default values for V1-V28 (PCA components)
        ...Array.from({ length: 28 }, (_, i) => ({ [`v${i + 1}`]: 0.0 })).reduce((acc, curr) => ({ ...acc, ...curr }), {})
      };

      const response = await axios.post(`${API}/predict`, payload);
      const prediction = response.data;

      // Show result toast
      if (prediction.risk_label === 'HIGH_RISK') {
        toast.error('High Risk Transaction Detected!', {
          description: `Risk Score: ${prediction.risk_score} | Fraud Probability: ${(prediction.fraud_probability * 100).toFixed(2)}%`
        });
      } else if (prediction.risk_label === 'MEDIUM_RISK') {
        toast.warning('Medium Risk Transaction', {
          description: `Risk Score: ${prediction.risk_score} | Fraud Probability: ${(prediction.fraud_probability * 100).toFixed(2)}%`
        });
      } else {
        toast.success('Low Risk Transaction', {
          description: `Risk Score: ${prediction.risk_score} | Fraud Probability: ${(prediction.fraud_probability * 100).toFixed(2)}%`
        });
      }

      // Add to flagged transactions if medium or high risk
      if (prediction.risk_label !== 'LOW_RISK') {
        setFlaggedTransactions(prev => [
          {
            ...prediction,
            amount: formData.amount,
            time: formData.time,
            id: Date.now()
          },
          ...prev.slice(0, 19) // Keep last 20
        ]);
      }

      // Reload analytics
      loadAnalytics();

      // Reset form
      setFormData({ amount: '', time: '' });

    } catch (error) {
      console.error('Prediction failed:', error);
      const errorMsg = error.response?.data?.detail || 'Prediction failed. Please check if the model is trained.';
      toast.error(errorMsg);
    } finally {
      setPredicting(false);
    }
  };

  const getRiskBadgeClass = (label) => {
    switch (label) {
      case 'HIGH_RISK':
        return 'risk-badge high';
      case 'MEDIUM_RISK':
        return 'risk-badge medium';
      case 'LOW_RISK':
        return 'risk-badge low';
      default:
        return 'risk-badge low';
    }
  };

  const pieChartData = analytics ? [
    { name: 'High Risk', value: analytics.high_risk_count, color: '#ef4444' },
    { name: 'Medium Risk', value: analytics.medium_risk_count, color: '#fbbf24' },
    { name: 'Low Risk', value: analytics.low_risk_count, color: '#10b981' }
  ] : [];

  const riskTrendData = analytics?.recent_predictions?.slice(0, 10).reverse().map((p, i) => ({
    name: `T${i + 1}`,
    score: p.risk_score
  })) || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f1419] via-[#1a2332] to-[#0f1419] text-gray-100">
      {/* Header */}
      <header className="glass-effect border-b border-gray-700/50 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Shield className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gradient">SecureScore AI</h1>
                <p className="text-sm text-gray-400">Real-time Fraud Detection & Risk Scoring</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {healthStatus && (
                <div className="flex items-center gap-2 px-4 py-2 glass-effect rounded-lg">
                  <div className={`pulse-dot ${healthStatus.model_loaded ? 'bg-primary' : 'bg-yellow-500'}`} />
                  <span className="text-sm font-medium">
                    {healthStatus.model_loaded ? 'Model Ready' : 'Model Not Loaded'}
                  </span>
                </div>
              )}
              <Button
                data-testid="refresh-btn"
                variant="outline"
                size="sm"
                onClick={() => {
                  checkHealth();
                  loadAnalytics();
                }}
                className="border-primary/30 hover:bg-primary/10"
              >
                <Activity className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="dashboard-grid mb-8">
          <Card className="stat-card bg-card/50 border-gray-700/50" data-testid="total-predictions-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
                <Database className="w-4 h-4" />
                Total Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-primary">{analytics?.total_predictions || 0}</p>
              <p className="text-xs text-gray-500 mt-1">Analyzed transactions</p>
            </CardContent>
          </Card>

          <Card className="stat-card bg-card/50 border-gray-700/50" data-testid="high-risk-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                High Risk
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-red-500">{analytics?.high_risk_count || 0}</p>
              <p className="text-xs text-gray-500 mt-1">Flagged as high risk</p>
            </CardContent>
          </Card>

          <Card className="stat-card bg-card/50 border-gray-700/50" data-testid="avg-risk-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Average Risk
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-yellow-500">{analytics?.average_risk_score || 0}</p>
              <p className="text-xs text-gray-500 mt-1">Mean risk score</p>
            </CardContent>
          </Card>

          <Card className="stat-card bg-card/50 border-gray-700/50" data-testid="model-status-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Model Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-bold text-primary">
                {healthStatus?.model_info?.model_type || 'Not Loaded'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {healthStatus?.model_info?.version || 'N/A'}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Transaction Input Form */}
          <Card className="bg-card/50 border-gray-700/50 lg:col-span-1" data-testid="transaction-input-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                Analyze Transaction
              </CardTitle>
              <CardDescription>Enter transaction details for risk assessment</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handlePredict} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="amount">Amount ($)</Label>
                  <Input
                    id="amount"
                    name="amount"
                    type="number"
                    step="0.01"
                    min="0"
                    placeholder="150.50"
                    value={formData.amount}
                    onChange={handleInputChange}
                    required
                    data-testid="amount-input"
                    className="input-field bg-secondary border-gray-600 focus:border-primary"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="time">Time (seconds since first transaction)</Label>
                  <Input
                    id="time"
                    name="time"
                    type="number"
                    step="1"
                    min="0"
                    placeholder="12345"
                    value={formData.time}
                    onChange={handleInputChange}
                    required
                    data-testid="time-input"
                    className="input-field bg-secondary border-gray-600 focus:border-primary"
                  />
                </div>

                <Button
                  type="submit"
                  disabled={predicting || !healthStatus?.model_loaded}
                  className="w-full btn-primary bg-primary hover:bg-primary/90 text-black font-semibold"
                  data-testid="predict-button"
                >
                  {predicting ? (
                    <>
                      <div className="loading-spinner mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield className="w-4 h-4 mr-2" />
                      Check Risk Score
                    </>
                  )}
                </Button>
              </form>

              {!healthStatus?.model_loaded && (
                <div className="mt-4 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <p className="text-sm text-yellow-500">
                    <AlertTriangle className="w-4 h-4 inline mr-2" />
                    Model not trained. Run: <code className="text-xs bg-black/30 px-2 py-1 rounded">python -m app.ml.train</code>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Charts */}
          <Card className="bg-card/50 border-gray-700/50 lg:col-span-2" data-testid="charts-card">
            <CardHeader>
              <CardTitle>Risk Analytics</CardTitle>
              <CardDescription>Distribution and trends</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Pie Chart */}
                <div className="chart-container" data-testid="risk-distribution-chart">
                  <h3 className="text-sm font-semibold text-gray-400 mb-4">Risk Distribution</h3>
                  {pieChartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={pieChartData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {pieChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ backgroundColor: '#1a2332', border: '1px solid #374151' }} />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-[250px] text-gray-500">
                      No data available
                    </div>
                  )}
                </div>

                {/* Line Chart */}
                <div className="chart-container" data-testid="risk-trend-chart">
                  <h3 className="text-sm font-semibold text-gray-400 mb-4">Recent Risk Scores</h3>
                  {riskTrendData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={riskTrendData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#9ca3af" />
                        <YAxis stroke="#9ca3af" />
                        <Tooltip contentStyle={{ backgroundColor: '#1a2332', border: '1px solid #374151' }} />
                        <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-[250px] text-gray-500">
                      No data available
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Flagged Transactions Table */}
        {flaggedTransactions.length > 0 && (
          <Card className="mt-6 bg-card/50 border-gray-700/50" data-testid="flagged-transactions-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Flagged Transactions
              </CardTitle>
              <CardDescription>Medium and high risk transactions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="transaction-table" data-testid="flagged-transactions-table">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">Time</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">Amount</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">Risk Score</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">Probability</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {flaggedTransactions.map((transaction, index) => (
                      <tr key={transaction.id || index} className="border-b border-gray-700/50 hover:bg-gray-800/30 transition-colors" data-testid={`transaction-row-${index}`}>
                        <td className="px-4 py-3 text-sm text-gray-300">{transaction.time}s</td>
                        <td className="px-4 py-3 text-sm font-medium text-gray-200">${parseFloat(transaction.amount).toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm">
                          <span className="font-bold text-primary">{transaction.risk_score}</span>
                          <span className="text-gray-500">/100</span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">{(transaction.fraud_probability * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3">
                          <span className={getRiskBadgeClass(transaction.risk_label)} data-testid={`risk-badge-${index}`}>
                            {transaction.risk_label === 'HIGH_RISK' && <AlertTriangle className="w-3 h-3" />}
                            {transaction.risk_label.replace('_', ' ')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-700/50 mt-12">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <p>© 2025 SecureScore AI. Fraud Detection & Risk Scoring System.</p>
            <p>Powered by XGBoost & FastAPI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Dashboard;
