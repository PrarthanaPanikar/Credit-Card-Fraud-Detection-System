'use client';

import { useState, useEffect } from 'react';
import { 
  Shield, AlertTriangle, CheckCircle, TrendingUp, 
  Clock, DollarSign, Activity, Bell 
} from 'lucide-react';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
interface Transaction {
  id: string;
  tx_id: string;
  amount: number;
  merchant_cat: string;
  city: string;
  device_type: string;
  channel: string;
  hour: number;
  is_international: boolean;
  is_night: boolean;
}

interface Prediction {
  transaction_id: string;
  fraud_probability: number;
  risk_score: number;
  decision: 'ALLOW' | 'REVIEW' | 'BLOCK';
  timestamp: string;
}

interface Stats {
  total_transactions: number;
  flagged_transactions: number;
  fraud_rate: number;
  avg_amount: number;
}

// Sample transactions for demo
const generateSampleTransaction = (): Transaction => ({
  id: Math.random().toString(36).substr(2, 9),
  tx_id: `TX${Math.floor(Math.random() * 1000000000)}`,
  amount: Math.floor(Math.random() * 50000) + 100,
  merchant_cat: ['grocery', 'fuel', 'retail', 'travel', 'food'][Math.floor(Math.random() * 5)],
  city: ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'][Math.floor(Math.random() * 5)],
  device_type: ['mobile', 'desktop', 'tablet', 'pos'][Math.floor(Math.random() * 4)],
  channel: ['online', 'in_store', 'app', 'phone'][Math.floor(Math.random() * 4)],
  hour: Math.floor(Math.random() * 24),
  is_international: Math.random() < 0.1,
  is_night: Math.random() < 0.2,
});

export default function FraudDashboard() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [stats, setStats] = useState<Stats>({
    total_transactions: 0,
    flagged_transactions: 0,
    fraud_rate: 0,
    avg_amount: 0,
  });
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected'>('disconnected');

  // Check API health
  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('disconnected');
      }
    } catch {
      setApiStatus('disconnected');
    }
  };

  // Simulate transaction stream
  const simulateTransaction = async () => {
    setLoading(true);
    
    const tx = generateSampleTransaction();
    
    // Create full transaction object for API
    const fullTx = {
      ...tx,
      merchant_id_hash: `MERCH${Math.floor(Math.random() * 1000)}`,
      card_id_hash: `CARD${Math.floor(Math.random() * 1000)}`,
      country: tx.is_international ? 'US' : 'IN',
      dayofweek: Math.floor(Math.random() * 7),
      prev_24h_tx_count_card: Math.floor(Math.random() * 10),
      prev_24h_amt_card: Math.floor(Math.random() * 50000),
      prev_1h_tx_count_card: Math.floor(Math.random() * 5),
      velocity_amt_1h: Math.floor(Math.random() * 20000),
    };

    try {
      const response = await fetch(`${API_BASE_URL}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fullTx),
      });

      if (response.ok) {
        const prediction: Prediction = await response.json();
        
        setTransactions(prev => [tx, ...prev].slice(0, 50));
        setPredictions(prev => [prediction, ...prev].slice(0, 50));
        
        // Update stats
        setStats(prev => ({
          total_transactions: prev.total_transactions + 1,
          flagged_transactions: prev.flagged_transactions + (prediction.decision === 'REVIEW' ? 1 : 0),
          fraud_rate: ((prev.flagged_transactions + (prediction.decision === 'REVIEW' ? 1 : 0)) / (prev.total_transactions + 1)) * 100,
          avg_amount: (prev.avg_amount * prev.total_transactions + tx.amount) / (prev.total_transactions + 1),
        }));
      }
    } catch (error) {
      console.error('API Error:', error);
    }
    
    setLoading(false);
  };

  // Auto-simulate
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        simulateTransaction();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [loading]);

  const getRiskColor = (score: number) => {
    if (score >= 80) return 'bg-red-500';
    if (score >= 50) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getRiskTextColor = (score: number) => {
    if (score >= 80) return 'text-red-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">Fraud Detection Dashboard</h1>
                <p className="text-sm text-gray-500">Real-time transaction monitoring</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 ${apiStatus === 'connected' ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-2 h-2 rounded-full ${apiStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm font-medium">{apiStatus === 'connected' ? 'API Connected' : 'API Disconnected'}</span>
              </div>
              <button
                onClick={simulateTransaction}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Processing...' : 'Simulate Transaction'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Total Transactions</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_transactions.toLocaleString()}</p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Flagged for Review</p>
                <p className="text-2xl font-bold text-red-600">{stats.flagged_transactions.toLocaleString()}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Fraud Rate</p>
                <p className="text-2xl font-bold text-gray-900">{stats.fraud_rate.toFixed(2)}%</p>
              </div>
              <TrendingUp className="h-8 w-8 text-yellow-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Avg Transaction</p>
                <p className="text-2xl font-bold text-gray-900">₹{stats.avg_amount.toFixed(0)}</p>
              </div>
              <DollarSign className="h-8 w-8 text-green-500" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Threshold Control */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-sm p-6 border">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Risk Threshold</h2>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-500">Decision Threshold</label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer mt-2"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Lenient (0.1)</span>
                    <span className="font-semibold text-blue-600">{threshold.toFixed(2)}</span>
                    <span>Strict (0.9)</span>
                  </div>
                </div>
                
                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium text-gray-900 mb-2">Risk Levels</h3>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full bg-green-500" />
                      <span className="text-sm text-gray-600">Low Risk (0-49)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full bg-yellow-500" />
                      <span className="text-sm text-gray-600">Medium Risk (50-79)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <span className="text-sm text-gray-600">High Risk (80-100)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Alerts */}
            <div className="bg-white rounded-xl shadow-sm p-6 border mt-6">
              <div className="flex items-center space-x-2 mb-4">
                <Bell className="h-5 w-5 text-red-500" />
                <h2 className="text-lg font-semibold text-gray-900">Recent Alerts</h2>
              </div>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {predictions.filter(p => p.risk_score >= 70).slice(0, 5).map((pred, idx) => (
                  <div key={idx} className="flex items-center space-x-3 p-3 bg-red-50 rounded-lg">
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{pred.transaction_id}</p>
                      <p className="text-xs text-gray-500">Risk Score: {pred.risk_score}</p>
                    </div>
                  </div>
                ))}
                {predictions.filter(p => p.risk_score >= 70).length === 0 && (
                  <p className="text-sm text-gray-500 text-center py-4">No recent alerts</p>
                )}
              </div>
            </div>
          </div>

          {/* Transaction Table */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
              <div className="px-6 py-4 border-b">
                <h2 className="text-lg font-semibold text-gray-900">Recent Transactions</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Merchant</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Score</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Decision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {predictions.map((pred, idx) => {
                      const tx = transactions[idx];
                      if (!tx) return null;
                      
                      return (
                        <tr key={pred.transaction_id} className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-sm font-medium text-gray-900">
                            {tx.tx_id}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-600">
                            ₹{tx.amount.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-600 capitalize">
                            {tx.merchant_cat}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center space-x-2">
                              <div className={`w-2 h-2 rounded-full ${getRiskColor(pred.risk_score)}`} />
                              <span className={`text-sm font-semibold ${getRiskTextColor(pred.risk_score)}`}>
                                {pred.risk_score}
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              pred.decision === 'REVIEW' 
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {pred.decision === 'REVIEW' ? (
                                <AlertTriangle className="w-3 h-3 mr-1" />
                              ) : (
                                <CheckCircle className="w-3 h-3 mr-1" />
                              )}
                              {pred.decision}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {predictions.length === 0 && (
                  <div className="text-center py-12">
                    <Clock className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No transactions yet. Click "Simulate Transaction" to start.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
