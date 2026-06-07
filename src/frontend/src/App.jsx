import { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from 'recharts';
import { Activity, Users, CheckCircle, AlertCircle, Info } from 'lucide-react';
import './App.css'; // Just keeping default import, but our styles are in index.css

const API_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [formData, setFormData] = useState({
    Gender: '',
    Married: '',
    Dependents: '',
    Education: '',
    Self_Employed: '',
    ApplicantIncome: '',
    CoapplicantIncome: '',
    LoanAmount: '',
    Loan_Amount_Term: '',
    Credit_History: '',
    Property_Area: '' // Will be mapped to one-hot before sending
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch stats on load
    axios.get(`${API_URL}/stats`).then(res => {
      setStats(res.data);
    }).catch(err => console.error("Error fetching stats:", err));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Validate all fields are filled
    const missing = Object.entries(formData).filter(([_, v]) => v === '' );
    if (missing.length > 0) {
      alert('Please fill all fields before submitting.');
      setLoading(false);
      return;
    }
    
    // Map Property_Area to the dummy variables
    const payload = {
      Gender: parseInt(formData.Gender),
      Married: parseInt(formData.Married),
      Dependents: parseInt(formData.Dependents),
      Education: parseInt(formData.Education),
      Self_Employed: parseInt(formData.Self_Employed),
      ApplicantIncome: parseFloat(formData.ApplicantIncome),
      CoapplicantIncome: parseFloat(formData.CoapplicantIncome),
      LoanAmount: parseFloat(formData.LoanAmount),
      Loan_Amount_Term: parseFloat(formData.Loan_Amount_Term),
      Credit_History: parseInt(formData.Credit_History),
      Property_Area_Semiurban: formData.Property_Area === 'Semiurban' ? 1 : 0,
      Property_Area_Urban: formData.Property_Area === 'Urban' ? 1 : 0
    };

    try {
      const res = await axios.post(`${API_URL}/predict`, payload);
      setPrediction(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Error making prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const pieData = stats ? [
    { name: 'Urban', value: stats.segmentation.Urban },
    { name: 'Semiurban', value: stats.segmentation.Semiurban },
    { name: 'Rural', value: stats.segmentation.Rural },
  ] : [];

  const COLORS = ['#8b5cf6', '#3b82f6', '#10b981'];

  return (
    <div className="app-container">
      <header className="header">
        <h1>IntelliLoan Prediction Engine</h1>
        <p>AI-Powered Risk Assessment & Customer Insights</p>
      </header>

      {/* Top Stats Row */}
      {stats && (
        <div className="stats-container">
          <div className="card stat-box">
            <Users size={32} color="#3b82f6" style={{margin: '0 auto'}}/>
            <div className="stat-value">{stats.total_applications}</div>
            <div className="stat-label">Total Applications</div>
          </div>
          <div className="card stat-box">
            <CheckCircle size={32} color="#10b981" style={{margin: '0 auto'}}/>
            <div className="stat-value">{(stats.approval_rate * 100).toFixed(1)}%</div>
            <div className="stat-label">Approval Rate</div>
          </div>
          <div className="card stat-box">
            <Activity size={32} color="#f59e0b" style={{margin: '0 auto'}}/>
            <div className="stat-value">{stats.average_risk_score.toFixed(1)}</div>
            <div className="stat-label">Avg Risk Score</div>
          </div>
        </div>
      )}

      <div className="dashboard-grid">
        {/* Main Application Form */}
        <div className="card">
          <h2 className="card-title">New Loan Application</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Gender</label>
                <select className="form-control" name="Gender" value={formData.Gender} onChange={handleInputChange}>
                  <option value="1">Male</option>
                  <option value="0">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Married</label>
                <select className="form-control" name="Married" value={formData.Married} onChange={handleInputChange}>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
              <div className="form-group">
                <label>Dependents</label>
                <select className="form-control" name="Dependents" value={formData.Dependents} onChange={handleInputChange}>
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3+</option>
                </select>
              </div>
              <div className="form-group">
                <label>Education</label>
                <select className="form-control" name="Education" value={formData.Education} onChange={handleInputChange}>
                  <option value="1">Graduate</option>
                  <option value="0">Not Graduate</option>
                </select>
              </div>
              <div className="form-group">
                <label>Self Employed</label>
                <select className="form-control" name="Self_Employed" value={formData.Self_Employed} onChange={handleInputChange}>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
              <div className="form-group">
                <label>Applicant Income ($)</label>
                <input type="number" className="form-control" name="ApplicantIncome" value={formData.ApplicantIncome} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Coapplicant Income ($)</label>
                <input type="number" className="form-control" name="CoapplicantIncome" value={formData.CoapplicantIncome} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Loan Amount (k$)</label>
                <input type="number" className="form-control" name="LoanAmount" value={formData.LoanAmount} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Loan Term (Days)</label>
                <input type="number" className="form-control" name="Loan_Amount_Term" value={formData.Loan_Amount_Term} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Credit History</label>
                <select className="form-control" name="Credit_History" value={formData.Credit_History} onChange={handleInputChange}>
                  <option value="1">Good (1.0)</option>
                  <option value="0">Bad (0.0)</option>
                </select>
              </div>
              <div className="form-group">
                <label>Property Area</label>
                <select className="form-control" name="Property_Area" value={formData.Property_Area} onChange={handleInputChange}>
                  <option value="Urban">Urban</option>
                  <option value="Semiurban">Semiurban</option>
                  <option value="Rural">Rural</option>
                </select>
              </div>
            </div>
            
            <button type="submit" className="btn btn-primary" disabled={loading || Object.values(formData).some(v => v === '')}>
              {loading ? 'Analyzing...' : 'Predict Loan Default Risk'}
            </button>
          </form>

          {/* Prediction Result */}
          {prediction && (
            <div className="result-card">
              <div className={`status-badge ${prediction.approved ? 'status-approved' : 'status-rejected'}`}>
                {prediction.approved ? 'APPROVED' : 'REJECTED'}
              </div>
              
              <div style={{display: 'flex', justifyContent: 'space-around', margin: '2rem 0'}}>
                <div>
                  <div style={{fontSize: '2.5rem', fontWeight: '700', color: prediction.risk_score > 50 ? 'var(--danger)' : 'var(--success)'}}>
                    {prediction.risk_score.toFixed(1)}
                  </div>
                  <div style={{color: 'var(--text-muted)', fontSize: '0.875rem'}}>Risk Score (0-100)</div>
                </div>
                <div>
                  <div style={{fontSize: '2.5rem', fontWeight: '700'}}>
                    {(prediction.probability * 100).toFixed(1)}%
                  </div>
                  <div style={{color: 'var(--text-muted)', fontSize: '0.875rem'}}>Repayment Prob.</div>
                </div>
              </div>

              {/* Explainability Section */}
              {!prediction.approved && prediction.rejection_reasons.length > 0 && (
                <div className="card" style={{marginTop: '1.5rem', border: '1px solid var(--danger)', backgroundColor: 'rgba(239, 68, 68, 0.05)'}}>
                  <h3 style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--danger)', marginBottom: '1rem'}}>
                    <AlertCircle size={20} />
                    Primary Rejection Reasons
                  </h3>
                  <p style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem'}}>
                    Based on our ML risk assessment, the following factors negatively impacted your application:
                  </p>
                  <div className="shap-container">
                    {prediction.rejection_reasons.map((reason, idx) => (
                      <div key={idx} className="shap-item" style={{display: 'flex', alignItems: 'flex-start', gap: '0.5rem', padding: '1rem 0'}}>
                        <div style={{color: 'var(--danger)', marginTop: '2px'}}>•</div>
                        <span className="shap-feature" style={{lineHeight: '1.4', fontWeight: '400'}}>{reason.explanation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div>
          <div className="card" style={{height: '100%'}}>
            <h2 className="card-title">Customer Segmentation</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip contentStyle={{backgroundColor: '#1e293b', border: '1px solid #334155'}} />
                  <Legend verticalAlign="bottom" height={36}/>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{marginTop: '2rem'}}>
              <h3 style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', fontSize: '1.1rem'}}>
                <Info size={18} />
                About System
              </h3>
              <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6'}}>
                This Loan Default Prediction System uses a Logistic Regression model integrated with SHAP (SHapley Additive exPlanations) to provide fair, mathematically sound reasoning for loan rejections, rather than relying on generative LLMs.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
