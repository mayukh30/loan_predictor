import { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Activity, AlertCircle, Info, Lightbulb } from 'lucide-react';
import './App.css'; 

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

function App() {
  const [stats, setStats] = useState(null);
  const [formData, setFormData] = useState({
    person_age: '',
    person_income: '',
    person_home_ownership: 'RENT',
    person_emp_length: '',
    loan_intent: 'EDUCATION',
    loan_amnt: '',
    loan_int_rate: '',
    loan_term_days: ''
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
    
    // Validate required fields are present and numeric fields are valid numbers
    const requiredFields = [
      'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
      'loan_intent', 'loan_amnt', 'loan_int_rate', 'loan_term_days'
    ];

    const missing = requiredFields.filter((key) => {
      const v = formData[key];
      if (v === null || v === undefined) return true;
      if (String(v).trim() === '') return true;
      if (['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_term_days'].includes(key)) {
        return Number.isNaN(Number(v));
      }
      return false;
    });

    if (missing.length > 0) {
      alert(`Please fill valid values for: ${missing.join(', ')}`);
      setLoading(false);
      return;
    }
    
    const payload = {
      person_age: parseInt(formData.person_age),
      person_income: parseFloat(formData.person_income),
      person_home_ownership: formData.person_home_ownership,
      person_emp_length: parseInt(formData.person_emp_length),
      loan_intent: formData.loan_intent,
      loan_amnt: parseFloat(formData.loan_amnt),
      loan_int_rate: parseFloat(formData.loan_int_rate),
      loan_term_days: parseInt(formData.loan_term_days)
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
    { name: 'Rent', value: stats.segmentation.Rent || 0 },
    { name: 'Own', value: stats.segmentation.Own || 0 },
    { name: 'Mortgage', value: stats.segmentation.Mortgage || 0 },
  ] : [];

  const COLORS = ['#8b5cf6', '#3b82f6', '#10b981'];

  return (
    <div className="app-container">
      <header className="header">
        <h1>IntelliLoan Prediction Engine</h1>
        <p>AI-Powered Risk Assessment & Customer Insights (Pinecone Vector DB)</p>
      </header>

      <div className="dashboard-grid">
        {/* Main Application Form */}
        <div className="card">
          <h2 className="card-title">New Loan Application</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Age</label>
                <input type="number" className="form-control" name="person_age" value={formData.person_age} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Income ($)</label>
                <input type="number" className="form-control" name="person_income" value={formData.person_income} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Employment Length (Years)</label>
                <input type="number" className="form-control" name="person_emp_length" value={formData.person_emp_length} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Home Ownership</label>
                <select className="form-control" name="person_home_ownership" value={formData.person_home_ownership} onChange={handleInputChange}>
                  <option value="RENT">Rent</option>
                  <option value="OWN">Own</option>
                  <option value="MORTGAGE">Mortgage</option>
                </select>
              </div>
              <div className="form-group">
                <label>Loan Intent</label>
                <select className="form-control" name="loan_intent" value={formData.loan_intent} onChange={handleInputChange}>
                  <option value="EDUCATION">Education</option>
                  <option value="MEDICAL">Medical</option>
                  <option value="VENTURE">Venture / Business</option>
                  <option value="PERSONAL">Personal</option>
                  <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
                  <option value="HOMEIMPROVEMENT">Home Improvement</option>
                </select>
              </div>
              <div className="form-group">
                <label>Loan Amount ($)</label>
                <input type="number" className="form-control" name="loan_amnt" value={formData.loan_amnt} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Loan Term (Days)</label>
                <select className="form-control" name="loan_term_days" value={formData.loan_term_days} onChange={handleInputChange} required>
                  <option value="" disabled>Select Term...</option>
                  <option value="360">1 Year (360 Days)</option>
                  <option value="720">2 Years (720 Days)</option>
                  <option value="1080">3 Years (1080 Days)</option>
                  <option value="1800">5 Years (1800 Days)</option>
                  <option value="3600">10 Years (3600 Days)</option>
                </select>
              </div>
              <div className="form-group" style={{gridColumn: '1 / -1'}}>
                <label>Interest Rate (%)</label>
                <input type="number" step="0.1" className="form-control" name="loan_int_rate" value={formData.loan_int_rate} onChange={handleInputChange} required />
              </div>
            </div>
            
            <button type="submit" className="btn btn-primary" disabled={loading}>
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

              {prediction.suggestions && prediction.suggestions.length > 0 && (
                <div className="card advice-card" style={{marginTop: '1.5rem'}}>
                  <h3 style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--warning)', marginBottom: '0.75rem'}}>
                    <Lightbulb size={20} />
                    Pinecone Vector DB Advice
                  </h3>
                  <p style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem'}}>
                    These recommendations were semantically retrieved from the Pinecone vector database using sentence-transformers based on your exact SHAP profile.
                  </p>
                  <div style={{marginTop: '1rem'}}>
                    <ul className="advice-summary-list">
                      {prediction.suggestions.map((item, idx) => (
                        <li key={idx} style={{marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem'}}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Stats Comparison */}
              {stats && stats.averages && (
                <div className="card" style={{marginTop: '1.5rem', border: '1px solid var(--border)', backgroundColor: 'rgba(0,0,0,0.1)'}}>
                  <h3 style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text)', marginBottom: '1rem'}}>
                    <Activity size={20} />
                    Comparison with Historical Data
                  </h3>
                  <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', height: '250px'}}>
                    <div style={{padding: '1rem', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', height: '100%', display: 'flex', flexDirection: 'column'}}>
                      <h4 style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textAlign: 'center'}}>Applicant Income ($)</h4>
                      <div style={{flexGrow: 1, minHeight: 0}}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={[
                            { name: 'You', value: Number(formData.person_income) || 0 },
                            { name: 'Approved', value: stats.averages.approved.Income },
                            { name: 'Rejected', value: stats.averages.rejected.Income }
                          ]} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <RechartsTooltip contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '4px', color: '#f8fafc'}} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                              <Cell fill="#8b5cf6" />
                              <Cell fill="#10b981" />
                              <Cell fill="#ef4444" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div style={{padding: '1rem', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', height: '100%', display: 'flex', flexDirection: 'column'}}>
                      <h4 style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textAlign: 'center'}}>Loan Amount ($)</h4>
                      <div style={{flexGrow: 1, minHeight: 0}}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={[
                            { name: 'You', value: Number(formData.loan_amnt) || 0 },
                            { name: 'Approved', value: stats.averages.approved.LoanAmount },
                            { name: 'Rejected', value: stats.averages.rejected.LoanAmount }
                          ]} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <RechartsTooltip contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '4px', color: '#f8fafc'}} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                              <Cell fill="#8b5cf6" />
                              <Cell fill="#10b981" />
                              <Cell fill="#ef4444" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
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
                This Loan Default Prediction System uses a Logistic Regression model integrated with SHAP (SHapley Additive exPlanations) to provide fair, mathematically sound reasoning for loan rejections.
                <br/><br/>
                It uses a <strong>Pinecone Vector Database</strong> and `sentence-transformers` to map SHAP explanations to semantically similar financial advice instantly!
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
