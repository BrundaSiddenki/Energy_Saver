# SmartEnergySense: AI-Powered Household Energy Optimization

🧠 **Project Goal**
An intelligent web platform that uses AI to analyze, understand, and converse with users about their energy usage — without complex data pipelines or cloud infrastructure.

## 🚀 Features

### 🤖 Conversational Energy Assistant
- Ask questions like "Why was my energy usage so high on Monday?"
- Get AI-powered insights about your consumption patterns
- Natural language explanations of energy trends

### 📊 Universal CSV Support
- Upload any CSV file with energy data (automatic format detection)
- Supports various column names (energy, power, consumption, etc.)
- Automatic data standardization and validation

### 🎯 Device-Specific Energy Analysis
- Analyze energy consumption by different appliances/devices
- Air Conditioner, Refrigerator, Washing Machine, Dishwasher, etc.
- Device-specific savings recommendations
- Potential energy savings calculations

### 🧩 Pattern Clustering (Energy Personas)
- Automatic grouping of days into energy personas:
  - Saver Days
  - Heavy Load Days  
  - Idle Days
- Interactive exploration of your energy patterns

### 📈 Visual Explanations + AI Insights
- Simple language explanations of AI models
- "Your spike on Thursday matched patterns from last month"
- SHAP values for model explainability

## 🛠️ Tech Stack

| Area | Tool |
|------|------|
| Language | Python |
| ML/AI | Scikit-learn, Prophet, OpenAI GPT |
| UI | Streamlit |
| Data Storage | Local CSV or SQLite |
| Visualization | Plotly, Matplotlib |
| Deployment | Streamlit Cloud / Render / Localhost |

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SmartEnergySense
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
SmartEnergySense/
├── app.py                 # Main Streamlit application
├── data/
│   ├── sample_data.csv    # Sample energy usage data
│   └── energy_data.db     # SQLite database
├── src/
│   ├── __init__.py
│   ├── data_generator.py  # Sample data generation
│   ├── forecasting.py     # Prophet/ARIMA forecasting
│   ├── clustering.py      # Energy persona clustering
│   ├── ai_assistant.py    # OpenAI GPT integration
│   └── visualizations.py  # Plotly/Matplotlib charts
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
├── requirements.txt
└── README.md
```

## 🗓️ Development Timeline

- **Week 1** – Setup & Data Simulation ✅
- **Week 2** – Forecasting & Visualization
- **Week 3** – AI Chat Assistant
- **Week 4** – Energy Personas & Recommendations
- **Week 5** – Polishing & Deployment

## 📄 Copyright

@Branch (Diff with Main Branch) @mrecw

## 🎯 Usage

1. **Upload Data**: Upload your energy usage CSV file
2. **Explore Insights**: View AI-generated insights and recommendations
3. **Ask Questions**: Use the conversational assistant to get specific insights
4. **Forecast**: Get predictions for future energy usage
5. **Optimize**: Receive personalized recommendations for energy savings

## 🤝 Contributing

This is a learning project focused on AI-powered energy optimization. Feel free to contribute improvements!

## 📄 License

MIT License - feel free to use this project for learning and development.

---

**Copyright:** @Branch (Diff with Main Branch) @mrecw 