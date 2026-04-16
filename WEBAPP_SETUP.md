# 🚀 Web Application Quickstart

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Core ML libraries (numpy, torch, sentence-transformers)
- Streamlit for web UI
- Medical datasets
- Logging support

### 2. Run the Web Application

```bash
streamlit run streamlit_app.py
```

The app will automatically open at: **http://localhost:8501**

### 3. Features

#### 📋 **Prescription Analysis Tab**
- Enter prescription text (or choose from examples)
- Real-time analysis with 5-step pipeline
- Interactive visualizations

#### 📖 **Examples Tab**
- Pre-built example prescriptions
- Common medical scenarios
- One-click analysis

#### 📊 **System Info Tab**
- View knowledge base statistics
- Check embedding configuration
- Browse drug-to-condition mappings

### 4. Logging

Logs are saved to `logs/cdss.log`

View logs while running:
```bash
tail -f logs/cdss.log
```

## Features

✅ **Interactive Web Interface**
- Clean, modern UI
- Real-time prescription analysis
- Visual feedback

✅ **Integrated Logging**
- File-based logging (`logs/cdss.log`)
- Console output
- Debug-level detail

✅ **Example Gallery**
- Pre-built test cases
- One-click execution
- Learn from examples

✅ **System Dashboard**
- Real-time metrics
- Knowledge base size
- Configuration control

## Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server port
- Upload limits
- Logging level

## Troubleshooting

### Streamlit not found
```bash
pip install streamlit --upgrade
```

### Port 8501 already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Slow startup
First run downloads the embedding model (~130 MB). Subsequent runs are cached.

## Example Workflows

### Workflow 1: Analyze a Patient Prescription
1. Go to "Prescription Analysis" tab
2. Enter: `ibuprofen 200mg and metformin 500mg`
3. Click "Analyze Prescription"
4. Review predictions and recommendations

### Workflow 2: Learn from Examples
1. Go to "Examples" tab
2. Select an example from dropdown
3. Click "Run Example Analysis"
4. Review the predicted conditions

### Workflow 3: Check System Status
1. Go to "System Info" tab
2. View knowledge base size
3. Browse drug mappings
4. Understand model configuration

## Production Deployment

### Local Network Access
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Docker (Optional)
```bash
docker run -p 8501:8501 -v $(pwd):/app streamlit/streamlit-demo
```

## What's Next?

Phase 3 improvements:
- [ ] Add FastAPI REST endpoint
- [ ] Implement persistent vector storage
- [ ] Add confidence calibration
- [ ] Export reports as PDF

---

**Status**: ✅ Web App Ready | **Version**: 1.1
