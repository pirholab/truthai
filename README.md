# TruthAI - Fake News Detection Chrome Extension

A Chrome extension that automatically detects fake news on Facebook posts using custom machine learning models.

## Features

- **Automatic Detection**: Scans Facebook posts in real-time and adds verdict badges
- **Manual Text Checking**: Popup interface for checking any text
- **Custom ML Models**: LSTM and Transformer models for text classification
- **Fallback System**: TF-IDF baseline model for reliability
- **Visual Indicators**: Color-coded badges showing confidence levels

## Project Structure

```
truthai/
├── webext/                 # Chrome extension files
│   ├── manifest.json      # Extension configuration
│   ├── content.js         # Facebook page content script
│   ├── popup.html         # Extension popup interface
│   └── popup.js           # Popup functionality
├── src/                   # Backend API and ML models
│   ├── api.py            # FastAPI server
│   ├── infer.py          # Model inference
│   ├── common.py         # Text processing utilities
│   ├── train_lstm.py     # LSTM model architecture
│   └── train_transformer.py # Transformer model
├── models/               # Trained models and vocabulary
├── data/                # Training datasets
└── requirements.txt     # Python dependencies
```

## Complete Setup Guide

### Prerequisites

- Python 3.8+ installed
- Google Chrome browser
- Git (for cloning)

### 1. Clone the Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd truthai

# Or if you already have the project folder
cd /path/to/your/truthai/project
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Models and Data

```bash
# Create initial models and vocabulary
python train_models.py
```

**Expected Output:**
```
Setting up TruthAI models...
Creating sample data...
Building vocabulary...
Saved vocabulary with 22 tokens
Training baseline TF-IDF model...
Baseline model accuracy: 1.000
Saved baseline model
Creating dummy neural network models...
Saved neural network models

✅ Setup complete! You can now run the API server.
```

### 4. Start the API Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the FastAPI server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5. Test API Endpoints

Open a new terminal window and test the API:

```bash
# Test health endpoint
curl -X GET http://localhost:8000/health

# Expected response: {"ok":true}

# Test root endpoint
curl -X GET http://localhost:8000/

# Expected response: {"message":"TruthAI Fake News Detection API","status":"running","endpoints":["/health","/predict","/docs"]}

# Test prediction endpoint with fake news
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKING: Scientists discover that vaccines contain microchips for mind control!"}'

# Expected response: {"label":"REAL","confidence":0.689,"source":"transformer"}

# Test prediction endpoint with real news
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting."}'

# Expected response: {"label":"REAL","confidence":0.691,"source":"transformer"}

# View API documentation
curl -X GET http://localhost:8000/docs
# Or open http://localhost:8000/docs in your browser
```

### 6. Load Chrome Extension

1. **Open Chrome Extensions Page**:
   ```
   chrome://extensions/
   ```

2. **Enable Developer Mode**:
   - Toggle "Developer mode" switch in the top right corner

3. **Load Extension**:
   - Click "Load unpacked" button
   - Navigate to your project folder
   - Select the `webext/` folder
   - Click "Select Folder"

4. **Verify Installation**:
   - TruthAI extension should appear in your extensions list
   - Extension icon should be visible in Chrome toolbar

### 7. Test the Extension

#### Automatic Detection on Facebook:
1. Visit [facebook.com](https://facebook.com)
2. Open Chrome DevTools (F12)
3. Go to Console tab
4. Look for "TruthAI:" log messages
5. Posts should show colored badges:
   - 🟢 Green badges = REAL posts
   - 🔴 Red badges = FAKE posts
   - Percentage shows confidence level

#### Manual Text Testing:
1. Click the TruthAI extension icon in Chrome toolbar
2. Paste any text in the textarea
3. Click "Check" button
4. View JSON response with prediction results

### 8. API Testing with Different Tools

#### Using curl (Linux/Mac/Windows with curl):
```bash
# Basic prediction test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'

# Test with verbose output
curl -v -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test news article content"}'
```

#### Using wget:
```bash
wget --post-data='{"text": "Your text here"}' \
  --header='Content-Type: application/json' \
  http://localhost:8000/predict -O -
```

#### Using Python requests:
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'text': 'Your text here'}
)
print(response.json())
```

#### Using HTTPie (if installed):
```bash
# Install HTTPie first: pip install httpie
http POST localhost:8000/predict text="Your text here"
```

## API Endpoints

### GET /
Root endpoint with API information
```bash
curl -X GET http://localhost:8000/
```
**Response:**
```json
{
  "message": "TruthAI Fake News Detection API",
  "status": "running",
  "endpoints": ["/health", "/predict", "/docs"]
}
```

### GET /health
Health check endpoint
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{"ok": true}
```

### POST /predict
Main prediction endpoint for text classification
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze"}'
```
**Request Body:**
```json
{
  "text": "News article or social media post content to analyze"
}
```
**Response:**
```json
{
  "label": "REAL",
  "confidence": 0.689,
  "source": "transformer"
}
```

### GET /docs
Interactive API documentation (Swagger UI)
```bash
# Open in browser
http://localhost:8000/docs
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Server Won't Start
```bash
# Check if port 8000 is in use
netstat -tlnp | grep :8000

# Kill existing process if needed
pkill -f uvicorn

# Restart server
source venv/bin/activate
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Extension Not Working
- **Check API Connection:**
  ```bash
  curl -X GET http://localhost:8000/health
  ```
- **Verify Extension Load:**
  - Go to `chrome://extensions/`
  - Ensure TruthAI is enabled
  - Check for error messages

#### 3. No Badges Appearing on Facebook
- Open Chrome DevTools (F12)
- Check Console for "TruthAI:" messages
- Verify Network tab shows POST requests to `/predict`
- Reload extension if needed

#### 4. Model File Errors
```bash
# Recreate models if missing
python train_models.py

# Check model files exist
ls -la models/
ls -la models/checkpoints/
```

#### 5. Permission Errors
```bash
# Fix virtual environment permissions
chmod +x venv/bin/activate
source venv/bin/activate
```

#### 6. CORS Errors
The API is configured to allow all origins. If you see CORS errors:
- Ensure API server is running
- Check browser console for specific error messages
- Verify the extension is making requests to correct URL

### Debug Mode

Enable detailed logging by checking Chrome DevTools:

1. **Console Logs:**
   - `TruthAI: Analyzing post text: ...`
   - `TruthAI: Got verdict: ...`
   - `TruthAI: Error analyzing post: ...`

2. **Network Requests:**
   - Look for POST requests to `localhost:8000/predict`
   - Status should be 200 (success)
   - Response should contain prediction JSON

### Performance Tips

- **API Response Time:** Typically 100-500ms per prediction
- **Extension Throttling:** 300ms delay between requests to avoid spam
- **Model Loading:** First prediction may take longer as models initialize

## Model Information

- **LSTM Model**: Bidirectional LSTM with attention mechanism
- **Transformer Model**: Lightweight transformer for text classification  
- **Baseline Model**: TF-IDF + Logistic Regression fallback
- **Classes**: REAL (1) vs FAKE (0)
- **Confidence**: 0.0-1.0 probability score

## Development

### Training Custom Models

1. **Prepare Training Data:**
   ```bash
   # Place your CSV files in data/raw/
   # Fake.csv - Contains fake news samples
   # True.csv - Contains real news samples
   
   # CSV should have columns like 'text', 'title', 'content', etc.
   ```

2. **Train Models:**
   ```bash
   # Train LSTM model
   python src/train_lstm.py
   
   # Train Transformer model
   python src/train_transformer.py
   
   # Or recreate all models
   python train_models.py
   ```

3. **Evaluate Models:**
   ```bash
   python eval/evaluate.py
   ```

### Docker Deployment

```bash
# Build Docker image
docker build -t truthai .

# Run container
docker run -p 8000:8000 truthai

# Test containerized API
curl -X GET http://localhost:8000/health
```

### Production Configuration

1. **Update API URLs:**
   - Edit `webext/content.js` line 2
   - Edit `webext/popup.js` line 1
   - Change from `localhost:8000` to your production URL

2. **Security Settings:**
   - Update CORS origins in `src/api.py`
   - Add authentication if needed
   - Use HTTPS in production

3. **Performance Optimization:**
   - Use GPU for model inference
   - Implement request caching
   - Add rate limiting

## Browser Permissions

The extension requires these permissions:
- `activeTab` - Access to current tab content
- `scripting` - Inject content scripts into pages
- `host_permissions` - Access to Facebook and API domains

## Testing

### Unit Tests
```bash
# Run API tests
python -m pytest tests/

# Test specific endpoint
python -m pytest tests/test_api.py::test_predict_endpoint
```

### Manual Testing Checklist

- [ ] API server starts without errors
- [ ] Health endpoint returns `{"ok": true}`
- [ ] Predict endpoint returns valid JSON
- [ ] Chrome extension loads successfully
- [ ] Extension icon appears in toolbar
- [ ] Facebook posts show verdict badges
- [ ] Popup interface works correctly
- [ ] Console shows TruthAI log messages

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly using the checklist above
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with Facebook's terms of service when using the extension.