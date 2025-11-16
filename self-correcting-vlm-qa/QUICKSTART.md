# Quick Start Guide - Simple Demo

Get started with **Anthropic Claude 3.5 Sonnet powered self-correcting spatial reasoning** in minutes!

## 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 2. Set Up API Key

Create a `.env` file in the `config` folder:

```bash
# Copy the example
cp config/.env.example config/.env
```

Edit `config/.env` and add your Anthropic API key:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 3. Run the Demo

**Option A: Run API + Streamlit Demo**

Terminal 1 - Start API:
```bash
python -m uvicorn src.api.main:app --reload
```

Terminal 2 - Start Demo UI:
```bash
streamlit run demo/app.py
```

Then open http://localhost:8501 in your browser!

**Option B: Use Python Script**

```bash
python example_usage.py
```

(Update the `IMAGE_PATH` in the script first)

## 4. Try It Out

1. Upload an image in the Streamlit demo
2. Ask a spatial question like:
   - "Which object is closer to the camera?"
   - "How many cars are in the image?"
   - "Is the person taller than the building?"
3. Watch Claude 3.5 Sonnet analyze, verify, and self-correct!

## That's it!

The system will:
1. **Ask**: Claude 3.5 Sonnet generates the initial answer with reasoning
2. **Verify**: Depth maps check the model's spatial claims
3. **Self-Correct**: Claude 3.5 Sonnet explicitly reflects on errors and corrects them
4. **Show**: See Claude 3.5 Sonnet's self-reflection and proof overlays

✨ **Key Feature**: Claude 3.5 Sonnet uses an explicit self-reasoning loop to acknowledge mistakes and correct them honestly!

No complicated setup needed!
