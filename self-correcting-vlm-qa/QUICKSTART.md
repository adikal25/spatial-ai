# Quick Start Guide - Simple Demo

Get started with **Claude-powered self-correcting spatial reasoning** in minutes!

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

Edit `config/.env` and add your Anthropic + Hugging Face keys:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token
ENABLE_TRIPO_RECONSTRUCTION=true  # disable if you don't need 3D meshes
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
3. Watch Claude analyze, verify, and self-correct!

## That's it!

The system will:
1. **Ask**: Claude generates initial answer with reasoning
2. **Reconstruct (optional)**: TripoSG lifts the image/VR frame into a textured 3D mesh
3. **Verify**: Depth maps + mesh stats check Claude's spatial claims
4. **Self-Correct**: Claude explicitly reflects on errors and corrects them
5. **Show**: See Claude's self-reflection, TripoSG preview, and proof overlays

✨ **Key Feature**: Claude uses an explicit self-reasoning loop to acknowledge mistakes and correct them honestly!

No complicated setup needed!
