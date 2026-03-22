# Welcome to InstructFX2FX/PWFX

### Installation

```bash
conda create -n pwfx python=3.10
conda activate pwfx
pip install -r requirements.txt
```

Optional: add your Openrouter API key

```bash
cd src
touch env
echo 'OPENROUTER_API_KEY=your_api_key' >> env
cd ..
```