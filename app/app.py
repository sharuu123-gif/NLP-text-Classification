from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

print('Loading model...')
MODEL_PATH = 'saved_models/bert_model'
tokenizer  = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model      = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print('Model loaded!')

LABELS = {0:'World', 1:'Sports', 2:'Business', 3:'Technology'}
COLORS = {'World':'#4C72B0','Sports':'#DD8452','Business':'#55A868','Technology':'#C44E52'}

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>NLP Text Classifier</title>
    <style>
        * { box-sizing:border-box; margin:0; padding:0; }
        body { font-family:Segoe UI,sans-serif; background:#f0f2f5; display:flex; justify-content:center; padding:40px 16px; }
        .container { background:white; border-radius:12px; padding:36px; max-width:680px; width:100%; box-shadow:0 4px 20px rgba(0,0,0,0.08); }
        h1 { font-size:1.8rem; color:#1a1a2e; margin-bottom:6px; }
        p  { color:#666; margin-bottom:24px; }
        textarea { width:100%; border:2px solid #e0e0e0; border-radius:8px; padding:14px; font-size:0.95rem; resize:vertical; }
        button { margin-top:14px; width:100%; padding:14px; background:#4C72B0; color:white; border:none; border-radius:8px; font-size:1rem; cursor:pointer; }
        button:hover { background:#3a5a9a; }
        .result { margin-top:28px; padding:20px; border-radius:8px; background:#f8f9fa; display:none; }
        .badge  { display:inline-block; padding:8px 20px; border-radius:20px; color:white; font-size:1.1rem; font-weight:600; margin-bottom:12px; }
        .conf   { color:#333; margin-bottom:16px; font-weight:500; }
        .label-row { display:flex; justify-content:space-between; margin-top:10px; font-size:0.9rem; color:#555; }
        .bar-wrap  { background:#e0e0e0; border-radius:4px; height:10px; margin-top:4px; }
        .bar       { height:10px; border-radius:4px; transition:width 0.5s; }
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 NLP Text Classifier</h1>
    <p>Classify any web text into: World · Sports · Business · Technology</p>
    <textarea id="txt" rows="6" placeholder="Paste any news article here...&#10;&#10;Example: Apple launches new MacBook with M3 chip at developer conference"></textarea>
    <button onclick="classify()">🔍 Classify Text</button>
    <div class="result" id="result">
        <div class="badge" id="badge"></div>
        <div class="conf"  id="conf"></div>
        <div id="probs"></div>
    </div>
</div>
<script>
const COLORS = {World:"#4C72B0",Sports:"#DD8452",Business:"#55A868",Technology:"#C44E52"};
async function classify() {
    const text = document.getElementById("txt").value.trim();
    if (!text) { alert("Please enter some text!"); return; }
    const btn = document.querySelector("button");
    btn.textContent = "⏳ Classifying...";
    btn.disabled = true;
    const res  = await fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({text})
    });
    const data = await res.json();
    document.getElementById("badge").textContent      = "📌 " + data.category;
    document.getElementById("badge").style.background = COLORS[data.category];
    document.getElementById("conf").textContent       = "Confidence: " + (data.confidence*100).toFixed(1) + "%";
    let html = "";
    for (const [label, prob] of Object.entries(data.all_probs)) {
        const pct = (prob*100).toFixed(1);
        html += `<div class="label-row"><span>${label}</span><span>${pct}%</span></div>
                 <div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${COLORS[label]}"></div></div>`;
    }
    document.getElementById("probs").innerHTML = html;
    document.getElementById("result").style.display = "block";
    btn.textContent = "🔍 Classify Text";
    btn.disabled = false;
}
</script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    text   = request.get_json().get('text','').strip()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=1)[0].tolist()
    pred_id = int(torch.argmax(logits, dim=1))
    return jsonify({
        'category':   LABELS[pred_id],
        'confidence': round(max(probs), 4),
        'all_probs':  {LABELS[i]: round(p,4) for i,p in enumerate(probs)}
    })

if __name__ == '__main__':
    print('='*50)
    print('Starting Flask App...')
    print('Open browser → http://localhost:5000')
    print('='*50)
    app.run(debug=False, port=5000)