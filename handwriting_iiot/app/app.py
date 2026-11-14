"""Upload page: image -> preprocess -> CNN -> show result -> publish via MQTT. Open http://localhost:5000"""
import os, time, uuid, base64, io
from flask import Flask, request, render_template_string
from PIL import Image
from preprocess import preprocess_image
import predict as P
from mqtt_publish import publish_result

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UP = os.path.join(P.ROOT, "data", "uploads"); os.makedirs(UP, exist_ok=True)
try:
    P.get_model()
    print("Model loaded OK")
except Exception as e:
    print("Model load FAILED:", type(e).__name__, e)

HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IIoT Digit Recognition Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-main: #0f172a;
            --text-muted: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: var(--bg-gradient); color: var(--text-main); display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        
        .container { 
            background: var(--card-bg); 
            backdrop-filter: blur(20px);
            width: 100%; 
            max-width: 620px; 
            padding: 40px; 
            border-radius: 24px; 
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-flex { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        h2 { font-size: 26px; color: var(--text-main); font-weight: 800; letter-spacing: -0.5px; }
        
        .badge-live {
            background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px;
            display: flex; align-items: center; gap: 5px; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .badge-live::before { content: ''; width: 6px; height: 6px; background: #10b981; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

        p.subtitle { color: var(--text-muted); font-size: 14px; margin-bottom: 28px; line-height: 1.5; }

        .upload-card { 
            border: 2px dashed #cbd5e1; 
            padding: 28px; 
            text-align: center; 
            border-radius: 16px; 
            background: #f8fafc; 
            margin-bottom: 20px; 
            transition: all 0.3s ease; 
        }
        .upload-card:hover { border-color: var(--primary); background: #f0fdf4; transform: translateY(-2px); }
        
        input[type="file"] { font-size: 14px; color: #475569; margin-bottom: 10px; }
        
        button { 
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); 
            color: white; border: none; padding: 14px 24px; font-size: 16px; font-weight: 700; 
            border-radius: 12px; cursor: pointer; transition: all 0.3s ease; width: 100%;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }
        button:hover { opacity: 0.95; transform: translateY(-1px); box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6); }

        .error-msg { background: #fef2f2; color: #dc2626; padding: 14px; border-radius: 12px; margin-top: 18px; font-size: 14px; border: 1px solid #fca5a5; }

        .result-box { 
            margin-top: 30px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 18px; padding: 25px; text-align: center;
            animation: fadeIn 0.4s ease-in-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .digit-display { font-size: 80px; font-weight: 800; color: #1e1b4b; line-height: 1; margin: 15px 0 5px 0; }
        .confidence-pill { display: inline-block; font-size: 14px; font-weight: 700; padding: 6px 14px; border-radius: 30px; background: #e0e7ff; color: #4338ca; margin-bottom: 15px; }
        .low-conf-pill { background: #fef3c7 !important; color: #b45309 !important; }

        .status-bar { font-size: 13px; color: #475569; background: #fff; padding: 10px 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin: 15px 0; display: flex; justify-content: space-between; align-items: center; }

        .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px; font-size: 12px; color: #64748b; background: #ffffff; padding: 12px; border-radius: 12px; border: 1px solid #e2e8f0; }
        .meta-grid div b { display: block; color: var(--text-main); font-size: 13px; margin-top: 2px; }

        .preview-container { margin-top: 15px; display: flex; flex-direction: column; align-items: center; }
        .preview-container img { border-radius: 8px; border: 2px solid #cbd5e1; background: #000; padding: 4px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
<div class="container">
    <div class="header-flex">
        <h2>IIoT Digit Hub</h2>
        <div class="badge-live">System Active</div>
    </div>
    <p class="subtitle">Upload a clear handwritten digit image for high-speed CNN inference and local event logging.</p>
    
    <form method="post" enctype="multipart/form-data">
        <div class="upload-card">
            <input type="file" name="image" accept="image/*" required>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Supports PNG, JPG (Plain background recommended)</div>
        </div>
        <button type="submit">Run CNN Prediction</button>
    </form>

    {% if error %}
    <div class="error-msg"><b>Error:</b> {{error}}</div>
    {% endif %}

    {% if r %}
    <div class="result-box">
        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 700;">Recognized Output</div>
        <div class="digit-display">{{r.digit}}</div>
        
        <div class="confidence-pill {% if r.conf < 0.80 %}low-conf-pill{% endif %}">
            Confidence: {{'%.1f'|format(r.conf*100)}}%
        </div>
        
        {% if r.conf < 0.80 %}
        <p style="color: #b45309; font-size: 13px; margin-bottom: 10px; font-weight: 600;">⚠️ Low confidence score - manual review recommended.</p>
        {% endif %}

        <div class="status-bar">
            <span><b>Storage Status:</b></span>
            <span style="color: #059669; font-weight: 600;">{{'✓ Saved to local json' if r.ok else 'Failed ('+r.err+')'}}</span>
        </div>

        <div class="preview-container">
            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; text-transform: uppercase;">Model Normalized Input (28x28)</div>
            <img src="data:image/png;base64,{{r.img}}" width="64" style="image-rendering:pixelated;">
        </div>

        <div class="meta-grid">
            <div>Preprocess <b>{{r.t_pre}} ms</b></div>
            <div>Inference <b>{{r.t_inf}} ms</b></div>
            <div>Log Time <b>{{r.t_pub}} ms</b></div>
        </div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET": return render_template_string(HTML)
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template_string(HTML, error="No image selected. Please upload a PNG or JPG file.")
    path = os.path.join(UP, uuid.uuid4().hex + os.path.splitext(f.filename)[1].lower()); f.save(path)
    try:
        t = time.perf_counter(); arr, prev = preprocess_image(path); t1 = time.perf_counter()
        scores = P.get_model().predict(arr, verbose=0)[0]; t2 = time.perf_counter()
        digit = int(scores.argmax()); conf = float(scores[digit])
        _, ok, err = publish_result(digit, conf); t3 = time.perf_counter()
        buf = io.BytesIO(); Image.fromarray(prev).save(buf, "PNG")
        r = dict(digit=digit, conf=conf, ok=ok, err=err, img=base64.b64encode(buf.getvalue()).decode(),
                 t_pre=round((t1-t)*1000,1), t_inf=round((t2-t1)*1000,1), t_pub=round((t3-t2)*1000,1))
        return render_template_string(HTML, r=r)
        except Exception as e:
        import traceback; traceback.print_exc()
        return render_template_string(HTML, error=f"{type(e).__name__}: {e}")
    finally:
        try: os.remove(path)
        except OSError: pass

if __name__ == "__main__":
    app.run(port=5000)
