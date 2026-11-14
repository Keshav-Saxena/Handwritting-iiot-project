"""MQTT subscriber + web dashboard. Run, then open http://localhost:5001"""
import json, os
from collections import deque
import paho.mqtt.client as mqtt
from flask import Flask, jsonify

BROKER = os.getenv("MQTT_BROKER", "localhost"); PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "lab/digit/result")
history = deque(maxlen=10); state = {"connected": False}

def on_connect(c, u, f, rc, *a):
    state["connected"] = True; c.subscribe(TOPIC, qos=1)
def on_disconnect(c, *a): state["connected"] = False
def on_message(c, u, msg):
    try: history.appendleft(json.loads(msg.payload.decode()))
    except Exception: pass

def start_mqtt():
    try: c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError: c = mqtt.Client()
    c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
    c.connect_async(BROKER, PORT, 60); c.loop_start()

app = Flask(__name__)
PAGE = """<!doctype html><title>Digit Dashboard</title><body style="font-family:sans-serif;max-width:560px;margin:30px auto">
<h2>IIoT Digit Recognition Dashboard</h2><div id=s></div><div id=d style="font-size:110px;font-weight:700;margin:10px 0">-</div>
<div id=c></div><h4>Recent</h4><table border=1 cellpadding=6 style="border-collapse:collapse" id=h></table>
<script>async function u(){const r=await (await fetch('/data')).json();s.textContent='Broker: '+(r.connected?'connected':'DISCONNECTED');
const l=r.history[0];if(l){d.textContent=l.predicted_digit;c.textContent='Confidence '+(l.confidence*100).toFixed(1)+'% | '+l.timestamp+' | '+l.device_id}
h.innerHTML='<tr><th>Time</th><th>Digit</th><th>Conf.</th></tr>'+r.history.map(x=>`<tr><td>${x.timestamp}</td><td>${x.predicted_digit}</td><td>${(x.confidence*100).toFixed(1)}%</td></tr>`).join('')}
u();setInterval(u,1000)</script>"""
@app.route("/")
def index(): return PAGE
@app.route("/data")
def data(): return jsonify(connected=state["connected"], history=list(history))

if __name__ == "__main__":
    start_mqtt(); app.run(port=5001)
