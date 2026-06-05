"""
Kidney Stone Detection — Flask Web App
Supports: Browser Webcam, RTSP Stream, Video File Upload
Run:  python3 app.py
Open: http://localhost:5001
"""

import threading
import time
import os
import tempfile
import cv2
import numpy as np
import supervision as sv
from flask import Flask, Response, jsonify, request, render_template_string
from inference import InferencePipeline, get_model
from werkzeug.utils import secure_filename

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY  = "JOJ9YhKu3J1HA30qV7ol"
MODEL_ID = "kidney-stone-orz8j/3"

# ─── Load model for per-frame inference (webcam mode) ─────────────────────────
print("Loading model...")
infer_model = get_model(model_id=MODEL_ID, api_key=API_KEY)
print("Model ready.")

# ─── App state ────────────────────────────────────────────────────────────────
state = {
    "confidence":   0.5,
    "class_filter": [],
    "running":      False,
    "frame_count":  0,
    "latest_jpeg":  None,
    "lock":         threading.Lock(),
    "pipeline":     None,
    "video_source": None,
    "source_type":  None,   # "webcam" | "rtsp" | "file"
    "source_error": None,
}

# ─── Supervision annotators ───────────────────────────────────────────────────
box_annotator   = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW, thickness=3)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_matplotlib("viridis", 5),
    text_scale=0.55,
)
byte_tracker      = sv.ByteTrack()
webcam_tracker    = sv.ByteTrack()   # separate tracker for webcam mode

# ─── Shared annotate helper ───────────────────────────────────────────────────
def annotate_frame(frame_bgr, detections, tracker):
    detections = tracker.update_with_detections(detections)
    cf   = state["class_filter"]
    keep = (
        [i for i, name in enumerate(detections.data.get("class_name", []))
         if name in cf]
        if cf else list(range(len(detections)))
    )
    if keep:
        filtered = sv.Detections(
            xyxy=detections.xyxy[keep],
            confidence=detections.confidence[keep],
            class_id=detections.class_id[keep],
            data={k: v[keep] for k, v in detections.data.items()},
        )
    else:
        filtered = sv.Detections.empty()

    labels = [
        f"{name} {conf:.2f}"
        for name, conf in zip(
            filtered.data.get("class_name", []),
            filtered.confidence if filtered.confidence is not None else [],
        )
    ]
    annotated = box_annotator.annotate(frame_bgr.copy(), filtered)
    annotated = label_annotator.annotate(annotated, filtered, labels)
    return annotated

# ─── Pipeline callback (file / RTSP modes) ────────────────────────────────────
def on_prediction(predictions, video_frame):
    if not state["running"]:
        return
    detections  = sv.Detections.from_inference(predictions)
    frame_bgr   = cv2.cvtColor(np.array(video_frame.image), cv2.COLOR_RGB2BGR)
    annotated   = annotate_frame(frame_bgr, detections, byte_tracker)
    annotated   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with state["lock"]:
        state["latest_jpeg"] = buf.tobytes()
        state["frame_count"] += 1

# ─── Pipeline start / stop ────────────────────────────────────────────────────
def start_pipeline():
    try:
        pipeline = InferencePipeline.init(
            model_id=MODEL_ID,
            video_reference=state["video_source"],
            on_prediction=on_prediction,
            api_key=API_KEY,
            confidence=state["confidence"],
        )
        state["pipeline"]     = pipeline
        state["source_error"] = None
        pipeline.start()
        pipeline.join()
    except Exception as e:
        state["source_error"] = str(e)
    finally:
        state["running"] = False

def stop_pipeline():
    state["running"] = False
    p = state.get("pipeline")
    if p:
        try:
            p.terminate()
        except Exception:
            pass
    state["pipeline"] = None

# ─── MJPEG stream (file / RTSP modes) ────────────────────────────────────────
def generate_stream():
    while True:
        with state["lock"]:
            jpeg = state["latest_jpeg"]
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        else:
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            msg = state["source_error"] or "Select a source and press Start"
            cv2.putText(placeholder, msg, (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)
            _, buf = cv2.imencode(".jpg", placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1 / 30)

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kidney Stone Detection — Tyndall</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #080b10;
    --surface: #0e1218;
    --surface2:#151c25;
    --border:  rgba(255,255,255,0.07);
    --accent:  #00d4ff;
    --green:   #22c55e;
    --red:     #ef4444;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --radius:  12px;
  }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* ── Topbar ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px; height: 60px;
    background: var(--surface); border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .topbar-left { display: flex; align-items: center; gap: 16px; }
  .logo-wrap {
    display: flex; align-items: center;
    background: #fff; border-radius: 7px; padding: 4px 10px; height: 38px;
  }
  .logo-wrap img { height: 26px; width: auto; display: block; }
  .divider-v { width: 1px; height: 26px; background: var(--border); }
  .app-title h1 { font-size: 0.92rem; font-weight: 600; color: var(--text); }
  .app-title span { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .status-pill {
    display: flex; align-items: center; gap: 7px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 99px; padding: 5px 14px;
    font-size: 0.76rem; color: var(--muted);
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #333; flex-shrink: 0; }
  .status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* ── Layout ── */
  .main {
    display: grid; grid-template-columns: 1fr 300px;
    flex: 1; min-height: 0; height: calc(100vh - 60px); overflow: hidden;
  }

  /* ── Feed ── */
  .feed-wrap {
    position: relative; background: #000;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; border-right: 1px solid var(--border); min-height: 0;
  }
  /* Server stream (file/RTSP) */
  #feed-server { max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:none; }
  /* Browser webcam canvas */
  #feed-canvas { max-width:100%; max-height:100%; display:none; object-fit:contain; }
  .feed-overlay { position:absolute; bottom:14px; left:14px; display:flex; gap:7px; }
  .chip {
    background: rgba(0,0,0,0.55); backdrop-filter:blur(8px);
    border: 1px solid rgba(255,255,255,0.1); border-radius:6px;
    padding: 3px 9px; font-size:0.7rem; color:#ccc; font-weight:500;
  }
  .chip.accent { color:var(--accent); border-color:rgba(0,212,255,0.3); }

  /* ── Sidebar ── */
  .sidebar { display:flex; flex-direction:column; overflow-y:auto; background:var(--surface); }
  .section { padding:18px 18px; border-bottom:1px solid var(--border); }
  .section-label { font-size:0.63rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); margin-bottom:12px; }

  /* Source tabs */
  .src-tabs { display:flex; gap:6px; margin-bottom:12px; }
  .src-tab {
    flex:1; padding:8px 4px; border:1px solid var(--border); border-radius:8px;
    background:transparent; color:var(--muted); cursor:pointer; text-align:center;
    font-size:0.73rem; font-family:'Inter',sans-serif; font-weight:500;
    display:flex; flex-direction:column; align-items:center; gap:3px; line-height:1;
    transition:all 0.15s;
  }
  .src-tab .icon { font-size:1rem; }
  .src-tab:hover { border-color:rgba(255,255,255,0.2); color:var(--text); }
  .src-tab.active { border-color:var(--accent); color:var(--accent); background:rgba(0,212,255,0.07); }
  .source-status { font-size:0.75rem; color:var(--muted); min-height:16px; }
  .source-status.ready { color:var(--green); }
  .source-status.error { color:var(--red); }
  #upload-bar-wrap { display:none; height:3px; background:var(--surface2); border-radius:2px; overflow:hidden; margin-top:8px; }
  #upload-bar { height:100%; width:0%; background:var(--accent); transition:width 0.2s; }

  /* Buttons */
  .btn-row { display:flex; gap:8px; margin-bottom:2px; }
  .btn {
    flex:1; padding:10px; border:none; border-radius:9px;
    font-size:0.83rem; font-family:'Inter',sans-serif; font-weight:600;
    cursor:pointer; transition:all 0.15s;
  }
  .btn:hover { filter:brightness(1.1); transform:translateY(-1px); }
  .btn:active { transform:translateY(0); }
  .btn-start { background:var(--green); color:#000; }
  .btn-stop  { background:var(--surface2); color:var(--red); border:1px solid rgba(239,68,68,0.3); }

  /* Slider */
  .slider-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .slider-label  { font-size:0.78rem; color:var(--muted); font-weight:500; }
  .slider-val    { font-size:0.78rem; font-weight:600; color:var(--accent); background:rgba(0,212,255,0.1); border-radius:4px; padding:2px 8px; }
  input[type=range] { width:100%; accent-color:var(--accent); cursor:pointer; }

  /* Text field */
  .field-label { font-size:0.75rem; color:var(--muted); font-weight:500; margin-bottom:6px; display:block; }
  .field-input {
    width:100%; background:var(--surface2); border:1px solid var(--border);
    color:var(--text); border-radius:8px; padding:8px 12px;
    font-size:0.8rem; font-family:'Inter',sans-serif; outline:none; transition:border-color 0.15s;
  }
  .field-input:focus { border-color:var(--accent); }
  .field-input::placeholder { color:#334155; }

  /* Stats */
  .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .stat-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:12px; }
  .stat-card .val { font-size:1.3rem; font-weight:700; color:var(--text); line-height:1; margin-bottom:3px; }
  .stat-card .lbl { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }

  /* Modal */
  .modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,0.8); backdrop-filter:blur(4px);
    z-index:200; align-items:center; justify-content:center;
  }
  .modal-overlay.open { display:flex; }
  .modal {
    background:var(--surface2); border:1px solid var(--border); border-radius:16px;
    padding:26px; width:440px; max-width:90vw;
    display:flex; flex-direction:column; gap:14px;
    box-shadow:0 24px 60px rgba(0,0,0,0.6);
  }
  .modal-title { font-size:0.95rem; font-weight:600; }
  .modal-sub   { font-size:0.78rem; color:var(--muted); }
  .modal-btns  { display:flex; gap:10px; justify-content:flex-end; margin-top:4px; }
  .btn-confirm { background:var(--accent); color:#000; padding:8px 20px; border:none; border-radius:8px; cursor:pointer; font-weight:600; font-family:'Inter',sans-serif; font-size:0.82rem; }
  .btn-cancel  { background:transparent; color:var(--muted); padding:8px 20px; border:1px solid var(--border); border-radius:8px; cursor:pointer; font-family:'Inter',sans-serif; font-size:0.82rem; }
  .modal-error { color:var(--red); font-size:0.76rem; display:none; }
</style>
</head>
<body>

<!-- Topbar -->
<div class="topbar">
  <div class="topbar-left">
    <div class="logo-wrap">
      <img src="/logo" alt="Tyndall National Institute">
    </div>
    <div class="divider-v"></div>
    <div class="app-title">
      <h1>Kidney Stone Detection (Yolo11n)</h1>
      <span>Rev0 &nbsp;·&nbsp; Tyndall National Institute</span>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Idle</span>
    &nbsp;·&nbsp; <span id="frame-count">0</span> frames
  </div>
</div>

<!-- Main -->
<div class="main">

  <!-- Feed -->
  <div class="feed-wrap">
    <!-- Server MJPEG stream (file / RTSP) -->
    <img id="feed-server" src="/video_feed" alt="feed">
    <!-- Browser webcam canvas -->
    <canvas id="feed-canvas"></canvas>
    <div class="feed-overlay">
      <div class="chip accent" id="chip-model">kidney-stone-orz8j/3</div>
      <div class="chip" id="chip-source">—</div>
      <div class="chip" id="chip-conf">Conf: 0.50</div>
    </div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <div class="section">
      <div class="section-label">Input Source</div>
      <div class="src-tabs">
        <div class="src-tab" onclick="selectSource('webcam')">
          <span class="icon">📷</span>Webcam
        </div>
        <div class="src-tab" onclick="selectSource('rtsp')">
          <span class="icon">📡</span>RTSP
        </div>
        <div class="src-tab" onclick="selectSource('file')">
          <span class="icon">🎬</span>File
        </div>
      </div>
      <div id="upload-bar-wrap"><div id="upload-bar"></div></div>
      <div class="source-status" id="source-status">Choose a source above</div>
    </div>

    <div class="section">
      <div class="section-label">Detection</div>
      <div class="btn-row">
        <button class="btn btn-start" onclick="startDetection()">▶ Start</button>
        <button class="btn btn-stop"  onclick="stopDetection()">■ Stop</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Confidence Threshold</div>
      <div class="slider-header">
        <span class="slider-label">Min confidence</span>
        <span class="slider-val" id="conf-val">0.50</span>
      </div>
      <input type="range" id="conf" min="0.1" max="1.0" step="0.05" value="0.5"
             oninput="updateConfidence(this.value)">
    </div>

    <div class="section">
      <div class="section-label">Class Filter</div>
      <label class="field-label" for="classes">Blank = detect all classes</label>
      <input class="field-input" type="text" id="classes"
             placeholder="e.g. kidney_stone"
             onchange="updateClasses(this.value)">
    </div>

    <div class="section">
      <div class="section-label">Live Stats</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="val" id="stat-frames">0</div>
          <div class="lbl">Frames</div>
        </div>
        <div class="stat-card">
          <div class="val" id="stat-status">Idle</div>
          <div class="lbl">Status</div>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- RTSP modal -->
<div class="modal-overlay" id="rtsp-modal">
  <div class="modal">
    <div class="modal-title">📡 RTSP Stream URL</div>
    <div class="modal-sub">e.g. rtsp://admin:password@192.168.1.100:554/stream</div>
    <input class="field-input" type="text" id="rtsp-input" placeholder="rtsp://..." autocomplete="off">
    <div class="modal-error" id="rtsp-error"></div>
    <div class="modal-btns">
      <button class="btn-cancel"  onclick="closeModal('rtsp')">Cancel</button>
      <button class="btn-confirm" onclick="confirmRtsp()">Connect</button>
    </div>
  </div>
</div>

<input type="file" id="file-picker" accept="video/*" style="display:none" onchange="handleFileChosen(this)">

<!-- Hidden video element for webcam capture -->
<video id="webcam-video" autoplay playsinline muted style="display:none"></video>

<script>
  /* ── State ── */
  let currentSource = null;
  let sourceReady   = false;
  let pollTimer     = null;
  let webcamStream  = null;
  let webcamRunning = false;
  let webcamTimer   = null;

  const feedServer = document.getElementById('feed-server');
  const feedCanvas = document.getElementById('feed-canvas');
  const webcamVid  = document.getElementById('webcam-video');

  /* ── MJPEG reconnect ── */
  feedServer.onerror = () => {
    setTimeout(() => { feedServer.src = '/video_feed?' + Date.now(); }, 1000);
  };

  /* ── Show correct feed element ── */
  function showFeed(mode) {
    feedServer.style.display = mode === 'server' ? 'block' : 'none';
    feedCanvas.style.display = mode === 'canvas' ? 'block' : 'none';
  }

  /* ── Source selection ── */
  function selectSource(type) {
    stopDetection();
    currentSource = type;
    sourceReady   = false;

    document.querySelectorAll('.src-tab').forEach((t, i) => {
      t.classList.toggle('active', ['webcam','rtsp','file'][i] === type);
    });

    if (type === 'webcam') {
      setStatus('Requesting camera access…', '');
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          webcamStream = stream;
          webcamVid.srcObject = stream;
          webcamVid.onloadedmetadata = () => {
            feedCanvas.width  = webcamVid.videoWidth;
            feedCanvas.height = webcamVid.videoHeight;
          };
          sourceReady = true;
          setStatus('✔ Webcam ready', 'ready');
          setChip('chip-source', '📷 Webcam');
          showFeed('canvas');
        })
        .catch(() => setStatus('⚠ Camera access denied', 'error'));

    } else if (type === 'rtsp') {
      openModal('rtsp');

    } else if (type === 'file') {
      sourceReady = false;
      setStatus('Opening file picker…', '');
      document.getElementById('file-picker').value = '';
      document.getElementById('file-picker').click();
    }
  }

  /* ── Status / chip helpers ── */
  function setStatus(msg, cls) {
    const el = document.getElementById('source-status');
    el.textContent = msg; el.className = 'source-status ' + (cls||'');
  }
  function setChip(id, txt) { const el=document.getElementById(id); if(el) el.textContent=txt; }

  /* ── RTSP modal ── */
  function openModal(name) {
    document.getElementById(name+'-modal').classList.add('open');
    setTimeout(() => document.getElementById(name+'-input').focus(), 60);
  }
  function closeModal(name) {
    document.getElementById(name+'-modal').classList.remove('open');
    document.querySelectorAll('.src-tab').forEach((t,i) => {
      t.classList.toggle('active', ['webcam','rtsp','file'][i] === currentSource);
    });
  }
  function confirmRtsp() {
    const url = document.getElementById('rtsp-input').value.trim();
    const err = document.getElementById('rtsp-error');
    if (!url) { err.textContent='Enter a URL.'; err.style.display='block'; return; }
    err.style.display = 'none';
    fetch('/set_source', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ type:'rtsp', value: url }) })
    .then(r=>r.json()).then(d => {
      if (d.ok) {
        sourceReady=true; currentSource='rtsp';
        setStatus('✔ '+url, 'ready');
        setChip('chip-source','📡 RTSP');
        showFeed('server');
        closeModal('rtsp');
      } else { err.textContent=d.error; err.style.display='block'; }
    });
  }
  document.addEventListener('keydown', e => {
    if (e.key==='Enter' && document.getElementById('rtsp-modal').classList.contains('open')) confirmRtsp();
    if (e.key==='Escape') closeModal('rtsp');
  });

  /* ── File upload ── */
  function handleFileChosen(input) {
    const file = input.files[0];
    if (!file) { setStatus('No file selected.','error'); return; }
    setStatus('Uploading '+file.name+'…','');
    document.getElementById('upload-bar-wrap').style.display='block';
    document.getElementById('upload-bar').style.width='0%';
    const form=new FormData(); form.append('video',file);
    const xhr=new XMLHttpRequest();
    xhr.open('POST','/upload');
    xhr.upload.onprogress=e=>{
      if(e.lengthComputable)
        document.getElementById('upload-bar').style.width=(e.loaded/e.total*100)+'%';
    };
    xhr.onload=()=>{
      document.getElementById('upload-bar-wrap').style.display='none';
      const d=JSON.parse(xhr.responseText);
      if(d.ok){
        sourceReady=true;
        setStatus('✔ '+file.name,'ready');
        setChip('chip-source','🎬 '+file.name);
        showFeed('server');
      } else { sourceReady=false; setStatus('⚠ '+d.error,'error'); }
    };
    xhr.onerror=()=>setStatus('⚠ Upload failed.','error');
    xhr.send(form);
  }

  /* ── Webcam inference loop ── */
  async function webcamLoop() {
    if (!webcamRunning) return;
    const ctx = feedCanvas.getContext('2d');
    feedCanvas.width  = webcamVid.videoWidth  || feedCanvas.width;
    feedCanvas.height = webcamVid.videoHeight || feedCanvas.height;
    ctx.drawImage(webcamVid, 0, 0);

    feedCanvas.toBlob(async blob => {
      if (!webcamRunning) return;
      try {
        const form = new FormData();
        form.append('frame', blob, 'frame.jpg');
        const resp = await fetch('/infer_frame', { method:'POST', body:form });
        if (!resp.ok) throw new Error();
        const imgBlob = await resp.blob();
        const url = URL.createObjectURL(imgBlob);
        const img = new Image();
        img.onload = () => {
          feedCanvas.width  = img.naturalWidth;
          feedCanvas.height = img.naturalHeight;
          feedCanvas.getContext('2d').drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          updateStats();
          if (webcamRunning) webcamLoop();
        };
        img.src = url;
      } catch(e) {
        if (webcamRunning) setTimeout(webcamLoop, 200);
      }
    }, 'image/jpeg', 0.8);
  }

  /* ── Start / Stop ── */
  function startDetection() {
    if (!sourceReady) { setStatus('⚠ Configure a source first.','error'); return; }

    if (currentSource === 'webcam') {
      webcamRunning = true;
      setLive(true);
      webcamLoop();
    } else {
      fetch('/start', { method:'POST' })
        .then(r=>r.json())
        .then(d=>{ if(d.ok) startPolling(); else setStatus('⚠ '+d.error,'error'); });
    }
  }

  function stopDetection() {
    webcamRunning = false;
    fetch('/stop', { method:'POST' });
    setLive(false);
    clearInterval(pollTimer);
  }

  function setLive(on) {
    document.getElementById('status-dot').className = 'status-dot'+(on?' live':'');
    document.getElementById('status-text').textContent = on?'Live':'Idle';
    document.getElementById('stat-status').textContent = on?'Live':'Idle';
  }

  function updateStats() {
    fetch('/status').then(r=>r.json()).then(d=>{
      document.getElementById('frame-count').textContent = d.frame_count;
      document.getElementById('stat-frames').textContent = d.frame_count;
    });
  }

  /* ── Poll for server-side modes ── */
  function startPolling() {
    setLive(true);
    clearInterval(pollTimer);
    pollTimer = setInterval(()=>{
      fetch('/status').then(r=>r.json()).then(d=>{
        document.getElementById('frame-count').textContent = d.frame_count;
        document.getElementById('stat-frames').textContent = d.frame_count;
        if(d.error) setStatus('⚠ '+d.error,'error');
        if(!d.running){ setLive(false); clearInterval(pollTimer); }
      });
    }, 500);
  }

  function updateConfidence(val) {
    const v=parseFloat(val).toFixed(2);
    document.getElementById('conf-val').textContent=v;
    setChip('chip-conf','Conf: '+v);
    fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({confidence:parseFloat(val)})});
  }
  function updateClasses(val) {
    fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({class_filter:val.split(',').map(s=>s.trim()).filter(Boolean)})});
  }
</script>
</body>
</html>
"""

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/logo")
def logo():
    for name in ["tyndall_logo.png", "tyndall_logo.jpg", "tyndall_logo.svg"]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(path):
            ext  = name.rsplit(".", 1)[-1]
            mime = {"png":"image/png","jpg":"image/jpeg","svg":"image/svg+xml"}.get(ext,"image/png")
            return Response(open(path, "rb").read(), mimetype=mime)
    return "", 404

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/infer_frame", methods=["POST"])
def infer_frame():
    """Per-frame inference for browser webcam mode."""
    f = request.files.get("frame")
    if f is None:
        return "", 400
    img_array = np.frombuffer(f.read(), np.uint8)
    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return "", 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = infer_model.infer(frame_rgb, confidence=state["confidence"])[0]
    detections = sv.Detections.from_inference(results)
    annotated  = annotate_frame(frame_bgr, detections, webcam_tracker)
    annotated  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with state["lock"]:
        state["frame_count"] += 1
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/set_source", methods=["POST"])
def set_source():
    data      = request.get_json(force=True)
    src_type  = data.get("type")
    src_value = data.get("value")
    if src_type == "webcam":
        state["video_source"] = int(src_value) if str(src_value).isdigit() else 0
    elif src_type == "rtsp":
        if not src_value:
            return jsonify(ok=False, error="URL cannot be empty.")
        state["video_source"] = src_value
    elif src_type == "file":
        if not os.path.exists(str(src_value)):
            return jsonify(ok=False, error=f"File not found: {src_value}")
        state["video_source"] = src_value
    else:
        return jsonify(ok=False, error="Unknown source type.")
    state["source_type"]  = src_type
    state["source_error"] = None
    return jsonify(ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify(ok=False, error="No file received.")
    f = request.files["video"]
    if f.filename == "":
        return jsonify(ok=False, error="Empty filename.")
    ext = os.path.splitext(secure_filename(f.filename))[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    state["video_source"] = tmp.name
    state["source_type"]  = "file"
    state["source_error"] = None
    return jsonify(ok=True)

@app.route("/start", methods=["POST"])
def start():
    if state["running"]:
        return jsonify(ok=True)
    if state["video_source"] is None:
        return jsonify(ok=False, error="No source selected.")
    byte_tracker.__init__()
    state["frame_count"] = 0
    state["latest_jpeg"] = None
    state["running"]     = True
    threading.Thread(target=start_pipeline, daemon=True).start()
    return jsonify(ok=True)

@app.route("/stop", methods=["POST"])
def stop():
    stop_pipeline()
    return jsonify(ok=True)

@app.route("/status")
def status():
    return jsonify(
        running=state["running"],
        frame_count=state["frame_count"],
        error=state.get("source_error"),
    )

@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json(force=True)
    if "confidence"   in data: state["confidence"]   = float(data["confidence"])
    if "class_filter" in data: state["class_filter"] = data["class_filter"]
    return jsonify(ok=True)

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("="*50)
    print(f" Kidney Stone Detection Server — port {port}")
    print("="*50)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
