"use strict";

const $ = (id) => document.getElementById(id);
const RECORD_BYTES = 3073;
const NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"];
const COLORS = {loss: "#66e7d2", accuracy: "#a69bff", ema: "#f8ca81"};
let worker = null;
let history = [];
let predictions = new Map();
let testBytes = null;
let meta = null;
let currentConfig = null;
let checkpoint = null;
let runStart = 0;
let timer = null;

function status(label, kind) {
    $("phase").textContent = label;
    $("dot").className = "dot " + (kind || "");
}
function log(line) {
    const pre = $("logs");
    if (pre.textContent.length > 26000) {
        pre.textContent = pre.textContent.slice(-20000);
    }
    pre.textContent += String(line) + "\n";
    pre.scrollTop = pre.scrollHeight;
}
function percent(v) {
    return (100 * v).toFixed(1) + "%";
}
function format(v, places) {
    return Number.isFinite(v) ? v.toFixed(places) : "—";
}
function config() {
    return {
        train: Number($("train-count").value),
        eval: Number($("eval-count").value),
        width: Number($("width").value),
        batch: Number($("batch").value),
        epochs: Number($("epochs").value)
    };
}
function setActive(running) {
    for (const id of ["train-count", "eval-count", "width", "batch", "epochs"]) {
        $(id).disabled = running;
    }
    $("start").disabled = running || !testBytes;
    $("stop").disabled = !running;
    $("download").disabled = running || !checkpoint;
}
function progress(value, text) {
    const v = Math.max(0, Math.min(100, value));
    $("progressfill").style.width = v + "%";
    $("progress").setAttribute("aria-valuenow", String(v));
    $("progress-text").textContent = text;
}
function setupCanvas(canvas, cssHeight) {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(120, canvas.getBoundingClientRect().width);
    const height = cssHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {g: g, w: width, h: height};
}
function drawHistory() {
    const c = setupCanvas($("history"), 240);
    const {g, w, h} = c;
    const left = 42, top = 18, right = w - 38, bottom = h - 32;
    g.clearRect(0, 0, w, h);
    g.font = "11px system-ui";
    g.textAlign = "right";
    const maxLoss = Math.max(1, ...history.map((item) => item.loss * 1.25));
    for (let i = 0; i <= 4; i++) {
        const y = top + i * (bottom - top) / 4;
        g.strokeStyle = "#2a3850";
        g.beginPath();g.moveTo(left, y);g.lineTo(right, y);g.stroke();
        g.fillStyle = "#8494b0";
        g.fillText((maxLoss * (1 - i / 4)).toFixed(2), left - 8, y + 4);
        g.textAlign = "left";
        g.fillText(Math.round((1 - i / 4) * 100) + "%", right + 6, y + 4);
        g.textAlign = "right";
    }
    g.textAlign = "center";
    g.fillStyle = "#8ea0ba";
    if (!history.length) {
        g.font = "13px system-ui";
        g.fillText("Start training to draw actual epoch measurements", w / 2, h / 2);
        return;
    }
    const n = currentConfig ? currentConfig.epochs : history.length;
    for (let epoch = 1; epoch <= n; epoch++) {
        const x = left + (epoch - .5) * (right - left) / n;
        g.fillText(String(epoch), x, h - 13);
    }
    const draw = (key, color, maximum) => {
        g.strokeStyle = color;g.fillStyle = color;g.lineWidth = 2.5;
        g.beginPath();
        history.forEach((point, i) => {
            const x = left + (point.epoch - .5) * (right - left) / n;
            const y = bottom - Math.max(0, Math.min(1, point[key] / maximum)) * (bottom - top);
            if (!i) {g.moveTo(x, y);} else {g.lineTo(x, y);}
        });
        g.stroke();
        history.forEach((point) => {
            const x = left + (point.epoch - .5) * (right - left) / n;
            const y = bottom - Math.max(0, Math.min(1, point[key] / maximum)) * (bottom - top);
            g.beginPath();g.arc(x, y, 3.5, 0, Math.PI * 2);g.fill();
        });
    };
    draw("loss", COLORS.loss, maxLoss);
    draw("accuracy", COLORS.accuracy, 1);
    draw("ema", COLORS.ema, 1);
}
function drawImage(canvas, bytes, index) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(32, 32);
    const start = index * RECORD_BYTES + 1;
    for (let p = 0; p < 1024; p++) {
        const at = p * 4;
        img.data[at] = bytes[start + p];
        img.data[at + 1] = bytes[start + 1024 + p];
        img.data[at + 2] = bytes[start + 2048 + p];
        img.data[at + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}
function drawSamples() {
    const root = $("samples");
    root.replaceChildren();
    if (!testBytes) return;
    for (let i = 0; i < 12; i++) {
        const label = testBytes[i * RECORD_BYTES];
        const card = document.createElement("div");
        card.className = "sample";
        const canvas = document.createElement("canvas");
        canvas.width = 32;canvas.height = 32;
        canvas.setAttribute("aria-label", "CIFAR-10 sample " + i + ", true label " + NAMES[label]);
        drawImage(canvas, testBytes, i);
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = "#" + (i + 1) + " " + NAMES[label];
        const pred = document.createElement("div");
        pred.className = "pred";
        if (predictions.has(i)) {
            const classId = predictions.get(i);
            pred.textContent = "→ " + NAMES[classId];
            if (classId !== label) pred.classList.add("wrong");
        } else {
            pred.textContent = "not classified";
        }
        card.append(canvas, name, pred);
        root.append(card);
    }
}
function confusionMatrix() {
    const matrix = Array.from({length: 10}, () => Array(10).fill(0));
    if (testBytes && currentConfig) {
        for (const [i, predicted] of predictions) {
            if (i >= 0 && i < currentConfig.eval &&
                predicted >= 0 && predicted < 10) {
                matrix[testBytes[i * RECORD_BYTES]][predicted]++;
            }
        }
    }
    return matrix;
}
function drawConfusion() {
    const canvas = $("confusion");
    const c = setupCanvas(canvas, Math.max(280, Math.min(390, canvas.getBoundingClientRect().width)));
    const {g, w, h} = c;
    g.clearRect(0, 0, w, h);
    const matrix = confusionMatrix();
    const max = Math.max(1, ...matrix.flat());
    const left = 68, top = 24;
    const cell = Math.floor(Math.min((w - left - 12) / 10, (h - top - 55) / 10));
    const total = cell * 10;
    g.font = "10px system-ui";
    g.fillStyle = "#91a5c2";
    g.textAlign = "center";
    for (let x = 0; x < 10; x++) {
        g.fillText(String(x), left + cell * (x + .5), top - 7);
        for (let y = 0; y < 10; y++) {
            const val = matrix[y][x];
            const shade = val / max;
            g.fillStyle = "rgba(102,231,210," + (0.06 + .84 * shade).toFixed(3) + ")";
            g.fillRect(left + x * cell + 1, top + y * cell + 1, cell - 2, cell - 2);
            if (val) {
                g.fillStyle = shade > .55 ? "#10252e" : "#d0fbf3";
                g.textAlign = "center";
                g.fillText(String(val), left + cell * (x + .5), top + cell * (y + .5) + 4);
            }
        }
    }
    g.fillStyle = "#91a5c2";g.textAlign = "right";
    for (let y = 0; y < 10; y++) {
        g.fillText(NAMES[y].slice(0, 9), left - 7, top + cell * (y + .5) + 4);
    }
    g.textAlign = "center";
    g.fillText("Predicted class (0–9)", left + total / 2, top + total + 18);
    $("pred-counter").textContent = predictions.size ?
        predictions.size + " images classified" : "Not evaluated";
}
function begin() {
    if (worker) worker.terminate();
    currentConfig = config();
    worker = new Worker("./trainer.worker.js");
    history = [];
    predictions.clear();
    checkpoint = null;
    runStart = performance.now();
    $("metric-epoch").textContent = "0 / " + currentConfig.epochs;
    $("metric-loss").textContent = "—";
    $("metric-accuracy").textContent = "—";
    $("metric-ema").textContent = "—";
    $("logs").textContent = "";
    $("inference-result").textContent = "Training the Flow model…";
    progress(0, "Initializing the WebAssembly runtime.");
    status("Starting", "running");
    drawHistory();
    drawConfusion();
    drawSamples();
    setActive(true);
    timer = setInterval(() => {
        $("execution").textContent = ((performance.now() - runStart) / 1000).toFixed(1) + " s elapsed";
    }, 1000);

    worker.onmessage = ({data}) => {
        switch (data.type) {
            case "stage":
                status(data.phase, data.running ? "running" : "");
                progress(history.length / currentConfig.epochs * 90,
                         data.phase + " · " + history.length + "/" + currentConfig.epochs + " epochs complete");
                break;
            case "log":
                log((data.phase === "infer" ? "[inference] " : "") + data.text);
                break;
            case "metrics":
                if (![data.loss, data.accuracy, data.ema].every(Number.isFinite)) {
                    fail("Non-finite Flow training metrics");
                    return;
                }
                history.push(data);
                $("metric-epoch").textContent = data.epoch + " / " + currentConfig.epochs;
                $("metric-loss").textContent = format(data.loss, 4);
                $("metric-accuracy").textContent = percent(data.accuracy);
                $("metric-ema").textContent = percent(data.ema);
                progress(data.epoch / currentConfig.epochs * 90,
                         "Epoch " + data.epoch + "/" + currentConfig.epochs + " · Flow optimizer updated weights");
                drawHistory();
                break;
            case "prediction":
                predictions.set(data.index, data.predicted);
                drawSamples();
                drawConfusion();
                break;
            case "inferenceAccuracy":
                $("inference-result").textContent =
                    "EMA Flow inference accuracy: " + percent(data.accuracy) +
                    " on " + currentConfig.eval + " official test records.";
                break;
            case "complete":
                checkpoint = data.checkpoint;
                progress(100, "Flow training, checkpoint save, and inference finished.");
                status("Completed", "done");
                log("WASM experiment completed; EMA checkpoint is ready to download.");
                finish();
                break;
            case "error":
                fail(data.message);
                break;
        }
    };
    worker.onerror = (event) => {
        fail(event.message || "Browser Web Worker failed");
    };
    worker.postMessage({type: "run", config: currentConfig});
}
function finish() {
    if (timer) clearInterval(timer);
    timer = null;
    if (worker) worker.terminate();
    worker = null;
    setActive(false);
}
function fail(message) {
    log("ERROR: " + message);
    status("Failed", "error");
    progress(0, "Execution failed. See the Flow runtime console for details.");
    $("inference-result").textContent = "Experiment failed; no accuracy is claimed.";
    checkpoint = null;
    finish();
}
function stop() {
    if (!worker) return;
    log("Stopped by user. The dedicated WASM worker has been terminated.");
    status("Stopped", "");
    progress(0, "Run stopped. Reconfigure and start a fresh model.");
    $("inference-result").textContent = "Training was stopped before inference.";
    checkpoint = null;
    finish();
}
function downloadCheckpoint() {
    if (!checkpoint) return;
    const url = URL.createObjectURL(new Blob([checkpoint], {type: "application/octet-stream"}));
    const a = document.createElement("a");
    a.href = url;
    a.download = "hlb-ema-flow-wasm.bin";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function loadData() {
    try {
        const [metaResponse, testResponse] = await Promise.all([
            fetch("./data/meta.json"),
            fetch("./data/test.bin")
        ]);
        if (!metaResponse.ok || !testResponse.ok) {
            throw new Error("The official CIFAR-10 site assets are unavailable.");
        }
        meta = await metaResponse.json();
        testBytes = new Uint8Array(await testResponse.arrayBuffer());
        if (testBytes.length !== 32 * RECORD_BYTES ||
            meta.test_records !== 32 || meta.training_records !== 64 ||
            meta.source_md5 !== "c32a1d4ab5d03f1284b67883e8d87530") {
            throw new Error("Unexpected source provenance or CIFAR-10 record length.");
        }
        for (let i = 0; i < 32; i++) {
            if (testBytes[i * RECORD_BYTES] > 9) throw new Error("Invalid CIFAR label.");
        }
        drawSamples();
        drawConfusion();
        status("Ready", "done");
        $("start").textContent = "Start real Flow training";
        progress(0, "Ready to initialise Flow WebAssembly and train.");
        $("logs").textContent = "Official CIFAR-10 binary dataset verified at build time.\n";
        log("Training subset: 64 real records; test subset: 32 real records.");
        log("No synthetic fixtures, Python ML code, or prerecorded training metrics.");
        setActive(false);
    } catch (error) {
        status("Data unavailable", "error");
        $("start").textContent = "Dataset unavailable";
        $("logs").textContent = "ERROR: " + error.message +
            "\nCheck the GitHub Pages build artifact and official dataset step.";
    }
}
$("start").addEventListener("click", begin);
$("stop").addEventListener("click", stop);
$("download").addEventListener("click", downloadCheckpoint);
window.addEventListener("resize", () => {drawHistory();drawConfusion();});
drawHistory();
drawConfusion();
loadData();
