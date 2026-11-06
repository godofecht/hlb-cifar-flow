"use strict";
/*
 * This Worker runs the real Flow-generated WASM executable. Nothing in JS
 * computes logits, gradients, losses or optimizer updates.
 */
let active = false;
const RECORD_BYTES = 3073;
const ROOT = "/data/cifar-10-batches-bin";

function send(type, payload) {
    self.postMessage(Object.assign({ type: type }, payload || {}));
}

async function load(name, expectedBytes) {
    const response = await fetch("./data/" + name + ".bin");
    if (!response.ok) {
        throw new Error("HTTP " + response.status + " downloading official " + name + " subset");
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== expectedBytes) {
        throw new Error("Incorrect " + name + " dataset byte length: " + bytes.length);
    }
    return bytes;
}

function mountDataset(mod, train, test) {
    mod.FS.mkdir("/data");
    mod.FS.mkdir(ROOT);
    mod.FS.writeFile(ROOT + "/data_batch_1.bin", train);
    mod.FS.writeFile(ROOT + "/test_batch.bin", test);
}

function receiveLine(phase, text) {
    send("log", { phase: phase, text: String(text) });
    if (phase === "train") {
        const match = /^epoch ([0-9]+) train_loss ([0-9.eE+-]+) eval_accuracy ([0-9.eE+-]+) ema_eval_accuracy ([0-9.eE+-]+)/.exec(text);
        if (match) {
            send("metrics", {
                epoch: Number(match[1]),
                loss: Number(match[2]),
                accuracy: Number(match[3]),
                ema: Number(match[4])
            });
        }
    } else {
        const pred = /^PRED ([0-9]+) ([0-9]+)$/.exec(text);
        if (pred) {
            send("prediction", { index: Number(pred[1]), predicted: Number(pred[2]) });
        }
        const accuracy = /^accuracy ([0-9.eE+-]+)/.exec(text);
        if (accuracy) {
            send("inferenceAccuracy", { accuracy: Number(accuracy[1]) });
        }
    }
}

function configureAndExecute(mod, cfg, dump) {
    const rc = mod._flow_web_configure(
        cfg.train, cfg.eval, cfg.width, cfg.batch, cfg.epochs,
        dump ? 0 : Math.min(8, Math.floor(cfg.train / 4)), dump
    );
    if (rc !== 0) {
        throw new Error("Flow browser configuration rejected parameters: " + rc);
    }
    const exit = mod.callMain([]);
    if (typeof exit === "number" && exit !== 0) {
        throw new Error("Flow executable returned exit code " + exit);
    }
}

async function createModule(factory, phase, train, test) {
    const module = await factory({
        locateFile: (name) => new URL(name, self.location.href).href,
        print: (s) => receiveLine(phase, s),
        printErr: (s) => receiveLine(phase, "stderr: " + s),
        noInitialRun: true
    });
    mountDataset(module, train, test);
    return module;
}

async function run(cfg) {
    if (active) {
        throw new Error("Training is already in progress");
    }
    active = true;
    if (![16, 32, 64].includes(cfg.train) ||
        ![8, 16, 32].includes(cfg.eval) ||
        ![2, 4].includes(cfg.width) ||
        ![2, 4, 8].includes(cfg.batch) ||
        ![1, 2, 3].includes(cfg.epochs) ||
        cfg.batch > cfg.train) {
        throw new Error("Invalid training controls");
    }

    send("stage", { phase: "Loading genuine CIFAR-10 subsets" });
    const [train, test] = await Promise.all([
        load("train", 64 * RECORD_BYTES),
        load("test", 32 * RECORD_BYTES)
    ]);
    send("stage", { phase: "Instantiating Flow trainer WASM" });
    importScripts("./train.js");
    if (typeof self.createFlowTrainer !== "function") {
        throw new Error("Missing compiled Flow WASM factory: createFlowTrainer");
    }
    const trainer = await createModule(self.createFlowTrainer, "train", train, test);
    send("stage", { phase: "Training in Flow", running: true });
    configureAndExecute(trainer, cfg, 0);

    const checkpoint = trainer.FS.readFile("/hlb-ema.bin");
    if (checkpoint.byteLength < 100) {
        throw new Error("Flow training returned without an EMA checkpoint");
    }

    send("stage", { phase: "Flow inference with EMA checkpoint", running: true });
    importScripts("./infer.js");
    if (typeof self.createFlowInference !== "function") {
        throw new Error("Missing compiled Flow WASM factory: createFlowInference");
    }
    const inference = await createModule(self.createFlowInference, "infer", train, test);
    inference.FS.writeFile("/hlb-ema.bin", checkpoint);
    configureAndExecute(inference, cfg, 1);
    const exportedCheckpoint = checkpoint.slice();
    send("complete", {
        checkpoint: exportedCheckpoint.buffer,
        train: cfg.train,
        eval: cfg.eval
    });
    active = false;
}

self.onmessage = (event) => {
    if (!event.data || event.data.type !== "run") {
        return;
    }
    run(event.data.config).catch((error) => {
        send("error", { message: error && error.stack ? error.stack : String(error) });
        active = false;
    });
};
