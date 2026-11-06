"use strict";
/* Actual WebAssembly smoke test. Fails CI on missing, stubbed or broken model. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const site = path.resolve(__dirname, "../site");

function cifarFs(mod, trainBytes, testBytes) {
    mod.FS.mkdir("/data");
    mod.FS.mkdir("/data/cifar-10-batches-bin");
    mod.FS.writeFile("/data/cifar-10-batches-bin/data_batch_1.bin", trainBytes);
    mod.FS.writeFile("/data/cifar-10-batches-bin/test_batch.bin", testBytes);
}

async function start(factoryName, lines) {
    const create = require(path.join(site, factoryName + ".js"));
    return create({
        locateFile: (name) => path.join(site, name),
        print: (line) => lines.push(line),
        printErr: (line) => lines.push("stderr: " + line)
    });
}

function run(mod, dump) {
    assert.equal(mod._flow_web_configure(16, 8, 2, 2, 1, dump ? 0 : 4, dump), 0);
    const code = mod.callMain([]);
    assert.ok(code === 0 || code === undefined, "Nonzero Flow main return " + code);
}

(async function main() {
    const trainBytes = fs.readFileSync(path.join(site, "data/train.bin"));
    const testBytes = fs.readFileSync(path.join(site, "data/test.bin"));
    assert.equal(trainBytes.length, 64 * 3073);
    assert.equal(testBytes.length, 32 * 3073);
    const trainLines = [];
    const trainer = await start("train", trainLines);
    cifarFs(trainer, trainBytes, testBytes);
    run(trainer, 0);
    const checkpoint = trainer.FS.readFile("/hlb-ema.bin");
    assert.ok(checkpoint.byteLength > 100, "Flow training produced no EMA checkpoint");
    assert.ok(trainLines.some((line) => /train_loss/.test(line)), "No real Flow training metrics");
    assert.ok(trainLines.some((line) => /checkpoint saved: hlb-ema.bin/.test(line)),
              "Flow checkpoint save not reported");

    const inferenceLines = [];
    const inference = await start("infer", inferenceLines);
    cifarFs(inference, trainBytes, testBytes);
    inference.FS.writeFile("/hlb-ema.bin", checkpoint);
    run(inference, 1);
    const predictions = inferenceLines.filter((line) => /^PRED [0-9]+ [0-9]+$/.test(line));
    assert.equal(predictions.length, 8, "Flow inference did not classify 8 real test images");
    assert.ok(inferenceLines.some((line) => /^accuracy (0(\.[0-9]+)?|1(\.0+)?)$/.test(line)),
              "Flow did not print real inference accuracy");
    console.log("PASS: real CIFAR-10 -> Flow WASM training -> checkpoint -> Flow WASM predictions");
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
