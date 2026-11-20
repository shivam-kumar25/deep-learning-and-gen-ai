# deep-learning-and-gen-ai

Deep learning from the single neuron to CNNs, transfer learning and U-Net. The notebooks are in the topic folders
(`deep-learning/`, `XOR-problem/`, `pytorch/`, `cnn/`, and the project folders); the same material is written up as
study notes in `notes/`, read through **Studyloom**, a small web app whose code cells run on a live Python kernel.

## Run the app

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/kernel/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
npm install
npm run build
npm start                 # http://localhost:4300
```

Sign up (the account lives in `data/studyloom.db` on this computer), open a note and press **▶ Run** on any cell.

- **One kernel per note.** Cells share their variables like a notebook; running a cell first runs the unrun cells above
  it. **+ cell** adds a scratch cell; the chip at the bottom left restarts the kernel; **■ Stop** stops a long cell.
- **Where code runs.** In `workspace/` with the project's `.venv` (PyTorch, on the GPU when CUDA is available).
  `workspace/dlviz.py` holds the drawing helpers the figures use; datasets go in `workspace/data/` (not in git).
- **Who can run code.** Only signed-in users, only from this computer. `KERNEL_ALLOW_LAN=1` also allows private-network
  addresses, `KERNEL_MAX` caps live kernels (default 4), `KERNEL_IDLE_MIN` shuts idle ones (default 30), `KERNEL=off`
  turns it off (cells then run in the browser with Pyodide, without PyTorch).

## Notes tooling

- Note sources: `tools/notes/src/<subject>/NN_*.md`. Build one subject with
  `tools/notes/build/build_dl.sh d1-neural-network-foundations`, the home page with `tools/notes/build/build_dl.sh home`.
  Every cell is run while building, so the outputs in the pages are real. After adding or renaming notes, run
  `npm run index` so the app's catalogue and search pick them up (the server only indexes an empty database).
- Tests: `npm test` (API and live kernel), `node tools/notes/test/kernel-e2e.mjs http://127.0.0.1:4300 d1` (every cell of
  the notes on the running app's kernel).
