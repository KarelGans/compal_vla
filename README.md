# Open-box-flaps robot policies (pi0.5, ACT)

Trains policies (LeRobot 0.7.0) to open the four flaps of a box with the right arm of the igus
ReBeL dual-arm setup, one flap at a time. They see the head and right-wrist cameras and output
right-arm joint targets.
- **pi0.5** (`lerobot/pi05_base`, fine-tuned) reads the flap as a prompt ("open the back flap").
- **ACT** (trained from scratch) has no text input, so it gets the flap as 4 numbers (see Design notes).

## Layout

```
env.sh                       all paths; everything large lives on /media/ai/aipc
scripts/                     shared by every model
  prepare_dataset.py         raw Mocap dataset -> training dataset (already run)
  add_flap_onehot.py         copy of that dataset with the flap as 4 numbers, for ACT (already run)
  eval_open_loop.py          compare predictions with the 4 held-out recordings
  eval_checkpoints.sh        run eval_open_loop.py on every checkpoint of a run, then print the table below
  compare_checkpoints.py     table of all scored checkpoints, to pick the one to try on the robot
  run_robot.py               30 Hz control loop, plus --dry-run (replays a recording, no hardware)
  log_to_trackio.py          load a log file's curves into the dashboard (runs from before Trackio)
src/open_box_vla/            constants.py (joints, cameras, prompts), flap_sequencer.py,
                             policy_runner.py (FlapPolicy: robot reading + images -> 14-joint command, any model)
pi05/
  train.sh                   fine-tune pi0.5 (settings at the top of the script)
  resume.sh                  continue a run from its last checkpoint
act/
  config.yaml                every ACT hyperparameter
  train.sh                   train ACT with config.yaml
  resume.sh                  continue a run from its last checkpoint
```

| What | Where |
|---|---|
| Raw recordings (read only) | `/media/ai/aipc/Karel/Mocap/Dataset` |
| Prepared dataset (pi0.5) | `/media/ai/aipc/Karel/compal_vla_models/lerobot_datasets/compal/open_box_flaps_right6` |
| Same + flap numbers (ACT) | `/media/ai/aipc/Karel/compal_vla_models/lerobot_datasets/compal/open_box_flaps_right6_onehot` |
| Base model + tokenizer cache | `/media/ai/aipc/Karel/compal_vla_models/hf_cache` |
| Checkpoints | `/media/ai/aipc/Karel/Mocap/checkpoints/<pi05 or act>/<run>/checkpoints/<step>/pretrained_model` |
| Training logs | `/media/ai/aipc/Karel/Mocap/checkpoints/<pi05 or act>/logs/<run>.log` |
| Training curves (Trackio) | `/media/ai/aipc/Karel/Mocap/checkpoints/trackio/` |

Keep large files out of `/home` and `/tmp`: user1 has a 250 GB quota there that is almost full.
A checkpoint that cannot be written kills the run with no error message.

## Tutorial (pi0.5)

All commands run from `~/compal_vla_open_box`. Steps 5 and 6 work the same for ACT checkpoints.

**1. One-time: tokenizer access.** pi0.5 uses the gated PaliGemma tokenizer.
Accept the license at https://huggingface.co/google/paligemma-3b-pt-224, then:
```bash
uv run hf auth login        # token is stored in ~/.cache/huggingface, not on the shared drive
```

**2. Five-minute test.** Before a day-long run, check that saving, evaluation and the dry run work:
```bash
RUN_NAME=smoke STEPS=20 SAVE_FREQ=20 BATCH_SIZE=4 bash pi05/train.sh
source env.sh
CKPT=$CHECKPOINT_ROOT/pi05/smoke/checkpoints/last/pretrained_model
uv run python scripts/eval_open_loop.py --checkpoint $CKPT --max-queries 3
uv run python scripts/run_robot.py --checkpoint $CKPT --dry-run --dry-run-steps 100
rm -r $CHECKPOINT_ROOT/pi05/smoke        # ~24 GB
```

**3. Train.** Run inside tmux so it survives closing the terminal:
```bash
tmux new -s pi05
STEPS=10000 bash pi05/train.sh           # default is 30000; Ctrl-b d to detach
tmux attach -t pi05                      # come back later
```
Check settings without training: `DRY_RUN=1 bash pi05/train.sh`.
Other settings go in front: `RUN_NAME`, `STEPS`, `BATCH_SIZE`, `SAVE_FREQ`, `NUM_WORKERS`.
Extra LeRobot flags go after: `bash pi05/train.sh --policy.use_relative_actions=true`.

**Watch the curves in a browser.** Runs log loss, learning rate, gradient norm and GPU use to
Trackio, a local dashboard similar to TensorBoard (LeRobot 0.7 has no TensorBoard support):
```bash
source env.sh
uv run trackio show --project pi05_open_box   # prints http://127.0.0.1:7860/ ; leave it running
```
The page refreshes while training runs. It only listens on the server itself. From your own computer, open it through
VS Code Remote-SSH (Ports tab, forward 7860) or `ssh -L 7860:localhost:7860 user1@ai-iece`, then browse to http://localhost:7860.

**4. Resume after a crash or reboot, or to train longer:**
```bash
bash pi05/resume.sh pi05_open_box_20261114_1300          # the run's folder name
bash pi05/resume.sh <run> --steps=50000 --scheduler.num_decay_steps=50000   # extend to 50k steps
```
It restarts from the last *saved* checkpoint, so steps since then are redone. The data order continues
exactly where it stopped. When extending, also pass `--scheduler.num_decay_steps` with the new step count;
otherwise the learning rate reaches its minimum at step 30,000 and stays there.
For a run trained before Trackio was switched on, first load its old curves:
`uv run python scripts/log_to_trackio.py <run>` (after `source env.sh`).

**5. Evaluate on the held-out episodes** (10, 19, 30, 40; never seen in training):
```bash
source env.sh
uv run python scripts/eval_open_loop.py --checkpoint $CHECKPOINT_ROOT/pi05/<run>/checkpoints/last/pretrained_model
```
This writes one plot per episode and `summary.json` next to the checkpoint.
`mae_chunk` must clearly beat `hold_mae_chunk`, the error of a robot that never moves.
It scores the first 50 predicted steps (`--horizon`) for every model, so pi0.5 and ACT scores compare directly.
Scores are printed per episode and per flap. Add `--prompt-swap` to test whether the model follows the prompt:
every query is asked again with the three wrong flaps, and the right prompt should give the lowest error
(4x the queries; chance level for "right prompt fits best" is 25%).

**Overfitting, and which checkpoint to use.** The training loss keeps falling either way, so it cannot show
overfitting. Score every saved checkpoint on the held-out episodes and on 4 training episodes:
```bash
bash scripts/eval_checkpoints.sh pi05/<run>      # or act/<run>; checkpoints already scored are skipped
```
It ends with a table (print it again any time: `uv run python scripts/compare_checkpoints.py pi05/<run> act/<run>`)
and adds the scores to the run's dashboard (`eval` section: `heldout_mae`, `train_episodes_mae`).
While `heldout_mae` keeps falling, training longer helps. If it stops falling or rises while `train_episodes_mae`
keeps falling, the model is memorizing its training episodes: keep the checkpoint with the lowest `heldout_mae`.
With only 4 held-out episodes, look at the trend rather than small differences. These are open-loop scores,
so try the best two checkpoints on the robot.

**6. Dry run, then the robot.** First replay a recording through the full control loop:
```bash
uv run python scripts/run_robot.py --checkpoint <...>/pretrained_model --dry-run
```
For hardware, implement `IgusRebelRobot` in `scripts/run_robot.py` (two camera images + 14 joint angles in,
14 joint targets out), then:
```bash
uv run python scripts/run_robot.py --checkpoint <...>/pretrained_model --order back front right left
```
Type `n` + Enter for the next flap, `q` + Enter to stop. A flap also advances after 90 s.
Each command is limited to 0.15 rad from the current joint position.

## ACT

Every hyperparameter is in `act/config.yaml`, with a comment on each. To train:
```bash
tmux new -s act
bash act/train.sh                                   # uses act/config.yaml
bash act/train.sh --batch_size=16 --steps=50000     # or change single values for one run
CONFIG=act/config_aug.yaml bash act/train.sh        # same + image augmentation (or any copy you edited)
uv run trackio show --project act_open_box          # curves (after source env.sh)
bash act/resume.sh <run>                            # continue after a crash
```
Each run keeps a copy of the YAML it used in `$CHECKPOINT_ROOT/act/logs/<run>.yaml`.
Checkpoints are saved every 10k steps: evaluate each one (step 5 above) and keep the best.
Evaluate and dry-run ACT checkpoints with steps 5 and 6 above (same scripts, same flags). For ACT,
`run_robot.py --temporal-ensemble 0.01` predicts at every tick and averages the overlapping chunks,
as in the ACT paper (~17 ms per tick); compare it with the default `--n-action-steps 25`.

## Numbers

- **Dataset:** 41 episodes, 180,811 frames (~100 min at 30 fps), 2 cameras, 4 flap prompts. 37 episodes are for training and 4 are held out.
- **Training speed:** at batch 32, ~3 s/step using ~45 GB of GPU memory. One pass over the data is ~5,100 steps (~4.3 h); 10k steps ≈ 8.5 h; 30k ≈ 25 h.
- **Disk:** each checkpoint is 24 GB, and LeRobot keeps all of them. Once you won't resume from a checkpoint, deleting its
  `training_state/` folder saves ~15 GB. You can still evaluate and deploy it from `pretrained_model/`.
- **ACT:** 52M parameters, 3.75 GB of GPU memory at batch 8. A 300-step test ran at ~0.15 s/step with 4 data
  workers while pi0.5 was training on the same GPU, so 100k steps ≈ 4 h or less. Half of each step was
  waiting for video decoding, so more workers help. Checkpoints are 0.6 GB. A fresh 100-step chunk takes
  ~17 ms (pi0.5: ~97 ms), so ACT can re-plan every 33 ms tick.

## Design notes

- **Only the right arm is learned.** In the recordings the left arm and both grippers never move.
  The model sees and predicts the 6 right-arm joints. At run time the left arm holds its pose and
  the grippers get the recorded constants.
- **The model doesn't decide when a flap is done.** Every frame is labelled with one flap, so
  an operator (`n`) or a timeout advances the prompt (`src/open_box_vla/flap_sequencer.py`).
  A success detector can call `FlapSequencer.advance()` later.
- **Actions are absolute joint targets** about 7 frames (0.23 s) ahead of the measured state.
  `--policy.use_relative_actions=true` is worth trying if the model drifts.
- **Re-planning:** `run_robot.py --n-action-steps 25` executes 25 steps of each predicted chunk
  (pi0.5: 50 steps, ACT: 100) before querying again. Lower values react faster and query the model more often.
- **ACT gets the flap as numbers.** The four recording sets open the flaps in different orders
  (back or front first, then right or left third), so a model without the prompt cannot know which
  flap comes next. `scripts/add_flap_onehot.py` adds `observation.environment_state` =
  [back, front, right, left] with a 1 for the current flap. At run time `FlapPolicy` converts
  the prompt into these numbers, so the scripts and `--order` work the same for both models.
