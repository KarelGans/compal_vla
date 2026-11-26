# Open-box-flaps robot policies (pi0.5, ACT)

Trains policies (LeRobot 0.7.0) to open the flaps of a box with the right arm of the igus
ReBeL dual-arm setup, one flap at a time. They see the robot's cameras and output right-arm
joint targets.
- **pi0.5** (`lerobot/pi05_base`, fine-tuned) reads the flap as a prompt ("open the back flap").
- **ACT** (trained from scratch) has no text input, so it gets the flap as 4 numbers (see Design notes).

There are four training datasets, each built from the raw recordings as its YAML in `datasets/` says:

| Dataset | YAML | Episodes | Cameras |
|---|---|---|---|
| `compal/open_box_thor_v3_right6` (current) | `datasets/thor_v3.yaml` | 83: thor_v2 plus the 2026-11-26 front, left and right recordings, one flap per episode, all trained on (none held out) | head, wrist_right, knife_left |
| `compal/open_box_thor_v2_right6` | `datasets/thor_v2.yaml` | 54: one flap per episode (new front, left, right; back from thor), all ending at the same pose | head, wrist_right, knife_left |
| `compal/open_box_thor_right6` | `datasets/thor.yaml` | 44: front or back flap, one per episode, all ending at the same pose | head, wrist_right, knife_left |
| `compal/open_box_flaps_right6` (first) | `datasets/allflaps.yaml` | 41: all four flaps in a row | head, wrist_right |

## Layout

```
env.sh                       all paths; everything large lives on /media/ai/aipc
datasets/                    one YAML per dataset: raw folders, label fixes, dropped episodes, robot facts
scripts/                     shared by every model
  prepare_dataset.py         raw recordings -> training dataset, as a datasets/*.yaml says
  add_flap_onehot.py         copy of a dataset with the flap as 4 numbers, for ACT
  train_with_heldout.py      lerobot-train plus a held-out loss curve (pi05/train.sh uses it)
  eval_open_loop.py          compare predictions with recordings (held-out, or training episodes)
  eval_checkpoints.sh        run eval_open_loop.py on every checkpoint of a run, then print the table below
  compare_checkpoints.py     table of all scored checkpoints, to pick the one to try on the robot
  run_robot.py               30 Hz control loop, plus --dry-run (replays a recording, no hardware)
  log_to_trackio.py          load a log file's curves into the dashboard (runs from before Trackio)
src/open_box_vla/            constants.py, dataset_specs.py (reads datasets/*.yaml),
                             flap_sequencer.py (flap order, finish-pose detector),
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
| Raw recordings (read only) | thor: `/media/ai/aipc/Karel/Mocap/datasets`, thor_v2: also its `dataset_v2/` subfolder, thor_v3: also `dataset_20261126/`, first: `/media/ai/aipc/Karel/Mocap/Dataset` |
| Prepared datasets | `/media/ai/aipc/Karel/compal_vla_models/lerobot_datasets/compal/<name>` (ACT: `<name>_onehot`) |
| Base model + tokenizer cache | `/media/ai/aipc/Karel/compal_vla_models/hf_cache` |
| Checkpoints | `/media/ai/aipc/Karel/Mocap/checkpoints/<pi05 or act>/<run>/checkpoints/<step>/pretrained_model` |
| Training logs | `/media/ai/aipc/Karel/Mocap/checkpoints/<pi05 or act>/logs/<run>.log` |
| Training curves (Trackio) | `/media/ai/aipc/Karel/Mocap/checkpoints/trackio/` |

Keep large files out of `/home` and `/tmp`: user1 has a 250 GB quota there that is almost full.
A checkpoint that cannot be written kills the run with no error message.

## The thor_v3 dataset (current)

`datasets/thor_v3.yaml` is thor_v2 plus the 2026-11-26 recordings in `dataset_20261126/`, which add
front, left and right episodes. Compared with thor_v2:
- **Left and right recorded together.** Each episode of `dataset_20261126/thor_box_flapright_newmovement`
  opens the left flap, goes back to the rest pose, then opens the right flap; the label was switched
  in between (episode 9: left only). `split_by_label` cuts every episode where its label changes, so
  each flap becomes an episode of its own. A cut must fall while the arm rests at the finish pose, or
  the build stops. In episode 1 the label was switched 13 s early, so the YAML puts that cut at 56.6 s.
  Without the cuts, the action chunks at the end of the left flap would hold the start of the right one.
- **Labels checked:** the head camera shows the left flap opening first in every episode, and every
  piece's joint path is closest to the thor_v2 recordings of its flap.
- **9 dropped** (the 5 of thor_v2, plus): new front 1 (works the half-open flap for ~30 s, a hand reaches
  into the box at the end), new front 2 and the right part of episode 0 (cut off before the rest pose),
  and episode 2 (goes from the left flap to the right one without returning to the rest pose).
- **Nothing held out** (`heldout_per_subset: 0`): all 83 kept episodes are trained on. So there is no
  held-out loss curve and no held-out score; the checkpoint is picked on the robot (step 5).
- **Balance.** Training episodes: front 18, back 27, left 20, right 18 (thor_v2: 12, 25, 6, 6).
  Training draws front 26%, back 26%, left 25% and right 23% of its samples.
- **EDA:** `notebooks/eda_thor_v3.ipynb` (Run All after a rebuild).

Episodes start as in thor_v2: front at the home pose with the box closed; back, left and right at
the finish pose with the flaps before them open.

## The thor_v2 dataset

`datasets/thor_v2.yaml` combines the `dataset_v2` recordings (new movements for the front, left and
right flaps) with the back-flap folders of the thor recordings. The robot opens the flaps in the order
front, back, left, right, the order they were recorded in: every flap's recordings start with the
flaps before it already open. Compared with the raw recordings:
- **Labels.** All five folders were recorded as "open the front flap". From the camera images,
  `flapleft`, `flapright` and the two `_back` folders are relabelled "open the left/right/back flap".
- **Not used:** the thor front folders. Their front movement is completely different from the new
  one, and under the same prompt the model would mix the two.
- **5 of 59 episodes dropped:** new front 0 (opens the flap but never returns to the rest pose) and 13
  (cut off with the flap still closed), plus the same three back episodes as in the thor dataset.
- **Held out:** the last episode of each folder, i.e. dataset episodes 12, 25, 39, 46 and 53.
- **Flap balance.** The training episodes are front 12, back 25, left 6 and right 6. With
  `flap_balance_exponent: 0.43` (how pi0 mixed datasets of different sizes), training draws front 29%,
  back 33%, left 19% and right 18% of its samples, instead of 32 / 45 / 13 / 11% by frames.
  `scripts/train_with_heldout.py` applies it; the build prints the shares.

Every new front episode starts at the home pose with the box closed. Every other episode starts at
the finish pose. New front episodes take up to 84 s, close to the robot's 90 s per-flap timeout, so run the
robot with `--timeout-s 120`.

## The thor dataset

What `datasets/thor.yaml` changes compared with the raw recordings (which are never modified):
- **Labels.** All four folders were recorded as "open the front flap". The `_back` folders open the
  back flap (camera images, and their joint ranges match the back flap of the first dataset), so
  they are relabelled "open the back flap".
- **6 of 50 episodes dropped:**
  - `box2_back` 0 and `box2_front` 1 are cut off with the flap still closed.
  - `box2_front` 0, `test1_back` 0 and `test1_front` 0 are each session's first episode. They keep pushing the already-open flap for ~30 s before finishing.
  - `box2_back` 1 approaches the flap along a path no other back-flap episode uses.
- **Held out** (never trained on): the last episode of each folder, i.e. dataset episodes 12, 20, 34 and 43.
- **Kept as recorded:** the 2–6 s of standing still at the start of every episode.

Every episode ends with the arm at one fixed pose (`finish_pose` in the YAML), standing still for 1–8 s.
Front episodes start at the home pose with the box closed. Back episodes start at the finish pose,
mostly with the front flap already open, so "front, then back" is in the data.

## Tutorial (pi0.5)

All commands run from `~/compal_vla_open_box`. Steps 5 and 6 work the same for ACT checkpoints.

**0. Build the dataset** (done for thor_v3, thor_v2 and thor; about a minute; re-run after editing the YAML).
It checks every episode against the raw recordings: frames, joints, labels and camera images.
```bash
source env.sh
uv run python scripts/prepare_dataset.py datasets/thor_v3.yaml --overwrite
```

**1. One-time: tokenizer access.** pi0.5 uses the gated PaliGemma tokenizer.
Accept the license at https://huggingface.co/google/paligemma-3b-pt-224, then:
```bash
uv run hf auth login        # token is stored in ~/.cache/huggingface, not on the shared drive
```

**2. Five-minute test.** Before a day-long run, check that saving, evaluation and the dry run work
(it trains at batch 4; for the memory and speed of a real run, see Numbers below):
```bash
RUN_NAME=smoke_thor STEPS=20 SAVE_FREQ=20 BATCH_SIZE=4 bash pi05/train.sh
source env.sh
CKPT=$CHECKPOINT_ROOT/pi05/smoke_thor/checkpoints/last/pretrained_model
uv run python scripts/eval_open_loop.py --checkpoint $CKPT --train-probe --max-queries 3
uv run python scripts/run_robot.py --checkpoint $CKPT --dry-run --dry-run-steps 100
rm -r $CHECKPOINT_ROOT/pi05/smoke_thor        # ~24 GB
```
On a dataset with held-out episodes (thor_v2, thor), also test the held-out loss: add `EVAL_STEPS=10
HELDOUT_STRIDE=100` to the first line, and the log shows `eval_loss=` at steps 10 and 20.

**3. Train.** Run inside tmux so it survives closing the terminal:
```bash
tmux new -s pi05
STEPS=10000 SAVE_FREQ=2000 bash pi05/train.sh   # thor_v3 (env.sh): 10k steps, a checkpoint every 2k; Ctrl-b d to detach
tmux attach -t pi05                             # come back later
```
Without `STEPS` and `SAVE_FREQ` it trains 30000 steps and saves every 10000.
The first line it prints names the dataset and the held-out episodes (thor_v3: `none (all episodes train)`).
Check settings without training: `DRY_RUN=1 bash pi05/train.sh`.
Other settings go in front: `RUN_NAME`, `STEPS`, `BATCH_SIZE`, `SAVE_FREQ`, `EVAL_STEPS`, `HELDOUT_STRIDE`,
`NUM_WORKERS`, `DATASET_REPO_ID` (default: thor_v3, see `env.sh`).
Extra LeRobot flags go after: `bash pi05/train.sh --policy.use_relative_actions=true`.

**Watch the curves in a browser.** Runs log the training loss, the held-out loss (datasets with held-out
episodes), learning rate, gradient norm and GPU use to Trackio, a local dashboard similar to TensorBoard (LeRobot 0.7 has no
TensorBoard support):
```bash
source env.sh
uv run trackio show --project pi05_open_box   # prints http://127.0.0.1:7860/ ; leave it running
```
The page refreshes while training runs. It only listens on the server itself. From your own computer, open it through
VS Code Remote-SSH (Ports tab, forward 7860) or `ssh -L 7860:localhost:7860 user1@ai-iece`, then browse to http://localhost:7860.

**4. Resume after a crash or reboot, or to train longer:**
```bash
bash pi05/resume.sh pi05_open_box_thor_right6_20261122_1300          # the run's folder name
bash pi05/resume.sh <run> --steps=50000 --scheduler.num_decay_steps=50000   # extend to 50k steps
```
It restarts from the last *saved* checkpoint, so steps since then are redone. The data order continues
exactly where it stopped. When extending, also pass `--scheduler.num_decay_steps` with the new step count;
otherwise the learning rate reaches its minimum at step 30,000 and stays there.
For a run trained before Trackio was switched on, first load its old curves:
`uv run python scripts/log_to_trackio.py <run>` (after `source env.sh`).

**5. Evaluate on the held-out episodes** (never seen in training). The dataset is the one the
checkpoint was trained on:
```bash
source env.sh
uv run python scripts/eval_open_loop.py --checkpoint $CHECKPOINT_ROOT/pi05/<run>/checkpoints/last/pretrained_model
```
This writes one plot per episode and `summary.json` next to the checkpoint.
`mae_chunk` must clearly beat `hold_mae_chunk`, the error of a robot that never moves.
It scores the first 50 predicted steps (`--horizon`) for every model, so pi0.5 and ACT scores compare directly.
Scores are printed per episode and per flap. Add `--prompt-swap` to test whether the model follows the prompt:
every query is asked again with the dataset's other flap prompts, and the right prompt should give the lowest
error (chance level for "right prompt fits best": 50% with front/back, 25% with four flaps).

**thor_v3 holds no episodes out**, so there is nothing new to score on: add `--train-probe` (one training
episode per folder and flap). A low error there only shows that the model reproduces its own recordings.
The `--prompt-swap` test still shows whether it tells the flaps apart (left vs right in particular).
`bash scripts/eval_checkpoints.sh pi05/<run>` does this for every checkpoint and prints a table without a
pick. Choose by trying checkpoints on the robot (step 6); the rest of this step is for datasets with held-out episodes.

**Overfitting, and which checkpoint to use.** The training loss keeps falling either way, so on its own it
cannot show overfitting. Two measurements can:

1. *Live, during training:* `eval/eval_loss` in Trackio is the same loss as `train/loss`, on the same
   scale, computed every 1000 steps on the held-out episodes (`scripts/train_with_heldout.py`; every
   5th frame, and the same random draws each time, so the curve only moves when the model changes).
   - Both falling: the model is still learning things that carry over to new episodes.
   - `eval_loss` flat while `train/loss` keeps falling: the model has started memorizing its training episodes.
   - `eval_loss` rising for several points in a row while `train/loss` keeps falling: overfitting.
     The steps around the lowest `eval_loss` are the ones to keep.

   For pi0.5 and other flow-matching policies this curve often bottoms out early and then creeps up while
   the robot keeps getting better for a while. Treat a small rise as a hint. A sustained rise with a growing
   gap to `train/loss` is the signal.
2. *Per saved checkpoint, in radians:*
   ```bash
   bash scripts/eval_checkpoints.sh pi05/<run>      # or act/<run>; checkpoints already scored are skipped
   ```
   This scores each checkpoint on the held-out episodes and on one training episode per recording folder.
   It ends with a table (print it again any time: `uv run python scripts/compare_checkpoints.py pi05/<run> act/<run>`)
   and adds the scores to the run's dashboard (`eval/heldout_mae`, `eval/train_episodes_mae`).
   `heldout_mae` far above `train_episodes_mae` means the model knows its training episodes much better
   than new ones.

There are only a few held-out episodes (5 in thor_v2, one per folder), so look at the trend rather than single
points. Both measurements are open-loop: try the best two checkpoints on the robot. Checkpoints are saved
every `SAVE_FREQ` steps (default 10k). If the held-out loss is lowest between two of them, train again with
`STEPS` near that point, or use a smaller `SAVE_FREQ`.

**6. Dry run, then the robot.** First replay a held-out recording through the full control loop
(thor_v3: a training episode; `--dry-run-episode` picks another; `--dry-run-steps 0` replays the whole
episode and shows when the finish detector fires):
```bash
uv run python scripts/run_robot.py --checkpoint <...>/pretrained_model --dry-run --dry-run-steps 0
```
For hardware, implement `IgusRebelRobot` in `scripts/run_robot.py`. Its `get_observation()` returns
`{"images": {"head": ..., "wrist_right": ..., "knife_left": ...}, "joints": ...}`: one 640x480 RGB image per
camera the checkpoint uses, plus the 14 joint angles. It sends 14 joint targets. Then:
```bash
uv run python scripts/run_robot.py --checkpoint <...>/pretrained_model      # thor: front, then back
```
A flap ends in one of three ways:
- the arm is back at the dataset's finish pose for 1 s (`--no-auto-finish` turns this off);
- you type `n` + Enter;
- 90 s pass (`--timeout-s`).

`q` + Enter stops. After the last flap, or on `q`, the arm is told to hold its current pose.
Each command is limited to 0.15 rad from the current joint position. At the start of each flap the script
warns if the arm is far from where that flap's recordings start: home for front, the finish pose for back.

## ACT

Every hyperparameter is in `act/config.yaml`, with a comment on each. It trains on the first dataset
(four flaps, two cameras). For the thor dataset, first run `uv run python scripts/add_flap_onehot.py`
(after `source env.sh`; writes `compal/open_box_thor_right6_onehot`), then use a copy of the YAML with
`dataset.repo_id: compal/open_box_thor_right6_onehot` and `dataset.exclude_episodes: [12, 20, 34, 43]`.
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

- **thor_v3:** 83 episodes, all for training, 153,494 frames (85 min at 30 fps), 3 cameras, 4 prompts.
  One pass over the data at batch 32 is ~4,800 steps, so 10k steps is ~2 passes.
- **Thor dataset:** 44 episodes (40 for training, 4 held out), 74,598 frames (41 min at 30 fps), 3 cameras,
  2 prompts. Episodes last 43–74 s (median 57 s). One pass over the training data at batch 32 is ~2,100 steps,
  so 30k steps is ~14 passes.
- **First dataset:** 41 episodes, 180,811 frames (~100 min), 2 cameras, 4 prompts. 37 for training, 4 held out.
- **Training speed (pi0.5, batch 32):** with 2 cameras, ~3 s/step and ~45 GB of GPU memory (10k steps ≈ 8.5 h).
  With 3 cameras (thor), ~3.9 s/step and ~54 GB, so 30k steps take ~33 h, including the held-out loss
  every 1000 steps. The thor_v2 run (3 cameras, GPU shared with other jobs) averaged ~9 s/step: 10k steps took ~25 h.
- **Disk:** each pi0.5 checkpoint is 24 GB, and LeRobot keeps all of them. Once you won't resume from a checkpoint,
  deleting its `training_state/` folder saves ~15 GB. You can still evaluate and deploy it from `pretrained_model/`.
- **ACT:** 52M parameters, 3.75 GB of GPU memory at batch 8. A 300-step test ran at ~0.15 s/step with 4 data
  workers while pi0.5 was training on the same GPU, so 100k steps ≈ 4 h or less. Half of each step was
  waiting for video decoding, so more workers help. Checkpoints are 0.6 GB. A fresh 100-step chunk takes
  ~17 ms (pi0.5: ~97 ms), so ACT can re-plan every 33 ms tick.

## Design notes

- **Only the right arm is learned.** In the recordings the left arm and both grippers never move.
  The model sees and predicts the 6 right-arm joints. At run time the left arm holds its pose and
  the grippers get the constant commands from the dataset's YAML (thor: both 0; first dataset: right gripper 1).
- **Stopping.** In the first dataset each episode opened all four flaps in a row. A flap's last action
  chunks already contained the motion towards the next flap, so nothing in the data showed the arm stopping.
  In the thor dataset each episode is one flap and ends with the arm at a fixed pose, standing still. The model
  learns "flap open: go to the finish pose and stay". `run_robot.py` treats "back at that pose for 1 s" as
  "flap done", so stopping does not depend on the model alone. The arm never passes through that pose in the
  middle of a recording (closest ~0.1 rad). Back-flap episodes also start at that pose, so the detector waits
  until the arm has first moved away.
- **Actions are absolute joint targets** about 7 frames (0.23 s) ahead of the measured state.
  `--policy.use_relative_actions=true` is worth trying if the model drifts.
- **Re-planning:** `run_robot.py --n-action-steps 25` executes 25 steps of each predicted chunk
  (pi0.5: 50 steps, ACT: 100) before querying again. Lower values react faster and query the model more often.
- **ACT gets the flap as numbers.** In the first dataset the four recording sets open the flaps in different
  orders (back or front first, then right or left third), so a model without the prompt cannot know which
  flap comes next. `scripts/add_flap_onehot.py` adds `observation.environment_state` =
  [back, front, right, left] with a 1 for the current flap. At run time `FlapPolicy` converts
  the prompt into these numbers, so the scripts and `--order` work the same for both models.
