# MOTT Training

## CrowdHuman Pre-training

The command use several configuration files specified after `with` directive.

* default (all config): `cfgs/train.yaml`
* `crowdhuman`: `cfgs/train_crowdhuman.yaml`
* `deformable`: `cfgs/train_deformable.yaml`
* `tracking`: `cfgs/train_tracking.yaml`
* `model_motr`: `cfgs/train_model_motr.yaml`

You can also overwrite the configuration directly in this command line console, for example `output_dir` is overwritten
here. `nohup` is used for background execution without interruption, the console output will be saved
in `mott_ch_pre.log` file. PyTorch's `distributed.launch` module is used for GPU parallelism.

```bash
nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
    src/train.py with \
    crowdhuman \
    deformable \
    tracking \
    model_motr \
    output_dir=mott_ch_pre >> "mott_ch_pre.log" &
```

### MOT17 training

Here we use almost the same configuration except the one for MOT17 dataset: `cfgs/train_mot17_crowdhuman.yaml`. We also
need to change the `resume` path to the output checkpoint from the first stage.

```bash
RESUME_CKPT=path/to/the/checkpoint.pth

nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
    src/train.py with \
    mot17_crowdhuman \
    deformable \
    tracking \
    model_motr \
    resume=$RESUME_CKPT \
    output_dir=mott_mot17 >> "$mott_mot17.log" &
```

### Other training procedures

Check the source file at `src/train.py` and the configuration `cfgs/train_crowdhuman.yaml` for all options and you can
create your own customized training command.
