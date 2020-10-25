# Deep SPD on fMRI data

Use the following directory structure:

- src (models, utilities)
- data (data files, preferably in .npy format)
- notebooks (experiments, graphs and tables generators)

<<<<<<< HEAD

Wandb:
https://wandb.ai/feathernox/fmri_deepspd_v3
https://wandb.ai/feathernox/fmri_deepspd_v3/reports/Final-Report--VmlldzoyOTA0MjE
=======
W&B (analogue of Comet.ml):
- https://wandb.ai/feathernox/fmri_deepspd_v3 - all runs
- https://wandb.ai/feathernox/fmri_deepspd_v3/reports/Final-Report--VmlldzoyOTA0MjE - final report

Main script is notebooks/train_multiprocessing.py. There are several hyperparameters which you can sweep over, as the code is based on Hydra framework, such as:
- network - choose name of the file in configs/network folder, e.g. spd_5/spd_10/spd_15/baseline_5/baseline_10/baseline_15
- train.lr - learning rate
- train.batch_size - batch size
- train.max_epochs - number of epochs

E.g. if you want to create sweep over batch sizes, you can run
```python notebooks/train_multiprocessing.py --multiline setup.num_workers=1 train.batch_size=16,32,64```
>>>>>>> 7f3f161cd6338a8f2706b51acd5c6d6126c251e5
