Implementation of Optimization of OmniMotion, a tracking algorithm, my Bachelor Thesis, done at University of Bern.

Implementation for an improved version of the paper [Tracking Everything Everywhere All at Once]((https://omnimotion.github.io/)), ICCV 2023.

## Installation
The code is tested with `python=3.8` and `torch=1.10.0+cu111`.
```
cd quasi-omnifasttrack/
conda create -n omnimotion python=3.8
conda activate omnimotion
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## Training
With processed input data, run the following command to start training:
```
python3 train_depth.py --config configs/default.txt --data_dir {sequence directory} --num_pts 150
```

## Troubleshooting
Extracted from OmniMotion's README

- The training code utilizes approximately 22GB of CUDA memory. If you encounter CUDA out of memory errors, 
  you may consider reducing the number of sampled points `num_pts` and the chunk size `chunk_size`.
- Due to the highly non-convex nature of the underlying optimization problem, we observe that the optimization process 
  can be sensitive to initialization for certain difficult videos. If you notice significant inaccuracies in surface
  orderings (by examining the pseudo depth maps) persist after 40k steps, 
  it is very likely that training won't recover from that. You may consider restarting the training with a 
  different `loader_seed` to change the initialization. 
  If surfaces are incorrectly put at the nearest depth planes (which are not supposed to be the closest), 
  we found using `mask_near` to disable near samples in the beginning of the training could help in some cases.  
- Another common failure we noticed is that instead of creating a single object in the canonical space with
  correct motion, the method creates duplicated objects in the canonical space with short-ranged motion for each.
  This has to do with both that the input correspondences on the object being sparse and short-ranged, 
  and the optimization being stuck at local minima. This issue may be alleviated with better and longer-range input correspondences 
  such as from [TAPIR](https://deepmind-tapir.github.io/) and [CoTracker](https://co-tracker.github.io/). 
  Alternatively, you may consider adjusting `loader_seed` or the learning rates.
