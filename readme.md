# ShuffleMixer Log

## Initial Commit

### Current Issues

**Training is too slow**

Would take about 5 days to run on jej1g19's desktop, at about a minute per epoch.

Most of this time is a flat cost - it doesn't vary between model sizes and upsampling types.

The training steps finish in bursts, the length of which change as the number of data loader worker threads changes. Therefore, the issue seems to be with the data

The only reasonable solution seems to be creating a 2X and 4X downscaled datasets so this step isn't done every iteration. This would very likely require writing a custom data loader class.


**VRAM Issues**

Training for 4X upsampling runs out of VRAM on jej1g19's desktop unless batch size is reduced (so that it no longer matches the original paper).

Training on Iridis5 would solve this, however currently it either takes almost for hours for one or debug output isn't working with slurm. 

Possible solutions:

- Fixing the data loading issues will help even more on Iridis5 as scratch access is likely very slow
- Add more debug output (remove reliance on tqdm for timing information)
- Parralelize to make the most of the four GPUs available

### Contingency

If neither of the above issues can be fixed, the model can run on jej1g19's desktop overnight for a bit over a week - we would have something trained.

### Notes

From what training has been done, the model seems to be converging correctly.

Report exists but is poorly structured and in note form. 

If the above issues can be sorted, and so training sped up, other aspects of the original paper could be reproduced - such as checking the effect of each part of the model (by removing them/changing to default).



## Update 17/04

### Custom DataLoader

Custom data loader about halves the train time on jej1g19's desktop, yet to be tested on Iridis. Batch sizes are small and the bottleneck is definitely loading full images. It seems that parralelizing across GPUs would not help in this situation. 

The only real solution would be to change the images to bitmaps, so that the cropped sections could be streamed from the disk - rather than loading the whole image and then cropping.

### Notes

At current, it seems to make sense to abandon the batch size used by the paper, and use the largest one that can be used for the standard model and 4X scale. This is because the output quality is more move obvious at the four times scale, for the standard model; it is the most impressive result of the original paper, so should be the reproduction target.

TQDM is no longer used - timing print statements have been added which should improve logging on Iridis.

Experimented with fft2 loss and its effects - loss curves are much better using the new version (the shape/gradient). Need to research what the paper actually implies.