# JuvenileSong_pipeline
## Preprocessing Steps
```
python main.py INPUT_folder
```
1. dat2wav *matlab/electro_gui*
2. combine: gather around 25 bouts in one file
![raw data](https://github.com/AFurryBear/JuvenileSong_pipeline/blob/main/img/Before%20filtering.jpg)
3. filter: bandfilter applied
![filtered](https://github.com/AFurryBear/JuvenileSong_pipeline/blob/main/img/After%20filtering.jpg)
4. denoise
![denoised](https://github.com/AFurryBear/JuvenileSong_pipeline/blob/main/img/After%20Denoising.jpg)
## Data Labeling with DAS
```
conda activate DAS
das gui
```
4 TCN block used.
feedback loop used.

## Postprocessing Steps: UMAP
```
python umap_yirong.py INPUT_folder ‘wav’
```
use trained model (with data from 7 days) to predict data on bigger time-range.
UMAP visualize predict annotation, to check how's it performance.
https://github.com/AFurryBear/syllable_modify
