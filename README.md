# forecast-video-diffmodels
Code repository for the paper Improving Tropical Cyclone Forecasting With Video Diffusion Models

## Abstract

Tropical cyclone (TC) forecasting is crucial for disaster preparedness and mitigation. While recent deep learning approaches have shown promise, existing methods often treat TC evolution as a series of independent frame-to-frame predictions, limiting their ability to capture long-term dynamics. We present a novel application of video diffusion models for TC forecasting that explicitly models temporal dependencies through additional temporal layers. Our approach enables the model to generate multiple frames simultaneously, better capturing cyclone evolution patterns. We introduce a two-stage training strategy that significantly improves individual-frame quality and performance in low-data regimes. Experimental results show our method outperforms the previous approach of Nath et al. by 19.3\% in MAE, 16.2\% in PSNR, and 36.1\% in SSIM. Most notably, we extend the reliable forecasting horizon from 36 to 50 hours. Through comprehensive evaluation using both traditional metrics and Fréchet Video Distance (FVD), we demonstrate that our approach produces more temporally coherent forecasts while maintaining competitive single-frame quality.

## Environment Setup

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ren-creater/forecast-video-diffmodels.git
   cd forecast-video-diffmodels
   ```

2. Install dependencies:
   - Using Conda (recommended):
     ```bash
     conda env create research_env python=3.10
     conda activate research_env
     cd ./imagen && pip install -r requirements.txt
     ```

3. Evaluation Metrics:
   clone the repository https://github.com/JunyaoHu/common_metrics_on_video_quality,
   place its files and folders in the directory ./imagen/64_FC


## Data Preparation

### 1. Download ERA5 and IR Data
- Download data using the notebooks and python scripts in ./dataproc

### 2. Data Processing
- Run python files of the form *create-dataloaders.py in ./dataproc

## Model Training

Please see ./imagen/64_FC/cx2_dim64.pbs for details.

## Model Evaluation

### 1. Testing on 10-frame Prediction Task
Please see ./imagen/64_FC/cx2_dim64.pbs for details.

### 2. Sample & Training Graph Generation
Please see ./imagen/64_FC/cx2_dim64.pbs for details.

### 3. Evaluating on Long-horizon Prediction Task & Generate Prediciton Animation
Please see ./imagen/64_FC/cx2_gen.sh for details.

## Acknowledgments

- The project is based on https://github.com/p3jitnath/forecast-diffmodels.
- The **VDM model** implementation is adapted from the repository https://github.com/lucidrains/imagen-pytorch.
- Thanks to https://github.com/JunyaoHu/common_metrics_on_video_quality for their implementation of FVD.

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for more details.
