# MHST-Net
This an official Pytorch implementation of our paper ["MHST: Multiscale Head Selection Transformer for Hyperspectral and LiDAR Classification"](https://ieeexplore.ieee.org/document/10438852).The specific details of the model are as follows.
![MHST-Net](./figure/MHST-Net.jpg)
****
# Datasets
- [The Houston2013 dataset](https://hyperspectral.ee.uh.edu/?page_id=459)
includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), collected by the National Center for Airborne Laser Mapping (NCALM) using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The HSI comprise 144 spectral bands covering a wavelength range from 0.38 to 1.05 µm while LiDAR data are provided for a single band. Both the HSI and LiDAR data share dimensions of 349 × 1905 pixels with a spatial resolution of 2.5 m. The dataset contains 15 categories, with a total of 15,029 real samples available. 
- [The Trento dataset](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
comprises HSI and LiDAR data obtained from southern Trento, Italy. The HSI was collected by an AISA Eagle sensor, consisting of 63 spectral bands with a wavelength range from 0.42 to 0.99 µm. LiDAR data with 1 raster were acquired by the Optech ALTM 3100EA sensor. The scene consists of 166 × 600 pixels, with a spatial resolution of 1 m. This dataset contains 6 land cover types
with a total of 30,214 real samples.
- [The MUUFL dataset](https://github.com/GatorSense/MUUFLGulfport)
was acquired in November 2010 over the area of the campus of University of Southern Mississippi Gulf Park, Long Beach Mississippi, USA. The HSI data was gathered using the ITRES Research Limited (ITRES) Compact Airborne Spectral Imager (CASI-1500) sensor, initially comprising 72 bands. Due to excessive noise, the first and last eight spectral bands were removed, resulting in a total of 64 available spectral channels ranging from 0.38 to 1.05 µm. LiDAR data was captured by an ALTM sensor, containing two rasters with a wavelength of 1.06 µm. The dataset consists of 53,687 groundtruth pixels, encompassing 11 different land-cover classes. 
****
# Train MHST-Net
``` 
python demo.py
``` 
****
# Results
All the results presented here are referenced from the original paper.
| Dataset | OA (%) | AA (%) | Kappa (%) |
| :----: |:------:|:------:|:---------:|
| Houston  | 96.19  | 96.80  |   95.88   |
| Trento  | 99.45  | 99.09  |   99.26   |
| MUUFL  | 88.71  | 90.08  |   85.32   |
****
# Citation
If you find this paper useful, please cite:
``` 
@ARTICLE{10438852,
  author={Kang Ni, Duo Wang, Zhizhong Zheng, Peng Wang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={MHST: Multiscale Head Selection Transformer for Hyperspectral and LiDAR Classification}, 
  year={2024},
  volume={17},
  number={},
  pages={5470-5483},
  doi={10.1109/JSTARS.2024.3366614}
}
```
****
# Contact
Duo Wang: [njwangduo@163.com](njwangduo@163.com)
