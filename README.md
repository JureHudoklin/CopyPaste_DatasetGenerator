[![basis](https://img.shields.io/badge/based%20on-debidatta/syndata--generation-brightgreen.svg)](https://github.com/debidatta/syndata-generation)
[![basis](https://img.shields.io/badge/based%20on-debidatta/syndata--generation-brightgreen.svg)](https://github.com/a-nau/synthetic-dataset-generation/actions)
[![arxiv](http://img.shields.io/badge/paper-arxiv.2210.09814-B31B1B.svg)](https://arxiv.org/abs/2210.09814)


# Synthetic Dataset Generation

> This repository is a modified (and extended) version
> of [debidatta/syndata-generation](https://github.com/debidatta/syndata-generation) and [a-nau/synthetic-dataset-generation](https://github.com/a-nau/synthetic-dataset-generation/actions)
> The augmentation code is not changed. The code was made more modular All credits to the original authors (also see [Citation](#citation)).
>

Create copy/paste synthetic images for object detection and instance segmentation.

## Set-up
Clone the repository

```shell
git clone https://github.com/JureHudoklin/CopyPaste_DatasetGenerator.git
```

Create a virtual environment (optional) and activate it

```shell
virtualenv -p python3.8 copy_paste_env
source copy_paste_env/bin/activate
```


Install python dependencies

```shell
pip install -r requirements.txt
```

## Configuration

## Run


## Examples


## Citation

If you use this code for scientific research, please consider citing the following two works on which this repository is based on.

### Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection

The original work, including the [code](https://github.com/debidatta/syndata-generation) on which this repository is
built. Thanks a lot to the authors for providing their code!

```latex
@InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

- Paper: [arxiv](https://arxiv.org/abs/1708.01642)
  and [ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf)

### Scrape, Cut, Paste and Learn: Automated Dataset Generation Applied to Parcel Logistics

Our work for which this repository was developed.

```latex
@inproceedings{naumannScrapeCutPasteLearn2022,
  title = {Scrape, Cut, Paste and Learn: Automated Dataset Generation Applied to Parcel Logistics},
  booktitle = {{{IEEE Conference}} on {{Machine Learning}} and Applications} ({{ICMLA}})},
  author = {Naumann, Alexander and Hertlein, Felix and Zhou, Benchun and Dörr, Laura and Furmans, Kai},
  date = {2022},
}
```
