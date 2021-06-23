# BoneCementInjectionPlanning

## What is this?
This repository provides the code for running BICEPS (Bone Injection of CEment Planning System). BICEPS takes a CT image of a patient with one or more Vertebral Compression Fractures and generates an estimation of the healthy state of the patients spine to enable treatment planning. The methodology was first introduced and is described in [Patient-specific virtual spine straightening and vertebra inpainting: An automatic framework for osteoplasty planning](https://arxiv.org/abs/2103.07279).

## Installation:
The code has been implemented using **Python 3.7**. To install the necessary packages for this framework run:
```
pip install -r requirements.txt
```

## How does this work?

To run this framework the user needs to specify two inputs, the patient directory and the vertebra fracture id. For example:
```
python main.py --patient_dir ./patient01/ --fracture 22
```
The patient directory must include a sub-directory with the fractured scan; this can be named arbitrarily but should include the chars 'ct' in it. Additionally, the directory can also include scans of the pre-fractured, healthy, state and post-operative state (again the scans in these sub-directories must include 'ct' in them). The sub-directories are named with the date of the scan. Then the framework will include these in the analysis by adding either ``` --healthy ``` or ``` --post_op ``` as arguments when running main. An example of the structure is as follows:

```
--patient01
  --02052016
    --ct_scan.nii
  --04122019
    --ct_scan.nii
  --20122019
    --ct_scan.nii
    --mask_scan.nii
```

The vertebra - label correspondence is as follows:

| Vertebra | Label |
| ----------- | ---------------:|
| C1 | 1 |
| C2 | 2 |
| ... | ... |
| C7 | 7 |
| T1 | 8 |
| ... | ... |
| T12 | 19 |
| L1 | 20 |
| ... | ... |
| L5 | 24 |

Additional optional arguments the user may provide are:

```
height_scale: _to be described_
visualize: If set the - caution can be expensive to visualize all steps' outputs
save: If set the intermediate scans are stored
```

## How to cite?
If you use this repo for your research, please cite us in your work by:

```
@article{bukas2021patient,
  title={Patient-specific virtual spine straightening and vertebra inpainting: An automatic framework for osteoplasty planning},
  author={Bukas, Christina and Jian, Bailiang and Venegas, Luis F Rodriguez and De Benetti, Francesca and Ruehling, Sebastian and Sekuboyina, Anjany and Gempt, Jens and Kirschke, Jan S and Piraud, Marie and Oberreuter, Johannes and others},
  journal={arXiv preprint arXiv:2103.07279},
  year={2021}
}
```
