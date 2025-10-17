# PredNet / GAN / Circuit Toolkit (three-project bundle)



This Project only uses CPU. 



## Debugging: 



Look for missing weight files in `BOS-in-Video-Prediction`, see README file in that directory.

Check Directory path in all files



This repo bundles three directories: `prednet\_gan\_opt`, `BOS-in-Video-Prediction`, and `circuit\_toolkit`.

It uses \*\*two Conda environments\*\*:



- \*\*border\_ownership\*\* — PredNet / analysis side

- \*\*ct\_cpupy310\*\*     — GAN optimizer / circuit\_toolkit side



## Setup

Create the environments from specs in `envs/`:

```bash

conda env create -f envs/environment.border\_ownership.yml

conda env create -f envs/environment.ct\_cpupy310.yml

