# ridge-model-matlab-baseline

Minimal dataset and code to run the original MATLAB implementation of the ridge regression encoding model. 

This code is tested on __MATLAB R2021b__ (and is known to __NOT__ work on R2022a).

Access the example dataset [here](https://drive.google.com/file/d/1ytS-GJyO_08lYxg9EYDKpBzDChS8Vr1z/view?usp=sharing) and unzip it. 

Edit `line 5` of `main.m` with the with the path to the downloaded data, then run `main.m` for an example of how the encoding model is fit. This is the only line that should need modification for the code to run.
There is a lot of refactoring we can do here, especially in `ridgeModel_returnDesignMatrix.m`, but I figure we can save that work for the Python implementation.

Todo:
- [ ] add code to generate plots of cvR^2^ and beta kernels
- [ ] tests to compare design matrices and model predictions
- [ ] coefficient of determination calculation

Feel free to slack Max Melin if anything is unclear or not running as expected.
