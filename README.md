# antibiotic_prediction

The input data is available from following zenodo link

https://zenodo.org/record/3464542#.YwXkhnZBxD8

# Steps to reproduce results

1) download the features_gpa_expr_snps.zip and metadata.zip and unzip them.

2) Step one will result in two folders `features_gpa_expr_snps` and `metadata`. `features_gpa_expr_snps` folder will further
consist of 3 folders namely `genexp`, `gpa` and `snps`. On the other hand, `metadata` folder will have `phenotypes.txt` file in it.

3) install ai4water at >= 1.6.

4) run the file ``exp.py`` to run experiments. 

5) Change the `target` in `exp.py` file to run experiments for different antibiotics.

The file `main.py` can be used to build, train and make predictions from a single model.
