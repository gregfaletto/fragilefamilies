# fragilefamilies

## Summary

Cleaned data and created principal components regression and lasso-penalized linear regression models to predict three different continuous outcome variables in a training set with only 2,121 observations, nearly 13,000 covariates, and a great deal of missing data. All three of my models finished in the top 40% of entries and outperformed a baseline null model on the test set.

After pre-processing the data, my code trainsed lasso-penalized linear regression models and well as principal components linear regression models on each of the three response variables, as measured at age 15:

* **GPA** 

* **"grit"**: a measure of perserverance--see http://www.fragilefamilieschallenge.org/grit/ for details.

* **"material hardship"**: a measure of extreme poverty--see http://www.fragilefamilieschallenge.org/material-hardship/ for details.

I used cross-validation to find the best lasso-penalized model as well as the best PCR model. Final, I compared the root-mean-squared error (RMSE) on the cross-validation sets for the best lasso-penalized model to the cross-validation RMSE on the best PCR model to choose the final model for each variable. The final selections were as follows:

* **GPA model:** Lasso-penalized linear regression with 57 variables.

* **Grit model:** Principal components linear regression with 1 (!) component.

* **Material hardship model:** Lasso-penalized linear regression with 57 variables.

Thank you to Stephen McKay AKA the_Brit, whose code 'FFC-simple-R-code.R' I
used to get started, Viola Mocz (vmocz) & Sonia Hashim (shashim), whose code
FeatEngineering.R I used, and hty, whose code 'COS424_HW2_imputation_Rcode.R' I
used. Those files were taken from the Fragile Familes github, at https://github.com/fragilefamilieschallenge.

## Files

Here are descriptions of each of the R scripts in this repository:

* **ffc2.R:** My code for my final entry which loads the raw data, pre-processes it, and trains models, as described above.