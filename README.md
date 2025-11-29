# Bivariate-Student-t-distribution
Study the properties of the bivariate [Student's t distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution) through simulation

`python xbivariate_student_t.py` gives

```
Parameters used:
  nu        = 5.0
  sigma_x   = 1.0
  sigma_y   = 2.0
  rho       = 0.5
  n_samples = 300000
  band h    = 0.05
  x0_grid   = [-2. -1.  0.  1.  2.]

Sample covariance matrix of (X, Y) for Student t:
[[0.99631851 1.00310255]
 [1.00310255 4.00489814]]

Student t: comparison of empirical and theoretical Var(Y | X = x):
  x0  n_points  empirical_var  theoretical_var  ratio_empirical/theoretical
-2.0      1219       5.147096             5.25                     0.980399
-1.0      6249       3.051619             3.00                     1.017206
 0.0     14674       2.213105             2.25                     0.983602
 1.0      6104       2.941086             3.00                     0.980362
 2.0      1113       5.225092             5.25                     0.995256

Sample covariance matrix of (X, Y) for normal:
[[1.00009646 0.99924968]
 [0.99924968 3.99636017]]

Normal: comparison of empirical and theoretical Var(Y | X = x):
  x0  n_points  empirical_var  theoretical_var  ratio_empirical/theoretical
-2.0      1623       2.934992              3.0                     0.978331
-1.0      7404       2.973429              3.0                     0.991143
 0.0     12099       3.027863              3.0                     1.009288
 1.0      7179       3.010133              3.0                     1.003378
 2.0      1576       3.000733              3.0                     1.000244
```

If (x,y) data has the bivariate Student's t distribution, the dependence of y on x is linear, as
with the normal distribution, but the conditional variance of y given x rises with the deviation of x from its mean,
as shown below. For the normal distribution the conditional variance is constant.<br>
![Alt text](/conditional_mean.png)
<br>Here is a dot plot:<br>
![Alt text](/dots.png)
