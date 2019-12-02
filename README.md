# Face relighting in Python
A python implementation of [portrait lighting transfer using a mass transport approach](https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/shu_tog2017.pdf).\
3DMM model in original implementation is replaced by [PRNet](https://github.com/YadiraF/PRNet).

# Examples
input img | reference img | relighting img
![img](imgs/portrait_o1.jpg)
![img](imgs/portrait_o2.jpg)
![img](imgs/portrait_o3.jpg)
![img](imgs/portrait_o4.jpg)
![img](imgs/portrait_o5.jpg)
![img](imgs/portrait_o6.jpg)

# Run
```bash
/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r1.jpg: 873x799x3
relight time: 19.50

/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r2.jpg: 950x950x3
relight time: 19.54

/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r3.jpg: 1024x683x3
relight time: 17.38

/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r4.jpg: 2403x1927x3
relight time: 55.92

/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r5.jpg: 1024x744x3
relight time: 18.34

/root/python-portrait-relight/imgs/portrait_s1.jpg: 800x800x3
/root/python-portrait-relight/imgs/portrait_r6.jpg: 981x774x3
relight time: 18.94
```
Most time is spent on PRNet, which rely on dlib to detect human faces. 
Replacing the face detector will dramatically improve the speed.

# Methods

Let input image be I, reference image be R and output image be O.\
Let posI, posR be frontal 3d face position map of img I, R, with shape=[n, 3].\ 
Let colorI, colorR be rgb colors of the reconstructed vertices of img I, R, with shape=[n, 3].\
Let normalI, normalR be normal vectors of the vertices of img I, R, with shape=[n, 3].\
We obtain features fI=[colorI, posI[:,:,:2], nomralI], fR=[colorR, posR[:,:,:2], normalR] of img I, R, with shape=[n, 8].\
Then we determine pdf transfer function t, so that f{t(fI)}=f{fR}, where f{x} is the probability density function of array x.\
t(colorI) is the relighted image of I with R for reference.\
Finally, we apply regrain algorithm for postprocessing.

# Dependency
Process of pdf transfer and regrain are introduced in [Automated colour grading using colour distribution transfer](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694&rep=rep1&type=pdf) by F. Pitie , A. Kokaram and R. Dahyot.
This repository denpends on my [python implementation of this paper](https://github.com/pengbo-learn/python-color-transfer).


# References
[portrait lighting transfer using a mass transport approach](https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/shu_tog2017.pdf) 
by Zhixin Shu, Sunil Hadap, Eli Shechtman, Kalyan Sunkavalli, Sylvain Paris and Dimitris Samaras.\
[Author's matlab implementation](https://github.com/AjayNandoriya/PortraitLightingTransferMTP)


