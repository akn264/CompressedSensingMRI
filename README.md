# CompressedSensingMRI
Image processing project in medical imaging -- experiment with compressed sensing (CS) framework for MR image reconstruction.
Essentially modeled an objective function as a denoising problem with a data fidelity term and two regularization parameters (horizontal and vertical finite differences). The solver I implemented was iterative soft thresholding algorithm (ISTA), but there are other iterative algorithms that can be used obviously. Better to use matrix-free methods for image processing. ISTA is real slow in Python, even compared to MATLAB. 

When initially trying to understand the concept of CS I relied heavily on Prof Miki Lustig's lecture material and demos. The data I used were also obtained from his website here: https://people.eecs.berkeley.edu/~mlustig/index.html
