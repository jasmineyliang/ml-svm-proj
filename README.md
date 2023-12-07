# ml-skill-proj2
SVM and Kernel


1. Consider the toy data set f([0; 0];-1); ([2; 2];-1); ([2; 0]; +1)g. Set up the dual problem for the toy data set. Then, solve the dual problem and compute $\alpha^{\star}$ the optimal Lagrange multipliers. (Note that there will be three weights w = [w0;w1;w2] by considering the bias.)
   
2. In a separable case, when a multiplier $\alpha$ i > 0, its corresponding data point (xi; yi) is on the boundary of the optimal separating hyperplane with yi(wTxi) = 1. Show that the inverse is not True. Namely, it is possible that $\alpha$ i = 0 and (xi; yi) is on the boundary satisfying yi(wTxi) = 1.
   
[Hint: Consider a toy data set with two positive examples at ([0,0], +1) and ([1, 0], +1), and one negative example at ([0, 1], -1).] (Note that there will be three weights w = [w0;w1;w2] by considering the bias.)

3. Non-separable Case SVM: In Lecture 8 (page 18), we compare the hard-margin SVM and soft-margin SVM. Prove that the dual problem of soft-margin SVM is almost identical to the hard-margin SVM, except that $\alpha_{i}S$ is are now bounded by C (tradeoff parameter).
   
4. Kernel Function: A function K computes K(xi; xj) = -xiTxj . Is this function a valid kernel function for SVM? Prove or disprove it.

5.  Support Vector Machine for Handwritten Digits Recognition: use the software package LIBSVM http://www.csie.ntu.edu.tw/cjlin/libsvm/ 
Two functions svm train() and svm predict() from LIBSVM library will be used.

(a) Complete the svm with svm_with_diff_c() function. In this function, you are asked to try different values of cost parameter c.

(b) Complete the svm with svm_with_diff_kernel() function. In this function, you are asked to try different kernels (linear, polynomial and radial basis function kernels).

(c)  Summarize observations from (a) and (b) into a short report, report the accuracy result and total support vector number of each model. A briefly analysis based on the results is also needed. For example, how the number of support vectors changes as parameter value changes and why.
