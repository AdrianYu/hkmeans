# hkmeans
Hierarchical K-Means using kmeans++ initialization algorithm and Elkan training algorithm.

A single header for K-Means and Hierarchical K-Means. For kmeans, I use kmeans++ initialization algorithm and Elkan training algorithm to accelerate converage. To parallelise some of the procedures, I use OpenMP. I also use Eigen and some of the C++11 specifications. So to compile, you must have [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) in your system and include it. Also, don't forget to add -fopenmp and -std=c++11 to your compiling flags.

The main differences between this module and the [VLFeat Hierarchical integer k-means](http://www.vlfeat.org/overview/hikm.html) module are the following. First, this module doesn't constrain itself to integer data and using kmeans++ to initialize. Second, this module can exploit all the cpu cores.

Please feel free to ask any questions related to this module. Since this is not a completed module yet, any suggestions are more than welcome. You can alse email me and the address is [adrianandyu@gmail.com](mailto:adrianandyu@gmail.com)

# Reference
Arthur, D., & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. Symposium on Discrete Algorithms.

Elkan, C. (2003). Using the Triangle Inequality to Accelerate k-Means. International Conference on Machine Learning.
 

