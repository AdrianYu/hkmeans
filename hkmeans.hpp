#pragma once

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <ctime>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <functional>

// disable Eigen's own parallelization
//#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Eigen>

#include "omp.h"

namespace adrianyu {

class KahanSumationEigen
{
    /*
        Kahan Summation Algorithm, see ref.:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    */
public:
    /*
        if the first constructor is used, then the funtion init()
        must be called before using other functions.
    */
    explicit KahanSumationEigen(void) {}
    explicit KahanSumationEigen(const int dim) {
        init(dim);
    }
    void init(const int dim) {
        sum = Eigen::VectorXd::Zero(dim);
        c = Eigen::VectorXd::Zero(dim);
    }
    const KahanSumationEigen& add(const Eigen::VectorXd & val) {
        Eigen::VectorXd y(val - c);
        Eigen::VectorXd t(sum + y);
        c = (t - sum) - y;
        sum = t;
        return *this;
    }

    Eigen::VectorXd get(void) const {
        return sum;
    }

    KahanSumationEigen & operator+=(const KahanSumationEigen & rhs) {
        this->add(rhs.sum);
        this->add(rhs.c);
        return *this;
    }
    friend KahanSumationEigen operator+(KahanSumationEigen lhs, const KahanSumationEigen &rhs) {
        lhs += rhs;
        return lhs;
    }

protected:
    Eigen::VectorXd sum;
    Eigen::VectorXd c;
};

template <class DataType>
class kmeans
{
    /*
        K-Means using kmeans++ initialization algorithm and Elkan training algorithm.
        ref.:
            Arthur, D., & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. Symposium on Discrete Algorithms.
            Elkan, C. (2003). Using the Triangle Inequality to Accelerate k-Means. International Conference on Machine Learning.
    */
public:
    /*
        init kmeans using data and pre-specified k
        params:
            data: training data used to init the centers, must be column-major stored
            k: specify the number of centers
            return: fail on negative, otherwise return the true number of centers initialized.
    */
    int init(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data, int k) {
        std::cerr << "in kmeans init..." << std::endl;
        this->k = kmeanspp_init(data, k, centers);
        max_iter = 2000;

        half_inter_dist = Eigen::MatrixXd::Zero(this->k, this->k);
        #pragma omp parallel for
        for (int i = 0; i < this->k - 1; ++i) {
            for (int j = i + 1; j < this->k; ++j) {
                half_inter_dist(i, j) = (centers.col(i) - centers.col(j)).norm() / 2;
                half_inter_dist(j, i) = half_inter_dist(i, j);
            }
        }
        return this->k;
    }

    /*
        train kmeans using data by Elkan algorithm
        ref: Using the Triangle Inequality to Accelerate k-Means by Charles Elkan
        params:
            data: training data used to init the centers, must be column-major stored
            k: specify the number of centers
            return: fail on negative, otherwise return the true number of centers initialized.
    */
    int train(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data) {
        // init all params
        const int data_num = data.cols();
        const int data_dim = data.rows();
        // lower bound
        //Eigen::MatrixXd l_b = -1 * Eigen::MatrixXd::Ones(k, data_num);
        Eigen::MatrixXd l_b = Eigen::MatrixXd::Zero(k, data_num);
        // upper bound
        Eigen::VectorXd u_b = Eigen::VectorXd::Zero(data_num);
        // class assignment
        Eigen::VectorXi c_x = Eigen::VectorXi::Zero(data_num);
        // do the real assignment
        #pragma omp parallel for
        for (int i = 0; i < data_num; ++i) {
            //l_b.col(i).noalias() = -1 * Eigen::VectorXd::Ones(k);
            l_b(0, i) = (data.col(i).cast<double>() - centers.col(0)).norm();
            int class_s = 0;
            double closest_dist = l_b(0, i);
            for (int j = 1; j < k; ++j) {
                if (half_inter_dist(j, class_s) < closest_dist) {
                    l_b(j, i) = (data.col(i).cast<double>() - centers.col(j)).norm();
                    if (l_b(j, i) < closest_dist) {
                        closest_dist = l_b(j, i);
                        class_s = j;
                    }
                }
            }
            c_x[i] = class_s;
            u_b[i] = closest_dist;
        }

        // the algorithm, repeat until convergence
        size_t iter = 0;
        Eigen::VectorXd s_c = Eigen::VectorXd::Zero(k);
        //Eigen::MatrixXd inter_dist = Eigen::MatrixXd::Zero(k, k);
        std::vector<bool> r_x(data_num, false);
        int done = 0;
        std::cerr << "begin to iterate" << std::endl;
        std::clock_t done_clk = std::clock();
        while (iter < max_iter && 0 == done) {
            std::clock_t beg_clk = std::clock();
            std::cerr << "curr iter: " << iter << std::endl;
            done = 1;

            // step 4 update centers in parallel
            std::vector< std::vector<KahanSumationEigen> > m_c_th;
            std::vector< std::vector<size_t> > c_n_th;
            int nthreads;
            #pragma omp parallel
            {
                #pragma omp single
                {
                    nthreads = omp_get_num_threads();
                    m_c_th.resize(nthreads, std::vector<KahanSumationEigen>(k, KahanSumationEigen(data_dim)));
                    c_n_th.resize(nthreads, std::vector<size_t>(k, 0));
                }
                // the data number may not be divisible by nthreads
                size_t part_size = data_num / nthreads;
                if (data_num % nthreads) {
                    part_size++;
                }
                const int ithread = omp_get_thread_num();
                const size_t part_start = ithread * part_size;
                for (size_t i = part_start; i < part_start + part_size; ++i) {
                    if (i < data_num) {
                        size_t c = c_x[i];
                        c_n_th[ithread][c]++;
                        m_c_th[ithread][c].add(data.col(i).cast<double>());
                    }
                }
            }
            // do the reduction
            std::vector<KahanSumationEigen> m_c(k, KahanSumationEigen(data_dim));
            std::vector<size_t> c_n(k, 0);
            for (int th_s = 0; th_s < nthreads; ++th_s) {
                for (int i = 0; i < k; ++i) {
                    c_n[i] += c_n_th[th_s][i];
                    m_c[i] += m_c_th[th_s][i];
                }
            }

            // normalize, and get the real centers
            // step 5
            Eigen::MatrixXd c_m_c = Eigen::MatrixXd::Zero(k, k);
            #pragma omp parallel for
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    c_m_c(i, j) = (centers.col(i) - m_c[j].get() / c_n[j]).norm();
                }
            }
            for (int c = 0; c < k; ++c) {
                #pragma omp parallel for
                for (int x = 0; x < data_num; ++x) {
                    l_b(c, x) = std::max(l_b(c, x) - c_m_c(c, c), 0.0);
                }
            }
            // step 6
            #pragma omp parallel for
            for (int x = 0; x < data_num; ++x) {
                u_b[x] += c_m_c(c_x[x], c_x[x]);
                r_x[x] = true;
            }
            // step 7
            #pragma omp parallel for
            for (int i = 0; i < k; ++i) {
                centers.col(i).noalias() = m_c[i].get() / c_n[i];
            }

            // step 1. update min half inter class distances
            #pragma omp parallel for
            for (int i = 0; i < k; ++i) {
                Eigen::VectorXd dists = Eigen::VectorXd::Zero(k - 1);
                for (int j = 0, m = 0; j < k; ++j) {
                    if (j != i) {
                        dists[m] = (centers.col(i) - centers.col(j)).norm() / 2;
                        half_inter_dist(j, i) = dists[m];
                        m++;
                    }
                }
                s_c[i] = dists.minCoeff();
            }
            //std::cout << "half_inter_dist: " << half_inter_dist << std::endl;
            //std::cout << "s_c: " << s_c << std::endl;
            // step 2 & 3
            std::vector<bool> need_step3(data_num, true);
            // step 2
            #pragma omp parallel for
            for (int x = 0; x < data_num; ++x) {
                if (u_b[x] <= s_c[c_x[x]]) {
                    need_step3[x] = false;
                }
            }
            for (int c = 0; c < k; ++c) {
                #pragma omp parallel for
                for (int x = 0; x < data_num; ++x) {
                    //std::cerr << x << std::endl;
                    if (need_step3[x] && u_b[x] > half_inter_dist(c_x[x], c) \
                        && c != c_x[x] && u_b[x] > l_b(c, x)) {
                        double dist_org;
                        double dist_new;
                        // step 3a
                        if (r_x[x]) {
                            dist_org = (data.col(x).cast<double>() - centers.col(c_x[x])).norm();
                            u_b[x] = dist_org;
                            r_x[x] = false;
                        }
                        else {
                            dist_org = u_b[x];
                        }
                        // step 3b
                        if (dist_org > l_b(c, x) || dist_org > half_inter_dist(c_x[x], c)) {
                            dist_new = (data.col(x).cast<double>() - centers.col(c)).norm();
                            l_b(c, x) = dist_new;
                            if (dist_new < dist_org) {
                                c_x[x] = c;
                                u_b[x] = dist_new;
                                #pragma omp atomic
                                done *= 0; // for gcc, done = 0 should compile too.
                                continue;
                            }
                        }
                    }
                }
            }
            iter++;
            done_clk = std::clock();
            std::cerr << "time: "
                << static_cast<double>(done_clk - beg_clk) * 1000.0 / CLOCKS_PER_SEC
                << std::endl;
        }
        return 0;
    }
    
    /*
        Given one sample, return the class index starting from 0
        params:
            @sample: data sample vector, column-major
            @closest_dist: the distance between sample and its belonging class
            @return: class index (starting from 0)
    */
    int get_class(const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &sample,
        double &closest_dist) {
        Eigen::VectorXd dists(k);
        dists[0] = (sample.cast<double>() - centers.col(0)).norm();
        int class_s = 0;
        closest_dist = dists[0];
        for (int j = 1; j < k; ++j) {
            if (half_inter_dist(j, class_s) < closest_dist) {
                dists[j] = (sample.cast<double>() - centers.col(j)).norm();
                if (dists[j] < closest_dist) {
                    closest_dist = dists[j];
                    class_s = j;
                }
            }
        }
        return class_s;
    }
    int get_class(const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &sample) {
        double cdist;
        return get_class(sample, cdist);
    }

    /*
        Given some samples, return the class indices
        params:
            @data: data sample vectors, column-major
            @return: class indices (starting from 0)
    */
    void get_class_indices(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data, \
        std::vector<size_t> &c_indices) {
        c_indices.clear();
        const int data_num = data.cols();
        const int data_dim = data.rows();
        c_indices.resize(data_num);
        #pragma omp parallel for
        for (int i = 0; i < data_num; ++i) {
            c_indices[i] = get_class(data.col(i));
        }
    }

    /*
        return the centers
    */
    Eigen::MatrixXd get(void) {
        return centers;
    }
    
protected:
    static const double EPSILON;
    size_t max_iter;
    int k; // center number
    Eigen::MatrixXd centers;
    Eigen::MatrixXd half_inter_dist;

    /*
        Given a data point, compute the distances from all centers
        params:
            @sample: data sample vector, column-major
            @return: None
    */
    void get_dists(const Eigen::Matrix<DataType, Eigen::Dynamic, 1> & sample,
        Eigen::VectorXd &dists) {
        dists = -1 * Eigen::VectorXd::Ones(k);
        for (int i = 0; i < k; ++i) {
            // Euclidean distance
            dists[i] = (sample.cast<double>() - centers.col(i)).norm();
        }
    }

    /*
        Given the data and the number of centers k, 
        initialize all the k centers using kmeans++ algorithm.
        params:
            data: training data used to init the centers, must be column-major stored
            k: specify the number of centers
            centers: the initialized centers, also column-majored. Need not to be pre-allocated.
            return: fail on negative, otherwise return the true number of centers initialized.
    */
    static int kmeanspp_init(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data,
        const int k, Eigen::MatrixXd& centers) {
        if (k < 0) {
            std::cerr << "kmeans init failed: number of centers must be greater than 0, ["
                << k << "] given" << std::endl;
            return -1;
        }
        const int data_num = data.cols();
        const int data_dim = data.rows();
        centers.resize(data_dim, k);

        std::uniform_real_distribution<double> rnddist(0.0, 1.0);
        unsigned seed = static_cast<unsigned>(\
            std::chrono::system_clock::now().time_since_epoch().count());
        std::mt19937_64 rndgen(seed);

        // randomly choose a data point as the first center
        int idx = static_cast<int>(std::floor(\
                static_cast<double>(data_num) * rnddist(rndgen)));
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= data_num) {
            idx = data_num - 1;
        }
        centers.col(0).noalias() = data.col(idx).cast<double>();

        const double rnd_number = min(k * 1000, data_num);
        const double prop = rnd_number / static_cast<double>(data_num);

        // choose the rest of the centers based on their distances from the closest center.
        Eigen::VectorXd distances(data_num);
        Eigen::VectorXd maxdists(data_num);
        int real_k = k;
        // compare to the real training procedure, this is negligible?
        // so no sub-sampling is needed.
        for (int i = 1; i < k; ++i) {
            
            /*seed = static_cast<unsigned>(\
                std::chrono::system_clock::now().time_since_epoch().count());
            rndgen.seed(seed);
            std::vector<bool> not_compute(data_num, false);
            for (size_t j = 0; j < not_compute.size(); ++j) {
                not_compute[j] = (rnddist(rndgen) > prop);
            }*/

            distances = Eigen::VectorXd::Zero(data_num);
            maxdists = Eigen::VectorXd::Zero(data_num);
            #pragma omp parallel for
            for (int j = 0; j < data_num; ++j) {
                // sub-sample
                //if (not_compute[j]) {
                //    continue;
                //}

                Eigen::VectorXd dists_(i);
                for (int m = 0; m < i; ++m) {
                    dists_[m] = (data.col(j).cast<double>() - centers.col(m)).squaredNorm();
                    // using l1-norm to replace l2-norm
                    //dists_[m] = (data.col(j).cast<double>() - centers.col(m)).cwiseAbs().sum();
                }
                //int x, y;
                //maxdists[j] = dists_.maxCoeff(&x, &y);
                maxdists[j] = dists_.maxCoeff();
                distances[j] = dists_.minCoeff();
            }
            // just in case that there are less distinct data samples than centers.
            if (maxdists.maxCoeff() < EPSILON) {
                real_k = i;
                break;
            }
            // get the next center
            // choose the point using a weight probability distribution
            // where a point x is chosen with probability proportional to squared distance.
            double dist_sum = distances.sum();
            do {
                double posf = dist_sum * rnddist(rndgen);
                double cpos = 0;
                idx = data_num - 1;
                for (int j = 0; j < distances.size(); ++j) {
                    cpos += distances[j];
                    if (cpos > posf) {
                        idx = j;
                        break;
                    }
                }
            } while (distances[idx] < EPSILON);
            centers.col(i).noalias() = data.col(idx).cast<double>();
        }
        return real_k;
    }
};

template <class DataType> const double kmeans<DataType>::EPSILON = 1e-10;

template <class DataType>
class hkmeans
{
    /*
        Hierarchical K-Means
    */
public:

    explicit hkmeans()
    {
        hk_trees = NULL;
    }

    ~hkmeans() {
        destroy();
    }

    /*
        train hierarchical kmeans
        params:
            data: training data used to init the centers, must be column-major stored
            k: specify the number of centers
            height: the maximum height of the tree
            return: fail on false, otherwise return true
    */
    bool train(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data, \
        const int k, const int height) {
        if (height < 1) {
            return false;
        }

        this->k = k;
        this->height = height;
        // first destroy previous model
        destroy();
        
        // allocate first node
        hk_trees = new KMeansNode;
        // do the real training
        return train_tree(data, hk_trees, height);
    }

protected:
    int k;
    int height;
    // 2^15 should be of enough precision for most applications: 1GB memory
    // can store more than 4 million numbers.
    //Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> _train_data;

    struct KMeansNode
    {
        /*
            tree node struct
        */
        kmeans<DataType> node_km;
        std::vector<KMeansNode *> children_nodes;
    };
    KMeansNode *hk_trees;

    /*
        train a hierarchical kmeans tree recursively
        params:
            data: training data used to init the centers, must be column-major stored
            root: specify the current root of the tree
            height: the height of the current root node
            return: fail on false, otherwise return true
    */
    bool train_tree(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &data, \
        KMeansNode *root, const int height) {
        const int data_num = data.cols();
        const int data_dim = data.rows();
        // init and train
        int real_k = root->node_km.init(data, k);
        //std::cout << std::endl << data << std::endl << std::endl;
        root->node_km.train(data);
        //std::cout << std::endl << root->node_km.centers << std::endl;
        if (height > 1) {
            root->children_nodes.resize(real_k, NULL);
            std::vector<size_t> indices;
            root->node_km.get_class_indices(data, indices);
            for (int i = 0; i < real_k; ++i) {
                root->children_nodes[i] = new KMeansNode;

                // TODO: use the following, and not copy to sub-class data data_p
                //std::vector<bool> idx_f(indices.size());
                //auto eq_i = std::bind(std::equal_to<size_t>(), std::placeholders::_1, i);
                //std::transform(indices.begin(), indices.end(), idx_f.begin(), eq_i);

                // get data proportion
                size_t c_count = std::count(indices.begin(), indices.end(), i);
                Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> data_p = \
                    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::Zero(data_dim, c_count);
                for (int j = 0, c = 0; j < data_num && c < c_count; ++j) {
                    if (i == indices[j]) {
                        data_p.col(c).noalias() = data.col(j);
                        ++c;
                    }
                }
                // do the training recursively.
                bool ret = train_tree(data_p, root->children_nodes[i], height - 1);
            }
        }
        else {
            root->children_nodes.clear();
        }
        return true;
    }
    
    /*
        destroy the hierarchical kmeans tree recursively
    */
    void destroy(void) {
        std::queue<KMeansNode *> nodes;
        nodes.push(hk_trees);
        do {
            KMeansNode *anode = nodes.front();
            nodes.pop();
            if (anode) {
                for (size_t i = 0; i < anode->children_nodes.size(); ++i) {
                    nodes.push(anode->children_nodes[i]);
                }
                anode->children_nodes.clear();
                delete anode;
            }
        } while (!nodes.empty());
    }
};

}
