#ifndef LOG_GAUSSIAN_PROBABILTIY
#define LOG_GAUSSIAN_PROBABILTIY

template<typename STIFF>
struct log_gaussian_probability {
    //The inversed covariance
    typename STIFF::PlainObject cov_inv;
    //The factor in front of the exponential (logarithmic)
    double factor;

    /**
     * Calculates the inverse covariance and the factor of the gaussian
     * @param cov  The covariance of the Gaussian distribution
     */
    log_gaussian_probability(const Eigen::MatrixBase<STIFF> &cov) : cov_inv(cov.inverse()) {
        factor = log(1. / (pow(2 * M_PI, STIFF::RowsAtCompileTime / 2.) * sqrt(abs(cov.determinant()))));
    }

    /**
     * Calculates the logarithmic probability of  (a-b)
     * @tparam Derived The type of a and b
     * @param a minuend
     * @param b subtrahend
     * @return log(N((a-b),cov))
     */
    template<typename DerivedA, typename DerivedB>
    double operator()(const DerivedA &a, const DerivedB &b) {
        auto diff = (a - b).eval();
        return factor - 0.5 * (diff.transpose() * cov_inv * diff)(0);
    }
};

#endif //LOG_GAUSSIAN_PROBABILTIY