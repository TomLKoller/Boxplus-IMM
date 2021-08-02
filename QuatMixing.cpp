//
// Created by tomlucas on 26.06.20.
//

#include "types/SO3.h"
#include <vector>
#include "GaussianNoiseVector.h"
#include "adekf_viz.h"
#include <Eigen/Eigenvalues>

#define max_iterations 100
#define weighted_mean_epsilon 1e-12
using namespace adekf;

/**
 * Calculate the weighted mean of quaternions
 * @param quats  list of quaternions
 * @param probs the corresponding probabilities
 * @return The weighted mean of quats
 */
SO3d weightedMean(std::vector<SO3d> &quats, std::vector<double> &probs)
{
    SO3d sum = quats[0];
    SO3d old_sum = sum;
    decltype(sum - old_sum) diff_sum = diff_sum.Zero();
    decltype(sum - old_sum) selector_weights = selector_weights.Zero();

    int iterations = 0;
    do
    {
        iterations++;
        old_sum = sum;
        diff_sum = diff_sum.Zero();
        for (int i = 0; i < quats.size(); i++)
        {
            diff_sum = diff_sum + probs[i] * (quats[i] - sum); //.cwiseProduct(selectors[filter_bank[i].second]);
        }
        sum = sum + diff_sum; //.cwiseQuotient(selector_weights);
    } while (iterations <= max_iterations && (sum - old_sum).norm() > weighted_mean_epsilon);
    if (iterations > max_iterations)
        printf("Warning: stopped due to excess of iterations");
    return sum;
}

/**
 * Calculate the weighted covariance of quaternion distributions
 * @param quats The list of mean points of the distributions
 * @param covs The covariances of the distributions
 * @param probs The weights of the distributions
 * @param target The weighted mean of the means
 * @return The weighted covariance sum
 */
Eigen::Matrix3d weightedCovarianceSum(std::vector<SO3d> &quats, std::vector<Eigen::Matrix3d> &covs, std::vector<double> &probs, const SO3d &target)
{
    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d weights = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++)
    {
        if (probs[i] > 0.)
        {
            auto diff = (quats[i] - target).eval();
            auto plus_diff = eval((quats[i] + getDerivator<3>()) - target);
            Eigen::Matrix<double, 3, 3> D(3, 3);
            //Initialise the Jacobian
            for (size_t j = 0; j < 3; ++j)
                D.col(j) = plus_diff[j].v; //write to cols since col major (transposes matrix )
            //Covariance select = selectors[filter_bank[i].second] * selectors[filter_bank[i].second].transpose() * probabilities(i);
            sum += probs[i] * (D.transpose() * (covs[i]) * D + diff * diff.transpose());
            //weights += select;
        }
    }
    assert(sum.determinant() > 0.);
    return sum;
}

/**
 * Calculates the weighted mean of quaternions with naive/improper mixing (in parameter space)
 * @param quats The quaternions to mix
 * @param probs The weights
 * @return Weighted mean of quats
 */
SO3d badWeightedMean(std::vector<SO3d> &quats, std::vector<double> &probs)
{
    Eigen::Vector4d sum = sum.Zero();
    for (int i = 0; i < quats.size(); i++)
    {
        sum += quats[i].coeffs() * probs[i];
    }
    sum = sum.normalized();
    return SO3d(sum.data());
}
/**
 * Calculates a numeric gold standard Variance of quats.
 *
 * Uses a list of uniformly sampled quaternions from the distributions and combines them with the proper mixing
 *
 * @param quats The list of uniformly sampled quaternions
 * @param probs The probabilities of the quaternioms
 * @param target The weighted mean of means of the distributions
 * @return The weighted covariance sum
 */
Eigen::Matrix3d goldWeightedCovarianceSum(std::vector<SO3d> &quats, std::vector<double> &probs, const SO3d &target)
{

    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++)
    {
        if (probs[i] > 0.)
        {
            auto diff = (quats[i] - target).eval();
            sum += probs[i] * (diff * diff.transpose());
        }
    }
    return sum;
}

/**
 * Improper way of the weighted sum of covariances.
 *
 * Uses the naive approach from the original IMM
 * @param quats The list of quaternion means
 * @param covs The covariances
 * @param probs The weights of the quaternions
 * @param target The weighted mean of means
 * @return The naive weighted covariance sum
 */
Eigen::Matrix3d badWeightedCovarianceSum(std::vector<SO3d> &quats, std::vector<Eigen::Matrix3d> &covs, std::vector<double> &probs, const SO3d &target)
{

    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++)
    {
        if (probs[i] > 0.)
        {
            auto diff = (quats[i] - target).eval();
            sum += probs[i] * (covs[i] + diff * diff.transpose());
        }
    }
    return sum;
}

/**
 * Calculates the difference between the bp-weighted mean and the naive weighted mean
 * @param quats List of quaternions
 * @param probs weights
 * @return BPMean-NaiveMean
 */
double diff(std::vector<SO3d> &quats, std::vector<double> &probs)
{
    return (weightedMean(quats, probs) - badWeightedMean(quats, probs)).norm();
}

/**
 * Plots the naive mixing vs the boxplus mixing
 * @param argc command line argument counter
 * @param argv command line arguments
 */
void plot_bad_vs_bp()
{
    std::vector<SO3d> quats;
    std::vector<std::vector<double>> probs;
    std::vector<Eigen::Matrix3d> covs;
    covs.push_back(Eigen::Matrix3d::Identity() * 0.01);
    covs.push_back(Eigen::Matrix3d::Identity() * 0.02);

    quats.push_back(SO3d(Eigen::Vector3d(0, 0, 0)));
    quats.push_back(SO3d(Eigen::Vector3d(0, 0, 0.0)));
    std::vector<double> trial{0.7, 0.3};
    std::cout << weightedMean(quats, trial) << std::endl;
    //return 0;
    //list of probability pairs
    for (double prob = 0.5; prob <= 1.; prob += 0.05)
    {
        probs.push_back(std::vector<double>(2));
        probs.back()[0] = prob;
        probs.back()[1] = 1. - prob;
    }
    //compute for different sigmas
    for (double sigma = 0.0; sigma < 3.; sigma += 0.01)
    {
        quats[1] = SO3d{Eigen::Vector3d(0, 0, sigma)};
        Eigen::Matrix<double, -1, 1> diffs = diffs.Zero(probs.size());
        //compute mean diff for different probabilities
        for (int i = 0; i < probs.size(); i++)
        {
            diffs(i) = diff(quats, probs[i]);
        }
        std::cout << "Sigma: " << sigma << " Max diff: " << diffs.maxCoeff() << " Diffs: " << diffs.transpose() << std::endl;
        adekf::viz::plotVector(diffs, "Mean diff", 300, "abcdefghiklmn");
        //compute covariance diff for different probabilities
        for (int i = 0; i < probs.size(); i++)
        {
            diffs(i) = (weightedCovarianceSum(quats, covs, probs[i], weightedMean(quats, probs[i])) - badWeightedCovarianceSum(quats, covs, probs[i], badWeightedMean(quats, probs[i]))).norm();
        }
        std::cout << "Sigma: " << sigma << " Max diff: " << diffs.maxCoeff() << " Diffs: " << diffs.transpose() << std::endl;
        adekf::viz::plotVector(diffs, "Sigma diff", 300, "abcdefghiklmn");
    }
}

/**
 * Calculates the likelihood of a vector given the covariance of the distribution
 * @param delta the vector to check the likelihood of
 * @param sigma The covariance
 * @return N(delta; 0, sigma)
 */
double likelihood(const Eigen::Vector3d &delta, const Eigen::Matrix3d &sigma)
{
    return exp(-0.5 * (delta).transpose() * sigma.inverse() * delta) / sqrt(sigma.determinant() * pow((2 * M_PI), sigma.rows()));
}

/**
 * Plots a numeric gold standard mixing against the naive mixing and the bp-mixing
 * @param argc command line argument counter
 * @param argv command line arguments
 */
void plot_real_gold_vs_()
{
    std::vector<SO3d> quats(2);
    std::vector<double> probs{0.7, 0.3};

    std::vector<Eigen::Matrix3d> covs;
    covs.push_back(Eigen::Matrix3d::Identity() * 0.01);
    covs.push_back(Eigen::Matrix3d::Identity() * 0.02);
    //Compute for different sigmas
    for (double sigma = 0.0; sigma < 3.; sigma += 0.05)
    {
        quats[0] = (SO3d(Eigen::Vector3d(0, 0, 0)));
        quats[1] = (SO3d(Eigen::Vector3d(0, 0, sigma)));
        //return 0;

        const double max = M_PI / 2;
        const double step = 0.02;
        const int steps = max / step;
        std::vector<SO3d> gold_quats(pow(steps, 3));
        std::vector<double> gold_probs(pow(steps, 3));
        //uniform sample of quaternions from distributions
        for (int l = 0; l < probs.size(); ++l)
        {

            for (int i = -steps; i < steps; i++)
            {
                for (int j = -steps; j < steps; j++)
                {
                    for (int k = -steps; k < steps; ++k)
                    {
                        Eigen::Vector3d delta = step * Eigen::Vector3d(i, j, k);
                        gold_quats.push_back(quats[l] + delta);
                        gold_probs.push_back(likelihood(delta, covs[l]) * probs[l]);
                    }
                }
            }
        }
        double sum = 0;
        //normalise probabilities of sampled quats
        for (const auto &prob : gold_probs)
        {
            sum += prob;
        }
        for (auto &prob : gold_probs)
        {
            prob /= sum;
        }
        sum = 0;
        //compute means
        auto mean = weightedMean(quats, probs);
        auto badMean = badWeightedMean(quats, probs);
        auto goldMean = weightedMean(gold_quats, gold_probs);
        Eigen::Matrix<double, 2, 1> diffs = diffs.Zero();
        diffs(0) = (mean - goldMean).norm();
        diffs(1) = (badMean - goldMean).norm();
        adekf::viz::plotVector(diffs, "Gold Mean diff", 305, "mb");
        auto goldCov = goldWeightedCovarianceSum(gold_quats, gold_probs, goldMean);
        diffs(0) = (weightedCovarianceSum(quats, covs, probs, mean) - goldCov).norm();
        diffs(1) = (badWeightedCovarianceSum(quats, covs, probs, badMean) - goldCov).norm();
        adekf::viz::plotVector(diffs, "Gold Sigma diff", 305, "mb");
    }
}

Eigen::Vector3d weightedRefMean(aligned_vector<SO3d> &quats, std::vector<double> &probs, SO3d ref)
{
    Eigen::Vector3d sum = sum.Zero();

    for (int i = 0; i < quats.size(); i++)
    {
        sum = sum + probs[i] * (quats[i] - ref);
    }
    return sum;
}

/**
 * Calculates a numeric gold standard Variance of quats.
 *
 * Uses a list of uniformly sampled quaternions from the distributions and combines them with the proper mixing
 *
 * @param quats The list of uniformly sampled quaternions
 * @param probs The probabilities of the quaternioms
 * @param target The weighted mean of means of the distributions
 * @return The weighted covariance sum
 */
Eigen::Matrix3d goldWeightedCovarianceSumRef(aligned_vector<SO3d> &quats, std::vector<double> &probs, const SO3d &ref)
{

    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mu_ref = weightedRefMean(quats, probs, ref);
    for (size_t i = 0; i < quats.size(); i++)
    {
        if (probs[i] > 0.)
        {
            auto diff = ((quats[i] - ref) - mu_ref).eval();
            sum += probs[i] * (diff * diff.transpose());
        }
    }
    return sum;
}

/**
 * Plots a numeric gold standard mixing against the naive mixing and the bp-mixing
 * @param argc command line argument counter
 * @param argv command line arguments
 */
void plot_covariance_centered()
{
    SO3d r1, r2; //r1 r2

    Eigen::Matrix3d cov = cov.Identity() * 0.1;
    //Compute for different sigmas
    for (double sigma = 0.0; sigma < 3.; sigma += 0.05)
    {
        r1 = (SO3d(Eigen::Vector3d(0, 0, 0)));
        r2 = (SO3d(Eigen::Vector3d(0, 0, sigma)));
        Eigen::Vector3d diff = r2 - r1;
        const double max = M_PI / 2;
        const double step = 0.02;
        const int steps = max / step;
        aligned_vector<SO3d> gold_quats(pow(steps, 3));
        std::vector<double> gold_probs(pow(steps, 3));
        //uniform sample of quaternions from distribution

        for (int i = -steps; i < steps; i++)
        {
            for (int j = -steps; j < steps; j++)
            {
                for (int k = -steps; k < steps; ++k)
                {
                    Eigen::Vector3d delta = step * Eigen::Vector3d(i, j, k);
                    gold_quats.push_back(r1 + (diff + delta));
                    gold_probs.push_back(likelihood(delta, cov));
                }
            }
        }
        double sum = 0;
        //normalise probabilities of sampled quats
        for (const auto &prob : gold_probs)
        {
            sum += prob;
        }
        for (auto &prob : gold_probs)
        {
            prob /= sum;
        }
        sum = 0;
        //compute means

        Eigen::Matrix<double, 2, 1> diffs = diffs.Zero();
        auto goldCov = goldWeightedCovarianceSumRef(gold_quats, gold_probs, r2);
        auto transform = transformReferenceJacobian(r1, r2, diff);
        auto linearCov = transform * cov * transform.transpose();
        diffs(0)=(linearCov-goldCov).norm();
        diffs(1)=(cov-goldCov).norm();
        //diffs(0) = goldCov.determinant();
        adekf::viz::plotVector(diffs, "Covariance transform centered", 61, "mb");
    }
}

void plot_covariance_displaced()
{
    SO3d r1, r2; //r1 r2

    Eigen::Matrix3d cov = cov.Identity() * 0.1;
    //Compute for different sigmas
    for (double sigma = 0.0; sigma < 3.; sigma += 0.05)
    {
        r1 = (SO3d(Eigen::Vector3d(0, 0, 0)));
        r2 = (SO3d(Eigen::Vector3d(0, 0, sigma)));
        Eigen::Vector3d diff = r2 - r1;
        const double max = M_PI / 2;
        const double step = 0.02;
        const int steps = max / step;
        aligned_vector<SO3d> gold_quats(pow(steps, 3));
        std::vector<double> gold_probs(pow(steps, 3));
        //uniform sample of quaternions from distribution

        for (int i = -steps; i < steps; i++)
        {
            for (int j = -steps; j < steps; j++)
            {
                for (int k = -steps; k < steps; ++k)
                {
                    Eigen::Vector3d delta = step * Eigen::Vector3d(i, j, k);
                    gold_quats.push_back(r1 + (delta));
                    gold_probs.push_back(likelihood(delta, cov));
                }
            }
        }
        double sum = 0;
        //normalise probabilities of sampled quats
        for (const auto &prob : gold_probs)
        {
            sum += prob;
        }
        for (auto &prob : gold_probs)
        {
            prob /= sum;
        }
        sum = 0;
        //compute means

        Eigen::Matrix<double, 2, 1> diffs = diffs.Zero();
        auto goldCov = goldWeightedCovarianceSumRef(gold_quats, gold_probs, r2);
        auto transform = transformReferenceJacobian(r1, r2);
        auto linearCov = transform * cov * transform.transpose();
        diffs(0)=(linearCov-goldCov).norm();
        diffs(1)=(cov-goldCov).norm();
        //diffs(0) = goldCov.determinant();
        adekf::viz::plotVector(diffs, "Covariance transform displaced", 61, "mb");
    }
}


void plot_covariance_gold()
{
    SO3d r1, r2; //r1 r2

    Eigen::Matrix3d cov = cov.Identity() * 0.1;
    //Compute for different sigmas
    for (double sigma = 0.0; sigma < 3.; sigma += 0.05)
    {
        r1 = (SO3d(Eigen::Vector3d(0, 0, 0)));
        r2 = (SO3d(Eigen::Vector3d(0, 0, sigma)));
        Eigen::Vector3d diff = r2 - r1;
        const double max = M_PI / 2;
        const double step = 0.02;
        const int steps = max / step;
        aligned_vector<SO3d> gold_quats_centered(pow(steps, 3));
       
        aligned_vector<SO3d> gold_quats_displaced(pow(steps, 3));
       
        std::vector<double> gold_probs(pow(steps, 3));

        //uniform sample of quaternions from distribution

        for (int i = -steps; i < steps; i++)
        {
            for (int j = -steps; j < steps; j++)
            {
                for (int k = -steps; k < steps; ++k)
                {
                    Eigen::Vector3d delta = step * Eigen::Vector3d(i, j, k);
                    gold_quats_displaced.push_back(r1 + (delta));
                    gold_quats_centered.push_back(r1+(diff+delta));
                    gold_probs.push_back(likelihood(delta, cov));
                }
            }
        }
        double sum = 0;
        //normalise probabilities of sampled quats
        for (const auto &prob : gold_probs)
        {
            sum += prob;
        }
        for (auto &prob : gold_probs)
        {
            prob /= sum;
        }
        sum = 0;
        //compute means

        Eigen::Matrix<double, 3, 1> diffs = diffs.Zero();
        auto goldCovDisplaced = goldWeightedCovarianceSumRef(gold_quats_displaced, gold_probs, r2);
        auto goldCovCentered=goldWeightedCovarianceSumRef(gold_quats_centered,gold_probs,r2);
        diffs(0)=goldCovDisplaced.determinant()/cov.determinant();
        diffs(1)=goldCovCentered.determinant()/cov.determinant();
        diffs(2)=diffs(0)*diffs(1);


        //diffs(0) = goldCov.determinant();
        adekf::viz::plotVector(diffs, "Covariance transform gold", 61, "DCM");
        auto transform = transformReferenceJacobian(r1, r2);
        auto linearCov = (transform * cov * transform.transpose()).eval();
        diffs(0)=linearCov.determinant()/cov.determinant();
        transform=transformReferenceJacobian(r1,r2,diff);
        linearCov = transform * cov * transform.transpose();
        diffs(1)=linearCov.determinant()/cov.determinant();
        diffs(2)=diffs(0)*diffs(1);
        adekf::viz::plotVector(diffs, "Covariance transform linear", 61, "DCM");
    }
}

std::vector<std::thread> threads;
auto parallel=[](auto function){ threads.emplace_back(function);};
int main(int argc, char *argv[])
{
    viz::initGuis(argc, argv);
    
    //Choose some to plot
    //parallel(plot_bad_vs_bp);
    //parallel(plot_real_gold_vs_);//takes really long ~10 min
    //parallel(plot_covariance_displaced);
    //parallel(plot_covariance_centered); 
    parallel(plot_covariance_gold);
    viz::runGuis();
    for(auto & thread: threads) thread.join();
    return 0;
}