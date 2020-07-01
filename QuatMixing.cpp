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

SO3d weightedMean(std::vector<SO3d> &quats, std::vector<double> &probs) {
    SO3d sum = quats[0];
    SO3d old_sum = sum;
    decltype(sum - old_sum) diff_sum = diff_sum.Zero();
    decltype(sum - old_sum) selector_weights = selector_weights.Zero();

    int iterations = 0;
    do {
        iterations++;
        old_sum = sum;
        diff_sum = diff_sum.Zero();
        for (int i = 0; i < quats.size(); i++) {
            diff_sum = diff_sum + probs[i] * (quats[i] - sum);//.cwiseProduct(selectors[filter_bank[i].second]);
        }
        sum = sum + diff_sum;//.cwiseQuotient(selector_weights);
    } while (iterations <= max_iterations && (sum - old_sum).norm() > weighted_mean_epsilon);
    if (iterations > max_iterations)
        printf("Warning: stopped due to excess of iterations");
    return sum;
}


Eigen::Matrix3d weightedCovarianceSum(std::vector<SO3d> &quats, std::vector<Eigen::Matrix3d> &covs, std::vector<double> &probs, const SO3d &target) {
    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d weights = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++) {
        if (probs[i] > 0.) {
            auto diff = (quats[i] - target).eval();
            auto plus_diff = eval((quats[i] + getDerivator<3>()) - target);
            Eigen::Matrix<double, 3, 3> D(3, 3);
            //Initialise the Jacobian
            for (size_t j = 0; j < 3; ++j)
                D.col(j) = plus_diff[j].v;   //write to cols since col major (transposes matrix )
            //Covariance select = selectors[filter_bank[i].second] * selectors[filter_bank[i].second].transpose() * probabilities(i);
            sum += probs[i] * (D.transpose() * (covs[i]) * D + diff * diff.transpose());
            //weights += select;
        }
    }
    assert(sum.determinant() > 0.);
    return sum;
}


SO3d badWeightedMean(std::vector<SO3d> &quats, std::vector<double> &probs) {
    Eigen::Vector4d sum = sum.Zero();
    for (int i = 0; i < quats.size(); i++) {
        sum += quats[i].coeffs() * probs[i];
    }
    sum = sum.normalized();
    return SO3d(sum.data());
}

Eigen::Matrix3d goldWeightedCovarianceSum(std::vector<SO3d> &quats, std::vector<double> &probs, const SO3d &target) {

    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++) {
        if (probs[i] > 0.) {
            auto diff = (quats[i] - target).eval();
            sum += probs[i] * (diff * diff.transpose());
        }
    }
    return sum;
}


Eigen::Matrix3d badWeightedCovarianceSum(std::vector<SO3d> &quats, std::vector<Eigen::Matrix3d> &covs, std::vector<double> &probs, const SO3d &target) {

    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < quats.size(); i++) {
        if (probs[i] > 0.) {
            auto diff = (quats[i] - target).eval();
            sum += probs[i] * (covs[i] + diff * diff.transpose());
        }
    }
    return sum;
}


double diff(std::vector<SO3d> &quats, std::vector<double> &probs) {
    return (weightedMean(quats, probs) - badWeightedMean(quats, probs)).norm();
}



void plot_bad_vs_bp(int argc, char *argv[]) {
    viz::initGuis(argc, argv);
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
    for (double prob = 0.5; prob <= 1.; prob += 0.05) {
        probs.push_back(std::vector<double>(2));
        probs.back()[0] = prob;
        probs.back()[1] = 1. - prob;
    }
    for (double sigma = 0.0; sigma < 3.; sigma += 0.01) {
        quats[1] = SO3d{Eigen::Vector3d(0, 0, sigma)};
        Eigen::Matrix<double, -1, 1> diffs = diffs.Zero(probs.size());
        for (int i = 0; i < probs.size(); i++) {
            diffs(i) = diff(quats, probs[i]);
        }
        std::cout << "Sigma: " << sigma << " Max diff: " << diffs.maxCoeff() << " Diffs: " << diffs.transpose() << std::endl;
        adekf::viz::plotVector(diffs, "Mean diff", 300, "abcdefghiklmn");

        for (int i = 0; i < probs.size(); i++) {
            diffs(i) = (weightedCovarianceSum(quats,covs,probs[i], weightedMean(quats,probs[i]))- badWeightedCovarianceSum(quats,covs,probs[i], badWeightedMean(quats,probs[i]))).norm();
        }
        std::cout << "Sigma: " << sigma << " Max diff: " << diffs.maxCoeff() << " Diffs: " << diffs.transpose() << std::endl;
        adekf::viz::plotVector(diffs, "Sigma diff", 300, "abcdefghiklmn");
    }
    viz::runGuis();
}



double likelihood(const Eigen::Vector3d &delta,const Eigen::Matrix3d & sigma){
    return exp(-0.5 * (delta).transpose() * sigma.inverse() * delta)/sqrt(sigma.determinant() * pow((2 * M_PI), sigma.rows()));
}

void plot_real_gold_vs_bp(int argc, char *argv[]) {
    viz::initGuis(argc, argv);
    std::vector<SO3d> quats(2);
    std::vector<double> probs{0.7, 0.3};

    std::vector<Eigen::Matrix3d> covs;
    covs.push_back(Eigen::Matrix3d::Identity() * 0.01);
    covs.push_back(Eigen::Matrix3d::Identity() * 0.02);
    for (double sigma = 0.0; sigma < 3.; sigma += 0.05) {
        quats[0] = (SO3d(Eigen::Vector3d(0, 0, 0)));
        quats[1] = (SO3d(Eigen::Vector3d(0, 0, sigma)));
        //return 0;


        const double max = M_PI/2;
        const double step = 0.02;
        const int steps = max / step;
        std::vector<SO3d> gold_quats(pow(steps,3));
        std::vector<double> gold_probs(pow(steps,3));
        for (int l = 0; l < probs.size(); ++l) {

            for (int i = -steps; i < steps; i++) {
                for (int j = -steps; j < steps; j++) {
                    for (int k = -steps; k < steps; ++k) {
                        Eigen::Vector3d delta = step * Eigen::Vector3d(i, j, k);
                        gold_quats.push_back(quats[l] + delta);
                        gold_probs.push_back(likelihood(delta, covs[l]) * probs[l]);
                    }
                }
            }
        }
        double sum = 0;
        for (const auto &prob : gold_probs) {
            sum += prob;
        }
        for (auto &prob : gold_probs) {
            prob /= sum;
        }
        sum = 0;

        auto mean = weightedMean(quats, probs);
        auto badMean=badWeightedMean(quats,probs);
        auto goldMean = weightedMean(gold_quats, gold_probs);
        Eigen::Matrix<double, 2, 1> diffs = diffs.Zero();
        diffs(0) = (mean - goldMean).norm();
        diffs(1) = (badMean - goldMean).norm();
        adekf::viz::plotVector(diffs, "Gold Mean diff", 305, "mb");
        diffs(0)=(weightedCovarianceSum(quats,covs,probs,mean)-goldWeightedCovarianceSum(gold_quats,gold_probs,goldMean)).norm();
        diffs(1)=(badWeightedCovarianceSum(quats,covs,probs,mean)-goldWeightedCovarianceSum(gold_quats,gold_probs,goldMean)).norm();
        adekf::viz::plotVector(diffs, "Gold Sigma diff", 305, "mb");
    }
    viz::runGuis();
}


int main(int argc, char *argv[]) {
    plot_bad_vs_bp(argc, argv);
    //plot_real_gold_vs_bp(argc,argv);
    return 0;
}