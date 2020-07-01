//
// Created by tomlucas on 07.05.20.
//

#ifndef CYLINDEREXAMPLE_ADIMM_H
#define CYLINDEREXAMPLE_ADIMM_H


//#include "../../include/ADEKF/ADEKF.h"
#include "../../include/ADEKF/ADEKFUtils.h"
#include <list>
#include <any>
#include <tuple>
#include <vector>


namespace adekf {
    template<typename lambda, typename TUPLE, int depth = std::tuple_size<TUPLE>() - 1, typename ... lambda_args>
    void applyOnTupleElement(int target_depth, lambda &function, TUPLE tuple, lambda_args &...  function_args) {
        if (depth == target_depth)
            function(std::get<depth>(tuple), function_args ...);
        else if constexpr (depth > 0)
            applyOnTupleElement<lambda, TUPLE, depth - 1, lambda_args ...>(target_depth, function, tuple, function_args ...);
        else {
            LOG_STREAM << "Did not found a model with the given number" LOG_END
        }
    }

    /**
     * Todo doku
     * Todo manifold states ((manifold + manifold)*p ?)
     * @tparam State
     * @tparam Models
     */
    template<template<typename> class Filter, typename State, typename DYNAMIC_NOISE_TYPE, typename ...Models>
    class ADIMM {
        using Covariance=typename Filter<State>::Covariance;
        std::vector<std::pair<Filter<State>, int  >> filter_bank;
        std::tuple<Models ...> model_bank;

        static constexpr unsigned MODEL_COUNT = sizeof...(Models);
        static constexpr unsigned DOF = DOFOf<State>;
        static constexpr bool isManifoldState = std::is_convertible_v<std::remove_const_t<State> *, Manifold *>;
        std::array<DYNAMIC_NOISE_TYPE, MODEL_COUNT> dynamic_noises;
        std::array<Eigen::VectorXd, MODEL_COUNT> selectors;
        // t_prob(i,j) -> transition from mode i to mode  j
        Eigen::MatrixXd transition_probabilities;
        Eigen::VectorXd model_probabilities;
        Eigen::MatrixXd cs;
        static constexpr unsigned max_iterations = 100;
        static constexpr double weighted_mean_epsilon = 1e-8;
        bool multipleUpdates = false;


    public:
        State mu;
        Covariance sigma;

        Filter<State> &getFilter(size_t index) {
            return filter_bank[index].first;
        }

        const Eigen::VectorXd &getModelProbabilities() {
            return model_probabilities;
        }


        /**
         * Sets up the basics of the filter
         * @param _mu Initial Expected Value of the State
         * @param _sigma Initial Covariance of the State
         * @param NUM_MODELS number of models, which will be used
         */
        ADIMM(const State &_mu, const Covariance &_sigma, std::initializer_list<DYNAMIC_NOISE_TYPE> dyn_noises, Models ... models) : mu(_mu), sigma(_sigma),
                                                                                                                                     model_bank(models ...) {
            //model_bank=std::make_tuple(models ...);

            assert(dyn_noises.size() == MODEL_COUNT && "You have to pass as many dynamic noises as models");
            std::copy(dyn_noises.begin(), dyn_noises.end(), dynamic_noises.begin());
        }


        template<typename ... SELECTORS>
        void setSelectors(SELECTORS ... selectors_) {
            static_assert(sizeof...(SELECTORS) == MODEL_COUNT && "You have to pass as many dynamic noises as models");
            selectors = std::array<Eigen::VectorXd, sizeof...(Models)>({selectors_ ...});
        }


        size_t numFilters() const { return filter_bank.size(); }

        template<typename Derived>
        void setTransitionProbabilities(const Eigen::MatrixBase<Derived> &t_prob) {
            assert(t_prob.rows() == numFilters() && t_prob.rows() == t_prob.cols() && "Transition Probabilities need to be MxM for M models");
            transition_probabilities = t_prob;
        }


        void addModel(int index) {
            filter_bank.push_back(std::make_pair(Filter<State>(mu, sigma), index));
        }

        void addModels(std::initializer_list<int> indices) {
            for (int index: indices) {
                addModel(index);
            }
        }

        template<typename Derived>
        void setStartProbabilities(const Eigen::MatrixBase<Derived> &start_prob) {
            assert(start_prob.rows() == numFilters() && start_prob.cols() == 1 && "Start Probabilities need to be Mx1 for M models");
            model_probabilities = start_prob;
        }

        State weightedStateSum(const Eigen::Matrix<double, -1, 1> &probabilities, const State &last) {
            assert(abs(probabilities.sum() - 1) < 0.001 && "Probabilities must sum up to 1");
            if constexpr(isManifoldState) {
                State sum = last;
                State old_sum = last;
                decltype(sum - old_sum) diff_sum = diff_sum.Zero();
                decltype(sum - old_sum) selector_weights = selector_weights.Zero();
                for (size_t t = 0; t < numFilters(); t++) {
                    //  selector_weights += selectors[filter_bank[t].second] * probabilities[t];
                }

                int iterations = 0;
                do {
                    iterations++;
                    old_sum = sum;
                    diff_sum = diff_sum.Zero();
                    for (int i = 0; i < numFilters(); i++) {
                        if (probabilities(i) < 0.)
                            LOG_STREAM << "FAtal failure" LOG_END
                        else
                            diff_sum = diff_sum + probabilities(i) * (filter_bank[i].first.mu - sum);//.cwiseProduct(selectors[filter_bank[i].second]);
                    }
                    sum = sum + diff_sum;//.cwiseQuotient(selector_weights);
                } while (iterations <= max_iterations && (sum - old_sum).norm() > weighted_mean_epsilon);
                if (iterations > max_iterations)
                    printf("Warning: stopped due to excess of iterations");
                return sum;
            } else {
                State sum = sum.Zero();
                for (size_t i = 0; i < numFilters(); i++) {
                    if (probabilities(i) < 0.)
                        LOG_STREAM << "FAtal failure" LOG_END
                    else if (probabilities(i) > 0.)
                        sum += filter_bank[i].first.mu * probabilities(i);
                }
                return sum;
            }
        }

        Covariance weightedCovarianceSum(const Eigen::Matrix<double, -1, 1> &probabilities, const State &target) {
            assert(abs(probabilities.sum() - 1) < 0.001 && "Probabilities must sum up to 1");
            if constexpr(isManifoldState) {
                Covariance sum = Covariance::Zero(DOF, DOF);
                Covariance weights = Covariance::Zero(DOF, DOF);
                for (size_t i = 0; i < numFilters(); i++) {
                    if (probabilities(i) > 0.) {
                        auto diff = (filter_bank[i].first.mu - target).eval();
                        auto plus_diff = eval((filter_bank[i].first.mu + getDerivator<DOF>()) - target);
                        Eigen::Matrix<double, DOF, DOF> D(DOF, DOF);
                        //Initialise the Jacobian
                        for (size_t j = 0; j < DOF; ++j)
                            D.col(j) = plus_diff[j].v;   //write to cols since col major (transposes matrix )
                        //Covariance select = selectors[filter_bank[i].second] * selectors[filter_bank[i].second].transpose() * probabilities(i);
                        sum += probabilities(i) * (D.transpose() * (filter_bank[i].first.sigma) * D + diff * diff.transpose());
                        //weights += select;
                    }
                }
                assert(sum.determinant() > 0.);



                return sum;
            } else {
                Covariance sum = Covariance::Zero(DOF, DOF);
                for (size_t i = 0; i < numFilters(); i++) {
                    if (probabilities(i) > 0.) {
                        auto diff = (filter_bank[i].first.mu - target).eval();
                        sum += probabilities(i) * (filter_bank[i].first.sigma + diff * diff.transpose());
                    }
                }
                return sum;
            }
        }


        void calcCS() {
            cs = (transition_probabilities.transpose() * model_probabilities).transpose();
            multipleUpdates = false;
            for (size_t t = 0; t < cs.rows(); t++) {
                assert(cs(t) > 0. && "Normalisation values are erronous");
            }
        }

        void interaction() {
            calcCS();
            Eigen::MatrixXd mixing_prob = transition_probabilities;
            for (size_t i = 0; i < numFilters(); i++) {
                mixing_prob.col(i) = mixing_prob.col(i).array() * model_probabilities.array();
                mixing_prob.row(i) = mixing_prob.row(i).array() / cs.row(0).array();
            }

            std::vector<State> mixed_states(numFilters());
            for (size_t j = 0; j < numFilters(); j++) {
                mixed_states[j] = weightedStateSum(mixing_prob.col(j), filter_bank[j].first.mu);
            }

            std::vector<Covariance> mixed_cov(numFilters());
            for (size_t j = 0; j < numFilters(); j++) {
                mixed_cov[j] = weightedCovarianceSum(mixing_prob.col(j), mixed_states[j]);
            }
            for (size_t j = 0; j < numFilters(); j++) {
                filter_bank[j].first.mu = mixed_states[j];
                assert(filter_bank[j].first.sigma.determinant() >= 0.);
                assert(mixed_cov[j].determinant() >= 0.);
                filter_bank[j].first.sigma = mixed_cov[j];
            }

        }

        template<typename... Controls>
        void predict(const Controls &...u) {
            assert(dynamic_noises[0].rows() == sigma.rows() && "dynamic noises need to have the same dimension as the state for predict");
            auto apply_predict = [&u...](auto &lambda, auto &filter, auto &Q) {
                //LOG_STREAM << "filter sigma before\n" <<filter.sigma LOG_END
                filter.predict(lambda, Q, u...);
                //LOG_STREAM << "filter sigma\n" <<filter.sigma << "Q\n"<< Q LOG_END
            };
            for (auto &filter : filter_bank) {
                applyOnTupleElement(filter.second, apply_predict, model_bank, filter.first, dynamic_noises[filter.second]);
            }
        }

        template<typename... Controls>
        void predictWithNonAdditiveNoise(const Controls &...u) {
            auto apply_predict = [&u...](auto &lambda, auto &filter, auto &Q) {
                //LOG_STREAM << "filter sigma before\n" <<filter.sigma LOG_END
                filter.predictWithNonAdditiveNoise(lambda, Q, u...);
                //LOG_STREAM << "filter sigma\n" <<filter.sigma << "Q\n"<< Q LOG_END
            };
            for (auto &filter : filter_bank) {
                applyOnTupleElement(filter.second, apply_predict, model_bank, filter.first, dynamic_noises[filter.second]);
            }
        }




        template<typename Measurement, typename MeasurementModel, typename Derived, typename... Variables>
        void update(MeasurementModel measurementModel, const MatrixBase <Derived> &R, const Measurement &z,
                    const Variables &...variables) {
            double log_likelihood = 0;
            size_t index = 0;
            if (multipleUpdates)
                cs = model_probabilities;
            //std::cout << "BEfore " << filter_bank[0].first.mu << std::endl;
            for (auto &filter: filter_bank) {
                filter.first.update(log_likelihood, measurementModel, R, z, variables ...);
                //assert(log_likelihood > 0. && "Likelihood was badly calculated");
                model_probabilities(index) = (log_likelihood) + log(cs(index));
                index++;

            }

            //std::cout << "After " << filter_bank[0].first.mu << std::endl;
            double maxCoeff = model_probabilities.maxCoeff();
            model_probabilities = model_probabilities.unaryExpr([&maxCoeff](double value) { return exp(value - maxCoeff); });
            model_probabilities /= model_probabilities.sum();
            //std::cout<< model_probabilities<< std::endl;
            multipleUpdates = true;
        }

        template<typename Derived>
        void passMeasurementProbabilities(const MatrixBase <Derived> &meas_probs) {
            assert(meas_probs.rows() == model_probabilities.rows() && "You have to pass a measurement probability for each model");
            for (size_t i = 0; i < model_probabilities.rows(); i++) {
                model_probabilities(i) = cs(i) * meas_probs(i);
            }
            model_probabilities /= model_probabilities.sum();
        }


        void combination() {
            mu = weightedStateSum(model_probabilities, mu);
            sigma = weightedCovarianceSum(model_probabilities, mu);
        }


    };

    /**
  * General Deduction Template for the ADIMM based on StateRetriever.
  * This is needed so you can type ADIMM imm(State,COV) without template arguments
  */



    template<typename DERIVED, typename COV_TYPE, typename DYN_COV, typename ... Models>
    ADIMM(const DERIVED &, const COV_TYPE &, std::initializer_list<DYN_COV>,
          Models ...) -> ADIMM<ADEKF, typename StateInfo<DERIVED>::type, DYN_COV, Models ...>;

    template<template<typename> class Filter, typename DERIVED, typename COV_TYPE, typename ... Models>
    auto createADIMM(const DERIVED &mu, const COV_TYPE &sigma, Models ... models) -> ADIMM<Filter, typename StateInfo<DERIVED>::type, Models ...> {
        return ADIMM<Filter, typename StateInfo<DERIVED>::type, Models ...>(mu, sigma, models ...);
    }

}

#endif //CYLINDEREXAMPLE_ADIMM_H
