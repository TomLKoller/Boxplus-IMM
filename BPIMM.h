//
// Created by tomlucas on 07.05.20.
//

#ifndef CYLINDEREXAMPLE_ADIMM_H
#define CYLINDEREXAMPLE_ADIMM_H


//#include "../../include/ADEKF/ADEKF.h"
#include "ADEKFUtils.h"
#include <list>
#include <any>
#include <tuple>
#include <vector>


namespace adekf {
    /**
     * Dynamically applies a function on an element of a tuple
     * @tparam lambda The type of the function (lambda object)
     * @tparam TUPLE The type of the tuple
     * @tparam depth The size of the tuple -1
     * @tparam lambda_args The argument types of the lambda function
     * @param target_depth index of the element to apply the function on
     * @param function  The Function
     * @param tuple  The tuple
     * @param function_args Arguments of the function
     */
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
     * Implements the boxplus IMM
     * @tparam State
     * @tparam Models
     */
    template<template<typename> class Filter, typename State, typename DYNAMIC_NOISE_TYPE, typename ...Models>
    class BPIMM {
        protected:
        /**
         * Type of the covariance, automatically retrieved from the state type
         */
        using Covariance=typename Filter<State>::Covariance;

        /**
       * Number of dynamic models
       */
        static constexpr unsigned MODEL_COUNT = sizeof...(Models);

        /**
         * The bank with all filters  filter_bank[i].first is the filter itself whereas filter_bank[i].second is the identifier of the used dynamic model
         */
        std::vector<std::pair<Filter<State>, int  >> filter_bank;
        /**
         * The used dynamic models. All models have to be passed at once at initialisation.
         * It is possible to instantiate multiple internal filters with the same dynamic model.
         */
        std::tuple<Models ...> model_bank;


        /**
         * The dynamic noises corresponding to the dynamic models.
         */
        std::array<DYNAMIC_NOISE_TYPE, MODEL_COUNT> dynamic_noises;


        /**
         * Degree of freedom of the state (dimension of covariance and geodesic distances)
         */
        static constexpr unsigned DOF = DOFOf<State>;
        /**
         * Whether the state is a manifold or not
         */
        static constexpr bool isManifoldState = std::is_convertible_v<std::remove_const_t<State> *, Manifold *>;
        /**
         * Transition probabilities
        * t_prob(i,j) -> transition from mode i to mode  j
        */
        Eigen::MatrixXd transition_probabilities;
        /**
         * Probabilities of the models
         */
        Eigen::VectorXd model_probabilities;
        /**
         * Normalisation factors. Stored, since they are calculated in predict and required in update.
         */
        Eigen::MatrixXd cs;
        /**
         * Max iterations of the iterative boxplus mean
         */
        static constexpr unsigned max_iterations = 100;
        /**
         * Termination criteria of the iterative boxplus mean
         * changes < weighted_mean_epsilon are interpreted as convergence
         */
        static constexpr double weighted_mean_epsilon = 1e-8;

        private:
        /**
         * Flag to track whether multiple updates are performed after each other
         */
        bool multipleUpdates = false;


    public:
        /**
         * The output state of the IMM
         */
        State mu;
        /**
         * The output covariance of the IMM
         */
        Covariance sigma;

        /**
         * Retrieves a reference to an internal filter.
         * @index Which filter to retrieve
         * @return a reference to an internal filter
         */
        Filter<State> &getFilter(size_t index) {
            return filter_bank[index].first;
        }
        /**
         * Returns the vector of model probabilities
         * @return the model probabilities
         */
        const Eigen::VectorXd &getModelProbabilities() {
            return model_probabilities;
        }


        /**
         * Sets up the basics of the filter
         * @param _mu Initial Expected Value of the State
         * @param _sigma Initial Covariance of the State
         * @param dyn_noises A initializer list of dynamic noises corresponding to the following dynamic models
         * @param models the dynamic models used in this filter
         */
        BPIMM(const State &_mu, const Covariance &_sigma, std::initializer_list<DYNAMIC_NOISE_TYPE> dyn_noises, Models ... models) : mu(_mu), sigma(_sigma),
                                                                                                                                     model_bank(models ...) {
            assert(dyn_noises.size() == MODEL_COUNT && "You have to pass as many dynamic noises as models");
            std::copy(dyn_noises.begin(), dyn_noises.end(), dynamic_noises.begin());
        }

        /**
         * Returns the number of used filters
         * @return number of filters
         */
        size_t numFilters() const { return filter_bank.size(); }

        /**
         * Sets the transition probabilities
         * @tparam Derived  the type of the transition probability matrix.
         * @param t_prob The transition probabilities. Has to be a  MxM square matrix.
         */
        template<typename Derived>
        void setTransitionProbabilities(const Eigen::MatrixBase<Derived> &t_prob) {
            assert(t_prob.rows() == numFilters() && t_prob.rows() == t_prob.cols() && "Transition Probabilities need to be MxM for M models");

            transition_probabilities = t_prob;
        }

        /**
         * Adds a model to the filter bank.
         * The IMM can have multiple filters running the same model
         * Each call to addFilter adds a filter with a chosen dynamic model
         * @param index the index of the dynamic model (as it was passed on the constructor)
         */
        void addFilter(int index) {
            filter_bank.push_back(std::make_pair(Filter<State>(mu, sigma), index));
        }
        /**
         * Adds multiple models to the filter bank.
         * Calls @see addFilter on each passed index.
         * @param indices a list of indices of the dynamic models to add as filters
         */
        void addFilters(std::initializer_list<int> indices) {
            for (int index: indices) {
                addFilter(index);
            }
        }
        /**
         * Set the start probabilities of the filters.
         * @tparam Derived The matrix type of the start probabilities
         * @param start_prob the start probabilities. Should sum up to 1.
         */
        template<typename Derived>
        void setStartProbabilities(const Eigen::MatrixBase<Derived> &start_prob) {
            assert(start_prob.rows() == numFilters() && start_prob.cols() == 1 && "Start Probabilities need to be Mx1 for M models");
            model_probabilities = start_prob;
        }
         constexpr static auto getCurrentState = [](auto &filter) { return filter.mu; };
        /**
         * Computes the weighted sum of all filters.
         * Decides on its own whether it uses boxplus weighted sum or vector normal sum.
         * @param probabilities
         * @param last  the last weighted sum as a starting point for the iterative algorithm.
         * @return The weighted sum of the filter means.
         */
        template <typename GetState = decltype(getCurrentState)>
        State weightedStateSum(const Eigen::Matrix<double, -1, 1> &probabilities, const State &last, const GetState &stateGetter = getCurrentState)
        {
            assert(abs(probabilities.sum() - 1) < 0.001 && "Probabilities must sum up to 1");
            if constexpr (isManifoldState)
            {
                State sum = last;
                State old_sum = last;
                decltype(sum - old_sum) diff_sum = diff_sum.Zero();
                decltype(sum - old_sum) selector_weights = selector_weights.Zero();
                for (size_t t = 0; t < numFilters(); t++)
                {
                    //  selector_weights += selectors[filter_bank[t].second] * probabilities[t];
                }

                int iterations = 0;
                do
                {
                    iterations++;
                    old_sum = sum;
                    diff_sum = diff_sum.Zero();
                    for (int i = 0; i < numFilters(); i++)
                    {
                        if (probabilities(i) < 0.)
                            LOG_STREAM << "Fatal failure" LOG_END else diff_sum = diff_sum + probabilities(i) * (stateGetter(filter_bank[i].first) - sum); //.cwiseProduct(selectors[filter_bank[i].second]);
                    }
                    sum = sum + diff_sum; //.cwiseQuotient(selector_weights);
                } while (iterations <= max_iterations && diff_sum.norm() > weighted_mean_epsilon);
                if (iterations > max_iterations)
                    printf("Warning: stopped due to excess of iterations");
                return sum;
            }
            else
            {
                State sum = sum.Zero();
                for (size_t i = 0; i < numFilters(); i++)
                {
                    if (probabilities(i) < 0.)
                        LOG_STREAM << "Fatal failure" LOG_END else if (probabilities(i) > 0.)
                                sum += stateGetter(filter_bank[i].first) * probabilities(i);
                }
                return sum;
            }
        }

        constexpr static auto getCurrentSigma = [](auto &filter) { return filter.sigma; };
        /**
         * Weighted Sum of covariances.
         * Calculates the covariance of a mixture of Gaussians.
         * @param probabilities Weights of the modes
         * @param target The mean of the mixed Gaussian
         * @return E[[Sigma_i -target]]
         */
        template <typename GetState = decltype(getCurrentState), typename GetSigma = decltype(getCurrentSigma)>
        Covariance weightedCovarianceSum(const Eigen::Matrix<double, -1, 1> &probabilities, const State &target, const GetState &stateGetter = getCurrentState, const GetSigma &sigmaGetter = getCurrentSigma)
        {
            assert(abs(probabilities.sum() - 1) < 0.001 && "Probabilities must sum up to 1");
            if constexpr (isManifoldState)
            {
                Covariance sum = Covariance::Zero(DOF, DOF);
                Covariance weights = Covariance::Zero(DOF, DOF);
                for (size_t i = 0; i < numFilters(); i++)
                {
                    if (probabilities(i) > 0.)
                    {
                        decltype(mu - mu) diff;
                        auto plus_diff = eval(stateGetter(filter_bank[i].first) + getDerivator<DOF>() - target);
                        Eigen::Matrix<double, DOF, DOF> D(DOF, DOF);
                        //Initialise the Jacobian
                        for (size_t j = 0; j < DOF; ++j)
                        {
                            D.col(j) = plus_diff[j].v; //write to cols since col major (transposes matrix )
                            diff(j) = plus_diff[j].a;
                        }
                        //Covariance select = selectors[filter_bank[i].second] * selectors[filter_bank[i].second].transpose() * probabilities(i);
                        sum += probabilities(i) * (D.transpose() * sigmaGetter(filter_bank[i].first) * D + diff * diff.transpose());
                        //weights += select;
                    }
                }
                //assert(isPositiveDefinite(sum));

                return sum;
                //Non-Manifold Case
            }
            else
            {
                Covariance sum = Covariance::Zero(DOF, DOF);
                for (size_t i = 0; i < numFilters(); i++)
                {
                    if (probabilities(i) > 0.)
                    {
                        auto diff = (stateGetter(filter_bank[i].first) - target).eval();
                        sum += probabilities(i) * (sigmaGetter(filter_bank[i].first) + diff * diff.transpose());
                    }
                }
                return sum;
            }
        }

        /**
         * Calculates the normalisation constants c_s
         */
        void calcCS() {
            cs = (transition_probabilities.transpose() * model_probabilities).transpose();
            multipleUpdates = false;
            for (size_t t = 0; t < cs.rows(); t++) {
                assert(cs(t) > 0. && "Normalisation values are erronous");
            }
        }

        /**
         * Performs the interaction step.
         * Calculates new c_s @see calcCS().
         * mixes the states with @see weightedStateSum and @see weightedCovarianceSum
         */
        void interaction() {
            calcCS();
            Eigen::MatrixXd mixing_prob = transition_probabilities;
            //Calc mixing probs
            for (size_t i = 0; i < numFilters(); i++) {
                mixing_prob.col(i) = mixing_prob.col(i).array() * model_probabilities.array();
                mixing_prob.row(i) = mixing_prob.row(i).array() / cs.row(0).array();
            }

            //calc new means
            std::vector<State> mixed_states(numFilters());
            for (size_t j = 0; j < numFilters(); j++) {
                mixed_states[j] = weightedStateSum(mixing_prob.col(j), filter_bank[j].first.mu);
            }

            //calc new covariances
            std::vector<Covariance> mixed_cov(numFilters());
            for (size_t j = 0; j < numFilters(); j++) {
                mixed_cov[j] = weightedCovarianceSum(mixing_prob.col(j), mixed_states[j]);
            }
            //Set values
            for (size_t j = 0; j < numFilters(); j++) {
                filter_bank[j].first.mu = mixed_states[j];
                //assert positiv semidefinitness
                assert(filter_bank[j].first.sigma.determinant() >= 0.);
                assert(mixed_cov[j].determinant() >= 0.);
                filter_bank[j].first.sigma = mixed_cov[j];
            }

        }
        /**
         * Performs the prediction step of the filtering.
         * Calls the predict method of each inner Filter.
         * @tparam Controls The types of the input controls of the dynamic models
         * @param u The input controls for the dynamic models.
         */
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

        /**
         * Calls predict on the inner Filters in the nonAdditiveNoise Variant.
         * @tparam Controls Types of the dynamic models inputs.
         * @param u The inputs of the dynamic models.
         */
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



        /**
         * Performs the update step of each inner filter
         * @tparam Measurement Type of the measurement
         * @tparam MeasurementModel Type of the Measurementmodel (lambda)
         * @tparam DerivCov Type of the Covariance Matrix
         * @tparam Variables Types of the function parameters
         * @param measurementModel The measurement model
         * @param R The Covariance matrix of the measurement
         * @param z The measurement
         * @param variables The parameters of the function
         */
        template<typename Measurement, typename MeasurementModel, typename DerivCov, typename... Variables>
        void update(MeasurementModel measurementModel, const MatrixBase <DerivCov> &R, const Measurement &z,
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
            //Calculate new likelihoods
            model_probabilities = model_probabilities.unaryExpr([&maxCoeff](double value) { return exp(value - maxCoeff); });
            //Normalise likelihods to sum 1
            model_probabilities /= model_probabilities.sum();
            //std::cout<< model_probabilities<< std::endl;
            multipleUpdates = true;
        }

        /**
         * Instead of calling update, pass measurement probabilities directly.
         * Needed if you update only a few internal filters while others have constant probabilities.
         * @tparam Derived Matrixtype of the probabilities. Has to be an Mx1 vector where M is the number of models.
         * @param meas_probs The measurement probabilities for each mode.
         */
        template<typename Derived>
        void passMeasurementProbabilities(const MatrixBase <Derived> &meas_probs) {
            assert(meas_probs.rows() == model_probabilities.rows() && "You have to pass a measurement probability for each model");
            for (size_t i = 0; i < model_probabilities.rows(); i++) {
                model_probabilities(i) = cs(i) * meas_probs(i);
            }
            model_probabilities /= model_probabilities.sum();
        }

        /**
         * Performs the combination step.
         * Calls @see weightedStateSum
         * and @see weightedCovarianceSum
         */
        void combination() {
            mu = weightedStateSum(model_probabilities, mu);
            sigma = weightedCovarianceSum(model_probabilities, mu);
        }


    };

    /**
  * General Deduction Template for the BPIMM based on StateRetriever.
  * This is needed so you can type BPIMM imm(State,COV) without template arguments
  */



    template<typename DERIVED, typename COV_TYPE, typename DYN_COV, typename ... Models>
    BPIMM(const DERIVED &, const COV_TYPE &, std::initializer_list<DYN_COV>,
          Models ...) -> BPIMM<ADEKF, typename StateInfo<DERIVED>::type, DYN_COV, Models ...>;

    /**
     * Helper Function for partial template deduction
     * @tparam Filter The type of the Filter
     * @tparam STATE_TYPE The type of the State
     * @tparam COV_TYPE The type of the covariance matrix.
     * @tparam Models The types of the dynamic models
     * @param mu The state
     * @param sigma The covariance
     * @param models The different dynamic models.
     * @return A new BPIMM  initialsied with the given data.
     */
    template<template<typename> class Filter, typename STATE_TYPE, typename COV_TYPE, typename ... Models>
    auto createADIMM(const STATE_TYPE &mu, const COV_TYPE &sigma, Models ... models) -> BPIMM<Filter, typename StateInfo<STATE_TYPE>::type, Models ...> {
        return BPIMM<Filter, typename StateInfo<STATE_TYPE>::type, Models ...>(mu, sigma, models ...);
    }

}

#endif //CYLINDEREXAMPLE_ADIMM_H
