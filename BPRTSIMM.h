//
// Created by tomlucas on 07.05.20.
//

#ifndef BPRTSIMM_H
#define BPRTSIMM_H

//#include "../../include/ADEKF/ADEKF.h"
#include "ADEKFUtils.h"
#include "rts-smoother.hpp"
#include "BPIMM.h"
#include "LogGaussianProbability.hpp"
#include <list>
#include <any>
#include <tuple>
#include <vector>

namespace adekf
{

    /**
     * @brief This class implements the boxplus rauch tung striebel interacting multiple model filter 
     * 
     * @tparam Filter The inner filter type (convenient template to use e.g. UKF later on)
     * @tparam State The Manifold state representation type 
     * (Either an Eigen::Vector for vector states, A Manifold like SO3 or a Compound Manifold created by ADEKF_MANIFOLD macro)
     * @tparam DYNAMIC_NOISE_TYPE Type of the dynamic noise (has to be the same for all models, so the biggest)
     * @tparam Models Variadic list of dynamic models of the IMM. Only pass each model once and create modes using the same dynamic model if required.
     */
    template <template <typename> class Filter, typename State, typename DYNAMIC_NOISE_TYPE, typename... Models>
    class BPRTSIMM  : public BPIMM<Filter,State,DYNAMIC_NOISE_TYPE, Models ...>
    {
        using T_BPIMM=BPIMM<Filter,State,DYNAMIC_NOISE_TYPE, Models ...>;
        /**
         * Type of the covariance, automatically retrieved from the state type
         */
        using Covariance = typename T_BPIMM::Covariance;

       
      

    public:
        aligned_vector<State> old_mus, smoothed_mus;
        aligned_vector<Covariance> old_sigmas, smoothed_sigmas;
        aligned_vector<Eigen::VectorXd> old_model_probabilities, smoothed_model_probabilities;

       
        /**
         * Sets up the basics of the filter
         * @param _mu Initial Expected Value of the State
         * @param _sigma Initial Covariance of the State
         * @param dyn_noises A initializer list of dynamic noises corresponding to the following dynamic models
         * @param models the dynamic models used in this filter
         */
        BPRTSIMM(const State &_mu, const Covariance &_sigma, std::initializer_list<DYNAMIC_NOISE_TYPE> dyn_noises, Models... models) : T_BPIMM(_mu,_sigma,dyn_noises, models ... )
        {
            storeEstimation(true);
        }

      
      
        /**
         * Performs the prediction step of the filtering.
         * Calls the predict method of each inner Filter.
         * stores the predicted estimation
         * @tparam Controls The types of the input controls of the dynamic models
         * @param u The input controls for the dynamic models.
         */
        template<typename... Controls>
        void predict(const Controls &...u) {
            assert(this->dynamic_noises[0].rows() == this->sigma.rows() && "dynamic noises need to have the same dimension as the state for predict");
            auto apply_predict = [&u...](auto &lambda, auto &filter, auto &Q) {
                //LOG_STREAM << "filter sigma before\n" <<filter.sigma LOG_END
                filter.predict(lambda, Q, u...);
                filter.storePredictedEstimation();
                //LOG_STREAM << "filter sigma\n" <<filter.sigma << "Q\n"<< Q LOG_END
            };
            for (auto &filter : this->filter_bank) {
                applyOnTupleElement(filter.second, apply_predict, this->model_bank, filter.first, this->dynamic_noises[filter.second]);
            }
        }

        /**
         * Calls predict on the inner Filters in the nonAdditiveNoise Variant.
         * 
         * stores the predicted estimation
         * @tparam Controls Types of the dynamic models inputs.
         * @param u The inputs of the dynamic models.
         */
        template<typename... Controls>
        void predictWithNonAdditiveNoise(const Controls &...u) {
            auto apply_predict = [&u...](auto &lambda, auto &filter, auto &Q) {
                //LOG_STREAM << "filter sigma before\n" <<filter.sigma LOG_END
                filter.predictWithNonAdditiveNoise(lambda, Q, u...);
                filter.storePredictedEstimation();
                //LOG_STREAM << "filter sigma\n" <<filter.sigma << "Q\n"<< Q LOG_END
            };
            for (auto &filter : this->filter_bank) {
                applyOnTupleElement(filter.second, apply_predict, this->model_bank, filter.first, this->dynamic_noises[filter.second]);
            }
        }
       



/**
 * @brief Stores the current estimation and model probabilities
 * 
 * Also for inner filters
 * On firstcall, Inner filters should not store the values as they store it automatically on construction. 
 * 
 * @param firstCall false to let inner filters store their estimates.
 */
        void storeEstimation(bool firstCall = false)
        {
            //Store Combined
            old_mus.push_back(this->mu);
            smoothed_mus.push_back(this->mu);
            old_sigmas.push_back(this->sigma);
            smoothed_sigmas.push_back(this->sigma);
            old_model_probabilities.push_back(this->model_probabilities);
            smoothed_model_probabilities.push_back(this->model_probabilities);
            //store for filters
            if (!firstCall) //smoothers store it automatically on creation
                std::for_each(this->filter_bank.begin(), this->filter_bank.end(), [](auto &filter) { filter.first.storeEstimation(); });
        }
        /**
         * @brief Smoothes all states in the given intervall
         * 
         * @tparam Controls Types of the dynamic model controls
         * @param steps the number of time steps to smooth
         * @param start the starting step to smooth (can be negative as in python). Assumes that the step start+1 is smoothed. Max is N-2 (or -1)
         * @param all_controls The control inputs of all time steps
         */
        template <typename... Controls>
        void smoothIntervalWithNonAdditiveNoise(size_t steps, int start, const std::vector<std::tuple<Controls...>> &all_controls)
        {
            if (start < 0)
            {
                start = old_mus.size() - 1 + start;
            }
            assert(start - steps >= 0 && "Can not smooth more estimates than available");
            assert(start < old_mus.size() - 1 && "Can not smooth a state which belongs to future");
            assert(all_controls.size() == old_mus.size() - 1 && "Requires all control inputs for the dynamic model.");
            size_t indefinit = 0, improved = 0;
            for (size_t k = start; k >= old_mus.size() - steps - 1; k--)
            {
                
                //std::cout << "IMM at iteration: " << k << std::endl;
                //bji
                Eigen::MatrixXd bs = this->transition_probabilities;
                Eigen::RowVectorXd es = (this->transition_probabilities.transpose() * old_model_probabilities[k]).transpose();
                //Calc mixing probs
                for (size_t j = 0; j < this->numFilters(); j++)
                {
                    //bs.col(j) = bs.col(j).array() * old_model_probabilities[k].array(); //old model probability shortens by dividation with ds still has influence by es
                    bs.row(j) = bs.row(j).array() / es.array(); //Due to multiplication order of operations does not matter
                }

                //mu i|j k+1|N
                Eigen::RowVectorXd ds = (bs * smoothed_model_probabilities[k + 1]).transpose();
                Eigen::MatrixXd mixing_prob = bs.transpose();
                for (size_t j = 0; j < this->numFilters(); j++)
                {
                    mixing_prob.col(j) = mixing_prob.col(j).array() * smoothed_model_probabilities[k + 1].array();
                    mixing_prob.row(j) = mixing_prob.row(j).array() / ds.array(); //Due to multiplication order of operations does not matter
                 }

                auto apply_smoothStep = [](auto &dynamicModel, auto &filter, auto &k, auto &Q, auto &controls, auto &X0, auto &P0) {
                    filter.smoothSingleStep(k, dynamicModel, Q, controls, X0, P0);
                };
                //X0j k|N  P0j k|N
                for (size_t j = 0; j < this->numFilters(); j++)
                {
                    
                    //std::cout << "mixing: " << mixing_prob.row(j) << std::endl;
                    auto getSmoothedState = [&k](auto &filter) { return filter.smoothed_mus[k + 1]; };
                    State X0 = this->weightedStateSum(mixing_prob.col(j), smoothed_mus[k + 1], getSmoothedState);
                    Covariance P0 = this->weightedCovarianceSum(mixing_prob.col(j), X0, getSmoothedState, [&k](auto &filter) { return filter.smoothed_sigmas[k + 1]; });
                    //Mode Matching Smoothing
                    size_t model_index = this->filter_bank[j].second;
                    applyOnTupleElement(model_index, apply_smoothStep, this->model_bank, this->filter_bank[j].first, k, this->dynamic_noises[model_index], all_controls[k], X0, P0);
                }


                //Collect log probabilities
                Eigen::MatrixXd log_probs{this->numFilters(),this->numFilters()};
                 for (size_t j = 0; j < this->numFilters(); j++)
                {
                    log_gaussian_probability N{this->filter_bank[j].first.predicted_sigmas[k + 1]};
                    auto predicted_mu = this->filter_bank[j].first.predicted_mus[k + 1];
                    for (size_t i = 0; i < this->numFilters(); i++){
                        log_probs(j,i)= N(this->filter_bank[i].first.smoothed_mus[k + 1], predicted_mu);
                    }
                }
                //normalize for more numeric robustness. Maybe improve to also fit minCoeff into a suitable range
                log_probs=log_probs.array()-log_probs.maxCoeff();

                //Smoothed mode Probability mu j k|N
                for (size_t j = 0; j < this->numFilters(); j++)
                {
                    //Lambda j k|N
                    double L = 0;
                    //likelihood
                    for (size_t i = 0; i < this->numFilters(); i++)
                    {
                        L += this->transition_probabilities(j, i) * exp(log_probs(j,i)); //N(this->filter_bank[i].first.smoothed_mus[k + 1], predicted_mu));
                    }
                    smoothed_model_probabilities[k](j) = L * old_model_probabilities[k](j);
                }
                //std::cout <<"old: "<< old_model_probabilities[k].transpose() << std::endl;
                //std::cout <<"smo: "<< smoothed_model_probabilities[k].transpose() << std::endl;
                //normalise
                smoothed_model_probabilities[k] /= smoothed_model_probabilities[k].sum();

                //Calculate Smoothed IMM estimate
                auto getSmoothedState = [&k](auto &filter) { return filter.smoothed_mus[k]; };
                smoothed_mus[k] = this->weightedStateSum(smoothed_model_probabilities[k], smoothed_mus[k], getSmoothedState);
                smoothed_sigmas[k] = this->weightedCovarianceSum(smoothed_model_probabilities[k], smoothed_mus[k], getSmoothedState, [&k](auto &filter) { return filter.smoothed_sigmas[k]; });
                if (!isPositiveDefinite(smoothed_sigmas[k]))
                    indefinit++;
                if (smoothed_sigmas[k].determinant() < old_sigmas[k].determinant())
                    improved++;
            }
            std::cout << "Indefinit Count: " << indefinit << std::endl;
            std::cout << "Improved Count: " << improved << std::endl;
        }
        /**
         * @brief Smooth all estimates
         * 
         * @tparam Controls Type of the dynamic controls
         * @param all_controls The control inputs of all time steps
         */
        template <typename... Controls>
        void smoothAllWithNonAdditiveNoise(const std::vector<std::tuple<Controls...>> &all_controls)
        {
            smoothIntervalWithNonAdditiveNoise(old_mus.size() - 2, -1, all_controls);
        }
    };

    /**
  * General Deduction Template for the BPIMM based on StateRetriever.
  * This is needed so you can type BPIMM imm(State,COV) without template arguments
  */

    template <typename DERIVED, typename COV_TYPE, typename DYN_COV, typename... Models>
    BPRTSIMM(const DERIVED &, const COV_TYPE &, std::initializer_list<DYN_COV>,
             Models...) -> BPRTSIMM<RTS_Smoother, typename StateInfo<DERIVED>::type, DYN_COV, Models...>;

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
    template <template <typename> class Smoother, typename STATE_TYPE, typename COV_TYPE, typename... Models>
    auto createBPRTSIMM(const STATE_TYPE &mu, const COV_TYPE &sigma, Models... models) -> BPRTSIMM<Smoother, typename StateInfo<STATE_TYPE>::type, Models...>
    {
        return BPRTSIMM<Smoother, typename StateInfo<STATE_TYPE>::type, Models...>(mu, sigma, models...);
    }

}

#endif //BPRTSIMM_H
