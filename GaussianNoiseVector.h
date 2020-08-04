//
// Created by tomlucas on 02.04.20.
//

#ifndef ADEKF_GAUSSIANNOISEVECTOR_H
#define ADEKF_GAUSSIANNOISEVECTOR_H

#include <Eigen/Core>
#include <random>


    /**
     * Class to create noise vectors with a gaussian random distribution.
     *
     * Use it like a usual eigen vector in computations. Call poll() to sample a new random vector.
     * @tparam Scalar type of the vector
     * @tparam size size of the vector
     */
    template<int size>
    class GaussianNoiseVector : public Eigen::Matrix<double, size, 1> {
        using Scalar=double;
        //Type of the vector
        typedef Eigen::Matrix<Scalar, size, 1> BASE;
        //generator for random numbers.
        static inline std::default_random_engine generator;
        //normal distribution to sample from
        std::vector<std::normal_distribution<Scalar>> distributions;
    public:
        //Mean and standard deviation of the gaussian distribution
        const Scalar mu;
        std::array<Scalar,size> sigmas;

        /**
         * Initialises the vectors random distribution and assigns a random vector
         * @param mu mean of the gaussian
         * @param sigma standard deviation of the gaussian
         */
         template<typename ... Sigmas>
        GaussianNoiseVector(Scalar mu, Sigmas ... _sigmas) :mu(mu), sigmas({_sigmas ...})  {
            for(Scalar sigma: sigmas){
                distributions.push_back(std::normal_distribution<Scalar>(mu,sigma));
            }
            this->poll();
        }
        /**
         * Samples a new gaussian random vector into this and also returns this
         */
        BASE poll() {
            for(size_t t=0; t< size; t++){
                (*this)(t)=distributions[t](generator);
            }
            return *this;
        }

        /**
         * Get the covariance matrix of this vector.
         * @return  The diagonal covariance matrix corresponding to the noise _sigmas.
         */
        Eigen::Matrix<double,size,size> getCov(){
            Eigen::Matrix<double,size,1> diag;
            for(size_t t=0; t < size;t++){
                diag(t)=pow(sigmas[t],2);
            }
            return diag.asDiagonal();
        }


    };
    template<typename ... Sigmas>
    GaussianNoiseVector(double, Sigmas ... ) ->GaussianNoiseVector<sizeof...(Sigmas)>;

#endif //ADEKF_GAUSSIANNOISEVECTOR_H
