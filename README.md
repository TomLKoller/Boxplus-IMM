# Boxplus-IMM ([+]-IMM)
This repository contains the [+]-IMM. A generic interacting multiple model filter which can handle manifold structures  (e.g. quaternions) in the state space. 
It can be used in vector only mode aswell.  This Readme is not intended to explain the [+]-IMM. Please refer to the publication:
"The Interacting Multiple Model Filter on Boxplus-Manifolds" -Tom Koller and Udo Frese; Septembre 2020.

In addition, it also contains the Boxplus-IMM Smoother which is described in the next section.

## Installation 
### Requirements
Download and install the requirements of ADEKF and ADEKF_VIZ:
1. GCC >= 7  (or other c++17 compiler)        
1. Cmake >= 3.0       
1. Eigen 3                  http://eigen.tuxfamily.org/index.php?title=Main_Page
1. Boost >= 1.65.1          https://www.boost.org/
1. QT5                      https://www.qt.io/
1. JKQT Plotter             https://github.com/jkriege2/JKQtPlotter.git
1. VTK                      https://vtk.org/


5.-7. Are needed for visualization. If not wanted, remove everything from ADEKF_VIZ from your stack.

### Build
Clone the repository with submodules:

```
git clone --recursive git@github.com:TomLKoller/Boxplus-IMM.git
```


Build the Code with cmake:
```
mkdir build ; cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make -j4
```


## Run the example code
```
./RadarFlightExample
```


## Usage of the [+]-IMM
The [+]-IMM is build upon the ADEKF. Hence, it can be helpful to read https://github.com/TomLKoller/ADEKF. The website gives a more detailed explanation of the State representation and how to define models.
For an example look into immBodyDistance.cpp
### State representation
The state can be composed of manifolds and vectors. Currently a quaternion representation of SO3 and a matrix representation of SO2 are available as manifolds.
You can declare a compound state with the macro: 

```c++
ADEKF_MANIFOLD(TYPE_NAME,((MAN_TYPE_1,man_name_1))((MAN_TYPE_2,man_name_2))...,(VECTOR_SIZE_1,vector_name_1),(VECTOR_SIZE_2,vector_name_2),...)
e.g.
ADEKF_MANIFOLD(Pose3D,((adekf::SO3,orientation)),(3,position),(3,velocity),(3,acc_bias))
```
This macro will create a type TYPE_NAME with attributes man_name_1, man_name_2,... which refer to the given manifold types and with vectors called vector_name1, ... with the given sizes.
You can pass an unlimited (including 0) amount  of manifolds or vectors.
Be careful that manifolds are declared by ((type,name)) with double parenthesis and no comma between two manifolds
whereas vectors are declared as (size,name) with single parenthesis and commas between the vectors.

### Models
Models can be declared either as lambdas or as structs with an operator().
#### Lambda Declaration:
You can create models as lambdas e.g.:
```c++
auto dyn_model=[](auto & state, auto velocity, double time_step){
state.position+=velocity*time_step;
};
```
The first argument is the state. The dynamic model has to write its changes to the passed state (dont forget the & at the declaration).
All following arguments can be set arbitrary and are the input parameters. The generic header is:
```c++
auto dyn_model=[](auto & state, INPUT_ARGS ... inputs){
...
}
```
If you want to use the non additive noise variant the second argument is the noise vector:
```c++
auto dyn_model=[](auto & state, auto noise, INPUT_ARGS ... inputs){
...
}
```
use the macro NOISE(start,size) to retrieve a segment of noise (be carefull to name the noise argument noise or use NOISE(NOISE_NAME,start,size))

The measurement models are declared similarly but they require a return value:
```c++
auto meas_model=[](auto  state,  INPUT_ARGS ... inputs){
return function_of(state,inputs...);
}
```
The return value is deduced automatically and has to be either an Eigen::Matrix or a Manifold.
Please read the pittfalls with lambdas page at  https://github.com/TomLKoller/ADEKF

#### Struct Declaration
If you want to use structs instead of lambdas you have to declare them as:
```c++
struct dynamic_model{
    template<typename T>
    void operator()(STATE_TYPE<T> &state, (const Eigen::Matrix<T,NOISE_SIZE,1> &noise), ParameterPack ... & params){
        state= ... //Implement dynamic model
    }
};
```
Where you can set arbitrary parameters (or no parameter) for params. Noise is only required if you want to use non-additive noise. Be carefull to use "&" to not copy the input parameters.
For measurement models use :
```c++
struct measurement_model{
    template<typename T>
    MEASUREMENT_TYPE operator()(const STATE_TYPE<T> & state, ParameterPack ... & params){
        return  ... //Implement measurement model
    }
};
```

The state needs to be templated with the scalar type T to use the automatic differentiation framework.

### Setup of the [+]-IMM
When initialising the [+]-IMM you have to pass start state, covariance, the dynamic models and the dynamic covariances. 
You can pass an arbitrary amount of dynamic models with corresponding covariances. 
Important: The dynamic covariances need to have the same size.
 They must have the size of the state covariance to use with additive noise
 or there size defines the size of the noise vector in non-additive mode.
Call:
```c++
adekf::BPIMM imm{start_state, start_cov, std::initializer_list<DYN_COV_TYPE>{cov1,cov2,...}, dyn_model_1, dyn_model2,...};
```
The dynamic covariances have to be past in an initializer list which can be constructed by {cov1,cov2,...}.
The template parameters are deduced automatically. You do not have to pass the same dynamic models twice (unless you want different covariances) since you add internal Filters by:
```c++
imm.addFilter({0,1});
```
where the numbers refer to the order in which the dynamic models were passed to the constructor. 
This allows you to run multiple filters with the same dynamic model. This has to be called since the constructor does not add any filter.

Now set the transition and start probabilities of the mode by e.g.:
```c++
    Eigen::Matrix<double, 2, 2> t_prob;
    t_prob << 0.95, 0.05,
            0.05, 0.95;
    imm.setTransitionProbabilities(t_prob);
    Eigen::Vector2d start_prob(0.5, 0.5);
    imm.setStartProbabilities(start_prob);
```


### Running the Filter steps
To run the [+]-IMM call the interaction, prediction, update and combination step:
```c++
 imm.interaction();
 imm.predictWithNonAdditiveNoise(inputs ...);
  //or call the simple predict with additive noise:
 imm.predict(inputs ...)  
 imm.update(meas_model, meas_sigma, target,meas_inputs ...);
 imm.combination();
```
The passed inputs to update and predict have to match the signature of the passed models. 
meas_sigma has to match the DOF of the returned measurement. target is the real measurement.
You can read the state of the imm via:
```c++
std::cout << imm.mu << std::endl;
std::cout << imm.sigma << std::endl;
```




# Boxplus-IMM-Smoother ([+]-RTSIMMS)
The [+]-RTSIMMS adds smoothing capability to the [+]-IMM. It builds upon the [+]-EKS (https://github.com/TomLKoller/Manifold-RTS-Smoother).

A publication is on its way.

## Installation 
Follow the installation instructions for the [+]-IMM and the [+]-EKS.

## Run the example code
```
./SmoothRadarFlightExample
```
## Usage
The [+]-RTSIMMS uses a forward filtering of the state followed by a backwards smoother. It makes use of the C++ lambda syntax to define models easily.
It is a child class of the [+]-IMM wherefore most of the Syntax can be reused.


### Initialization/Setup
The [+]-RTSIMMS has to be initialized exactly as the [+]-IMM
### Forward Filtering
The forward filtering can be implemented almost as in the [+]-IMM.
It is required to store the predicted and updated states. The predicted states are stored automatically.  The [+]-RTSIMMS can store the updated estimates by calling  storeEstimation() after the combination step. 
The controls of the dynamic models are required to smooth the estimate later. You can have an arbitrary number of control inputs. Store them during the forward pass in a vector of tuples:
```c++
//Before loop: Setup of Models 
std::vector<std::tuple<ControlTypeA,ControlTypeB>> all_controls;

//One iteration of the loop
ControlTypeA a=...;
ControlTypeB b=...;
rtsimms.interaction();
rtsimms.predictWithNonAdditiveNoise(a,b);
all_controls.emplace_back(a,b); //Automatically constructs the tuple of the controls
//update
Measurement measurement=...;
rtsimms.update(measurementModel,measurement_noise,measurement);
rtsimms.combination();
rtsimms.storeEstimation();
```
The stored estimates are available through std::vectors named:
1.old_mus (state after update, old_mus[0] is the start state)
2.old_sigmas (covariance after update)
3.predicted_mus (state after predict)
4.predicted_sigmas (covariance after predict)

### Backwards Smoothing
Two options are available for smoothing:
1. smoothIntervalWithNonAdditiveNoise smoothes a given intervall 
2. smoothAllWithNonAdditiveNoise smoothes all filtered states

Both require the stored control values:
```c++
rtsimms.smoothIntervalWithNonAdditiveNoise(steps, start,all_controls);
// or
rtsimms.smoothAllWithNonAdditiveNoise(all_controls);
```
The smoothed values are available through std::vectors named:
1. smoothed_mus (the smoothed states)
2. smoothed_sigmas (the smoothed covariances)

Before smoothing, the vector elements are the same as old_mus.

The storage of old estimates and controls is subject to change in future versions.


# Support
On request, i can provide minimal examples. 

Please create an Issue  if this short Readme is insufficient. 
I will adjust it if its necessary. 



