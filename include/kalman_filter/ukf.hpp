/// \file kalman_filter/ukf.hpp
/// \brief Defines the kalman_filter::ukf class.
#ifndef KALMAN_FILTER___UKF_H
#define KALMAN_FILTER___UKF_H

#include <kalman_filter/base.hpp>

namespace kalman_filter {

/// \brief An Unscented Kalman Filter (UKF)
/// \details The UKF can perform nonlinear state estimation with additive noise.
class ukf
    : public base
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new ukf object.
    /// \param n_variables The number of variables in the state vector.
    /// \param n_observers The number of state observers.
    ukf(uint32_t n_variables, uint32_t n_observers);

    // MODEL FUNCTIONS
    /// \brief Predicts a new state by transitioning from a prior state.
    /// \param xp The prior state to transition from.
    /// \param x (OUTPUT) The predicted new state.
    /// \note This function must not make changes to any external object.
    virtual void state_transition(const Eigen::VectorXd& xp, Eigen::VectorXd& x) const = 0;
    /// \brief Predicts an observation from a state.
    /// \param x The state to predict an observation from.
    /// \param z (OUTPUT) The predicted observation.
    /// \note This function must not make changes to any external object.
    virtual void observation(const Eigen::VectorXd& x, Eigen::VectorXd& z) const = 0;

    // FILTER METHODS
    void iterate() override;

    // PARAMETERS
    /// \brief Controls sigma point spread from the mean (-1 < wo < 1)
    /// \details wo < 0 gives points closer to the mean, wo > 0 gives points further from the mean.
    double_t wo;

private:
    // DIMENSIONS
    /// \brief The number of sigma points.
    uint32_t n_s;

    // STORAGE: WEIGHTS
    /// \brief The mean/covariance recovery weight vector.
    Eigen::VectorXd wj;

    // STORAGE: SIGMA
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;
    /// \brief The evaluated observation sigma matrix.
    Eigen::MatrixXd Z;

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;
    /// \brief An interface to the predicted observation vector.
    Eigen::VectorXd i_z;

    // STORAGE: TEMPORARIES
    /// \brief A temporary working matrix of size x,s.
    Eigen::MatrixXd t_xs;
    /// \brief A temporary working matrix of size z,s.
    Eigen::MatrixXd t_zs;

    // UTILITY
    /// \brief An LLT object for storing results of Cholesky decompositions.
    mutable Eigen::LLT<Eigen::MatrixXd> llt;

    // Hide base class protected members.
    // NOTE: State variable and covariance access is still protected.
    using base::n_x;
    using base::n_z;
    using base::z;
    using base::S;
    using base::C;
    using base::t_xx;
    using base::has_observations;
    using base::masked_kalman_update;
};

}

#endif