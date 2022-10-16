#include <kalman_filter/ukfa.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukfa_t::ukfa_t(uint32_t n_variables, uint32_t n_observers)
    : base_t(n_variables, n_observers)
{
    // Store augmented dimension sizes.
    ukfa_t::n_a = ukfa_t::n_x + ukfa_t::n_x + ukfa_t::n_z;
    ukfa_t::n_s = 1 + 2*ukfa_t::n_a;

    // Allocate weight vector.
    ukfa_t::wj.setZero(ukfa_t::n_s);

    // Allocate prediction components.
    ukfa_t::Xp.setZero(ukfa_t::n_x, ukfa_t::n_x);
    ukfa_t::Xq.setZero(ukfa_t::n_x, ukfa_t::n_x);
    ukfa_t::X.setZero(ukfa_t::n_x, ukfa_t::n_s);
    ukfa_t::dX.setZero(ukfa_t::n_x, ukfa_t::n_s);

    // Allocate update components.
    ukfa_t::Xr.setZero(ukfa_t::n_z, ukfa_t::n_z);
    ukfa_t::Z.setZero(ukfa_t::n_z, ukfa_t::n_s);

    // Allocate temporaries.
    ukfa_t::t_x.setZero(ukfa_t::n_x);
    ukfa_t::t_xs.setZero(ukfa_t::n_x, ukfa_t::n_s);
    ukfa_t::t_zs.setZero(ukfa_t::n_z, ukfa_t::n_s);

    // Set default parameters.
    ukfa_t::wo = 0.1;
}

// FILTER METHODS
void ukfa_t::iterate()
{
    // ---------- STEP 1: PREPARATION ----------

    // Calculate weight vector for mean and covariance averaging.
    ukfa_t::wj.fill((1.0 - ukfa_t::wo)/(2.0 * static_cast<double>(ukfa_t::n_x)));
    ukfa_t::wj[0] = ukfa_t::wo;

    // ---------- STEP 2: PREDICT ----------

    // Calculate sigma matrix.
    // NOTE: This implementation segments out the input sigma matrix for efficiency:
    // [u u+y*sqrt(P) u-y*sqrt(P) 0           0           0           0          ]
    // [0 0           0           u+y(sqrt(Q) u-y*sqrt(Q) 0           0          ]
    // [0 0           0           0           0           u+y*sqrt(R) u-y*sqrt(R)]
    // u is stored in x
    // y*sqrt(P) stored in Xp
    // y*sqrt(Q) stored in Xq
    // y*sqrt(R) stored in Xr.

    // Calculate square root of P using Cholseky Decomposition
    ukfa_t::llt.compute(ukfa_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Fill +sqrt(P) block of Xp.
    ukfa_t::Xp = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xp *= std::sqrt(static_cast<double>(ukfa_t::n_x) / (1.0 - ukfa_t::wo));

    // Calculate square root of Q using Cholseky Decomposition.
    ukfa_t::llt.compute(ukfa_t::Q);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix Q is not positive semi definite");
    }
    // Fill +sqrt(Q) block of Xq.
    ukfa_t::Xq = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xq *= std::sqrt(static_cast<double>(ukfa_t::n_x) / (1.0 - ukfa_t::wo));

    // Calculate square root of R using Cholseky Decomposition.
    ukfa_t::llt.compute(ukfa_t::R);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix R is not positive semi definite");
    }
    // Fill +sqrt(R) block of Xr.
    ukfa_t::Xr = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xr *= std::sqrt(static_cast<double>(ukfa_t::n_x) / (1.0 - ukfa_t::wo));

    // Calculate X by passing sigma points through the transition function.

    // Reset X to zero.
    ukfa_t::X.setZero();

    // Create sigma column index.
    uint32_t s = 0;

    // Pass first set of sigma points, which is just the mean (no Xp or Xq)
    // NOTE: the mean should already be normalized from the prior run.
    ukfa_t::t_x.setZero();
    state_transition(ukfa_t::x, ukfa_t::t_x, ukfa_t::X.col(s++));

    // Pass second set of sigma points, which injects Xp.
    // Use temporary matrix to store mean plus y*sqrt(P).
    ukfa_t::t_xs = ukfa_t::x.replicate(1, ukfa_t::n_s);
    ukfa_t::t_xs += ukfa_t::Xp;
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean PLUS y*sqrt(P)
        // Normalize the state.
        ukfa_t::normalize_state(ukfa_t::t_xs.col(j));
        // Pass state through transition function and increment s.
        state_transition(ukfa_t::t_xs.col(j), ukfa_t::t_x, ukfa_t::X.col(s++));
    }
    // Use temporary matrix to store mean minus y*sqrt(P).
    ukfa_t::t_xs = ukfa_t::x.replicate(1, ukfa_t::n_s);
    ukfa_t::t_xs -= ukfa_t::Xp;
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean MINUS y*sqrt(P)
        // Normalize the state.
        ukfa_t::normalize_state(ukfa_t::t_xs.col(j));
        // Pass state through transition function and increment s.
        state_transition(ukfa_t::t_xs.col(j), ukfa_t::t_x, ukfa_t::X.col(s++));
    }

    // Pass third set of sigma points, which injects Xq.
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean AND positive y*sqrt(Q)
        state_transition(ukfa_t::x, ukfa_t::Xq.col(j), ukfa_t::X.col(s++));
    }
    ukfa_t::t_xs = -ukfa_t::Xq;
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {  
        // mean AND negative y*sqrt(Q)
        state_transition(ukfa_t::x, ukfa_t::t_xs.col(j), ukfa_t::X.col(s++));
    }

    // Pass fourth set of sigma points, which injects Xr.
    // R has no effect on the transition function, so the output sigma matrix
    // just has extra copies of the mean at the end.
    for(;s < ukfa_t::n_s; ++s)
    {
        ukfa_t::X.col(s) = ukfa_t::X.col(0);
    }

    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wj.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wj vector.
    ukfa_t::x.noalias() = ukfa_t::X * ukfa_t::wj;
    // Normalize the state mean.
    ukfa_t::normalize_state(ukfa_t::x);

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    // This can be done more efficiently (speed & code) using (X-x)*wc*(X-x)', where wc is formed into a diagonal matrix.
    ukfa_t::dX = ukfa_t::X - ukfa_t::x.replicate(1, ukfa_t::n_s);
    ukfa_t::t_xs.noalias() = ukfa_t::dX * ukfa_t::wj.asDiagonal();
    ukfa_t::P.noalias() = ukfa_t::t_xs * ukfa_t::dX.transpose();

    // Log predicted state.
    ukfa_t::log_predicted_state();

    // ---------- STEP 3: UPDATE ----------
    
    // Check if update is necessary.
    if(ukfa_t::has_observations())
    {
        // Calculate Z by passing calculated X and Sr.
        // NOTE: X is already normalized.

        // Reset Z to zeros.
        ukfa_t::Z.setZero();

        // Pass the x/Xp/Xq portion of X through.
        for(s = 0; s < 1 + 4 * ukfa_t::n_x; ++s)
        {
            observation(ukfa_t::X.col(s), ukfa_t::t_x, ukfa_t::Z.col(s));
        }

        // Pass Sr through on top of the back of X.
        for(uint32_t j = 0; j < ukfa_t::n_z; ++j)
        {
            // mean AND positive y*sqrt(R)
            observation(ukfa_t::X.col(s), ukfa_t::Xr.col(j), ukfa_t::Z.col(s++));
        }
        ukfa_t::t_xs = -ukfa_t::Xr;
        for(uint32_t j = 0; j < ukfa_t::n_z; ++j)
        {
            // mean AND negative y*sqrt(R)
            observation(ukfa_t::X.col(s), ukfa_t::t_xs.col(j), ukfa_t::Z.col(s++));
        }

        // Calculate predicted observation mean and covariance, as well as cross covariance.
        
        // Predicted observation mean is a weighted average: sum(wj.*Z) over all sigma points.
        // Can be calculated via matrix multiplication with wj vector.
        ukfa_t::z.noalias() = ukfa_t::Z * ukfa_t::wj;

        // Log observations.
        ukfa_t::log_observations();

        // Predicted observation covariance is a weighted average: sum(wc.*(Z-z)(Z-z)') over all sigma points.
        // This can be done more efficiently (speed & code) using (Z-z)*wc*(Z-z)', where wc is formed into a diagonal matrix.
        // Calculate Z-z in place on Z as it's not needed afterwards.
        ukfa_t::Z -= ukfa_t::z.replicate(1, ukfa_t::n_s);
        ukfa_t::t_zs.noalias() = ukfa_t::Z * ukfa_t::wj.asDiagonal();
        ukfa_t::S.noalias() = ukfa_t::t_zs * ukfa_t::Z.transpose();

        // Predicted state/observation cross covariance is a weighted average: sum(wc.*(X-x)(Z-z)') over all sigma points.
        // This can be done more efficiently (speed & code) using (X-x)*wc*(Z-z)', where wc is formed into a diagonal matrix.
        // Recall that (X-x)*wc is currently stored in ukfa_t::t_xs, and Z-z is stored in Z.
        ukfa_t::C.noalias() = ukfa_t::t_xs * ukfa_t::Z.transpose();

        // Run masked Kalman update.
        ukfa_t::masked_kalman_update();
    }
    else
    {
        // Log empty observations.
        ukfa_t::log_observations(true);
    }

    // Log estimated state.
    ukfa_t::log_estimated_state();    
}