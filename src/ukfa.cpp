#include <kalman_filter/ukfa.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukfa::ukfa(uint32_t n_variables, uint32_t n_observers)
    : base(n_variables, n_observers)
{
    // Store augmented dimension sizes.
    ukfa::n_a = ukfa::n_x + ukfa::n_x + ukfa::n_z;
    ukfa::n_s = 1 + 2*ukfa::n_a;

    // Allocate weight vector.
    ukfa::wj.setZero(ukfa::n_s);

    // Allocate prediction components.
    ukfa::Xp.setZero(ukfa::n_x, ukfa::n_x);
    ukfa::Xq.setZero(ukfa::n_x, ukfa::n_x);
    ukfa::X.setZero(ukfa::n_x, ukfa::n_s);
    ukfa::dX.setZero(ukfa::n_x, ukfa::n_s);

    // Allocate update components.
    ukfa::Xr.setZero(ukfa::n_z, ukfa::n_z);
    ukfa::Z.setZero(ukfa::n_z, ukfa::n_s);

    // Allocate interface components.
    ukfa::i_xp.setZero(ukfa::n_x);
    ukfa::i_x.setZero(ukfa::n_x);
    ukfa::i_q.setZero(ukfa::n_x);
    ukfa::i_r.setZero(ukfa::n_z);
    ukfa::i_z.setZero(ukfa::n_z);

    // Allocate temporaries.
    ukfa::t_xs.setZero(ukfa::n_x, ukfa::n_s);
    ukfa::t_zs.setZero(ukfa::n_z, ukfa::n_s);

    // Set default parameters.
    ukfa::wo = 0.1;
}

// FILTER METHODS
void ukfa::iterate()
{
    // ---------- STEP 1: PREPARATION ----------

    // Calculate weight vector for mean and covariance averaging.
    ukfa::wj.fill((1.0 - ukfa::wo)/(2.0 * static_cast<double>(ukfa::n_x)));
    ukfa::wj[0] = ukfa::wo;

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
    ukfa::llt.compute(ukfa::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Fill +sqrt(P) block of Xp.
    ukfa::Xp = ukfa::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa::Xp *= std::sqrt(static_cast<double>(ukfa::n_x) / (1.0 - ukfa::wo));

    // Calculate square root of Q using Cholseky Decomposition.
    ukfa::llt.compute(ukfa::Q);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix Q is not positive semi definite");
    }
    // Fill +sqrt(Q) block of Xq.
    ukfa::Xq = ukfa::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa::Xq *= std::sqrt(static_cast<double>(ukfa::n_x) / (1.0 - ukfa::wo));

    // Calculate square root of R using Cholseky Decomposition.
    ukfa::llt.compute(ukfa::R);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix R is not positive semi definite");
    }
    // Fill +sqrt(R) block of Xr.
    ukfa::Xr = ukfa::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa::Xr *= std::sqrt(static_cast<double>(ukfa::n_x) / (1.0 - ukfa::wo));

    // Calculate X by passing sigma points through the transition function.

    // Create sigma column index.
    uint32_t s = 0;

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukfa::i_xp = ukfa::x;
    ukfa::i_q.setZero(ukfa::n_x);
    ukfa::i_x.setZero(ukfa::n_x);
    // Run transition function.
    state_transition(ukfa::i_xp, ukfa::i_q, ukfa::i_x);
    // Capture output into X.
    ukfa::X.col(s++) = ukfa::i_x;

    // Pass second set of sigma points, which injects Xp.
    for(uint32_t j = 0; j < ukfa::n_x; ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukfa::i_xp = ukfa::x + ukfa::Xp.col(j);
        ukfa::i_q.setZero(ukfa::n_x);
        ukfa::i_x.setZero(ukfa::n_x);
        // Run transition function.
        state_transition(ukfa::i_xp, ukfa::i_q, ukfa::i_x);
        // Capture output into X.
        ukfa::X.col(s++) = ukfa::i_x;
    }
    for(uint32_t j = 0; j < ukfa::n_x; ++j)
    {
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukfa::i_xp = ukfa::x - ukfa::Xp.col(j);
        ukfa::i_q.setZero(ukfa::n_x);
        ukfa::i_x.setZero(ukfa::n_x);
        // Run transition function.
        state_transition(ukfa::i_xp, ukfa::i_q, ukfa::i_x);
        // Capture output into X.
        ukfa::X.col(s++) = ukfa::i_x;
    }

    // Pass third set of sigma points, which injects Xq.
    for(uint32_t j = 0; j < ukfa::n_x; ++j)
    {
        // mean PLUS y*sqrt(Q)
        // Populate interface vectors.
        ukfa::i_xp = ukfa::x;
        ukfa::i_q = ukfa::Xq.col(j);
        ukfa::i_x.setZero(ukfa::n_x);
        // Run transition function.
        state_transition(ukfa::i_xp, ukfa::i_q, ukfa::i_x);
        // Capture output into X.
        ukfa::X.col(s++) = ukfa::i_x;
    }
    for(uint32_t j = 0; j < ukfa::n_x; ++j)
    {  
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukfa::i_xp = ukfa::x;
        ukfa::i_q = -ukfa::Xq.col(j);
        ukfa::i_x.setZero(ukfa::n_x);
        // Run transition function.
        state_transition(ukfa::i_xp, ukfa::i_q, ukfa::i_x);
        // Capture output into X.
        ukfa::X.col(s++) = ukfa::i_x;
    }

    // Pass fourth set of sigma points, which injects Xr.
    // R has no effect on the transition function, so the output sigma matrix
    // just has extra copies of the mean at the end.
    for(;s < ukfa::n_s; ++s)
    {
        ukfa::X.col(s) = ukfa::X.col(0);
    }

    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukfa::x.noalias() = ukfa::X * ukfa::wj;

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    // This can be done more efficiently (speed & code) using (X-x)*wc*(X-x)', where wc is formed into a diagonal matrix.
    ukfa::dX = ukfa::X - ukfa::x.replicate(1, ukfa::n_s);
    ukfa::t_xs.noalias() = ukfa::dX * ukfa::wj.asDiagonal();
    ukfa::P.noalias() = ukfa::t_xs * ukfa::dX.transpose();

    // Log predicted state.
    ukfa::log_predicted_state();

    // ---------- STEP 3: UPDATE ----------
    
    // Check if update is necessary.
    if(ukfa::has_observations())
    {
        // Calculate Z by passing calculated X and Sr.

        // Pass the x/Xp/Xq portion of X through.
        for(s = 0; s < 1 + 4 * ukfa::n_x; ++s)
        {
            // Populate interface vectors.
            ukfa::i_x = ukfa::X.col(s);
            ukfa::i_r.setZero(ukfa::n_z);
            ukfa::i_z.setZero(ukfa::n_z);
            // Run observation function.
            observation(ukfa::i_x, ukfa::i_r, ukfa::i_z);
            // Capture output into Z.
            ukfa::Z.col(s) = ukfa::i_z;
        }

        // Pass Sr through on top of the back of X.
        for(uint32_t j = 0; j < ukfa::n_z; ++j)
        {
            // mean PLUS y*sqrt(R)
            // Populate interface vectors.
            ukfa::i_x = ukfa::X.col(s);
            ukfa::i_r = ukfa::Xr.col(j);
            ukfa::i_z.setZero(ukfa::n_z);
            // Run observation function.
            observation(ukfa::i_x, ukfa::i_r, ukfa::i_z);
            // Capture output into Z.
            ukfa::Z.col(s++) = ukfa::i_z;
        }
        for(uint32_t j = 0; j < ukfa::n_z; ++j)
        {
            // mean MINUS y*sqrt(R)
            // Populate interface vectors.
            ukfa::i_x = ukfa::X.col(s);
            ukfa::i_r = -ukfa::Xr.col(j);
            ukfa::i_z.setZero(ukfa::n_z);
            // Run observation function.
            observation(ukfa::i_x, ukfa::i_r, ukfa::i_z);
            // Capture output into Z.
            ukfa::Z.col(s++) = ukfa::i_z;
        }

        // Calculate predicted observation mean and covariance, as well as cross covariance.
        
        // Predicted observation mean is a weighted average: sum(wm.*Z) over all sigma points.
        // Can be calculated via matrix multiplication with wm vector.
        ukfa::z.noalias() = ukfa::Z * ukfa::wj;

        // Log observations.
        ukfa::log_observations();

        // Predicted observation covariance is a weighted average: sum(wc.*(Z-z)(Z-z)') over all sigma points.
        // This can be done more efficiently (speed & code) using (Z-z)*wc*(Z-z)', where wc is formed into a diagonal matrix.
        // Calculate Z-z in place on Z as it's not needed afterwards.
        ukfa::Z -= ukfa::z.replicate(1, ukfa::n_s);
        ukfa::t_zs.noalias() = ukfa::Z * ukfa::wj.asDiagonal();
        ukfa::S.noalias() = ukfa::t_zs * ukfa::Z.transpose();

        // Predicted state/observation cross covariance is a weighted average: sum(wc.*(X-x)(Z-z)') over all sigma points.
        // This can be done more efficiently (speed & code) using (X-x)*wc*(Z-z)', where wc is formed into a diagonal matrix.
        // Recall that (X-x)*wc is currently stored in ukfa::t_xs, and Z-z is stored in Z.
        ukfa::C.noalias() = ukfa::t_xs * ukfa::Z.transpose();

        // Run masked Kalman update.
        ukfa::masked_kalman_update();
    }
    else
    {
        // Log empty observations.
        ukfa::log_observations(true);
    }

    // Log estimated state.
    ukfa::log_estimated_state();    
}