#include <cstdlib>
#include <iostream>
#include <string>

#include "gbdt/rdp_accountant.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }

    std::string mode = argv[1];
    std::cout.setf(std::ios::fixed);
    std::cout.precision(12);

    if (mode == "accounting") {
        if (argc != 10) {
            return 2;
        }

        ModelParams params;
        params.newton_boosting = std::atoi(argv[2]);
        params.leaf_denom_noise_weight = std::atof(argv[3]);

        TreeParams tree_params(0.0);
        tree_params.active_subsampling_ratio = std::atof(argv[4]);
        tree_params.active_threshold = std::atof(argv[5]);
        tree_params.hess_active_threshold = std::atof(argv[6]);
        tree_params.leaf_eps = std::atof(argv[7]);

        int compositions = std::atoi(argv[8]);
        double rho_alpha = std::atof(argv[9]);

        RDPAccountant accountant(&params, &tree_params);
        Accounting acc = accountant.setup_accounting(compositions);
        double hess_sens = params.newton_boosting ? tree_params.hess_active_threshold : 1.0;

        std::cout << acc.alpha << " " << acc.noise_scale << " " << acc.max_rho << " "
                  << acc.eps << " "
                  << accountant.gen_rho(rho_alpha, acc.noise_scale, tree_params.active_threshold, hess_sens)
                  << "\n";
        return 0;
    }

    if (mode == "rho") {
        if (argc != 11) {
            return 2;
        }

        ModelParams params;
        params.newton_boosting = std::atoi(argv[2]);
        params.leaf_denom_noise_weight = std::atof(argv[3]);

        TreeParams tree_params(0.0);
        tree_params.active_subsampling_ratio = std::atof(argv[4]);
        tree_params.active_threshold = std::atof(argv[5]);
        tree_params.hess_active_threshold = std::atof(argv[6]);

        double alpha = std::atof(argv[7]);
        double noise_scale = std::atof(argv[8]);
        double ind_g = std::atof(argv[9]);
        double ind_h = std::atof(argv[10]);

        RDPAccountant accountant(&params, &tree_params);
        std::cout << accountant.gen_rho(alpha, noise_scale, ind_g, ind_h) << "\n";
        return 0;
    }

    if (mode == "approx") {
        if (argc < 11 || ((argc - 9) % 2 != 0)) {
            return 2;
        }

        ModelParams params;
        params.newton_boosting = std::atoi(argv[2]);
        params.leaf_denom_noise_weight = std::atof(argv[3]);

        TreeParams tree_params(0.0);
        tree_params.active_subsampling_ratio = std::atof(argv[4]);
        tree_params.active_threshold = std::atof(argv[5]);
        tree_params.hess_active_threshold = std::atof(argv[6]);
        tree_params.rdp_alpha = std::atof(argv[7]);
        tree_params.active_noise_scale = std::atof(argv[8]);

        RDPAccountant accountant(&params, &tree_params);
        accountant.setup_approximation();

        for (int i = 9; i < argc; i += 2) {
            double ind_g = std::atof(argv[i]);
            double ind_h = std::atof(argv[i + 1]);
            double exact = accountant.gen_rho(ind_g, ind_h);
            double approx = accountant.approximate_rho(ind_g, ind_h);
            std::cout << ind_g << " " << ind_h << " " << exact << " " << approx << "\n";
        }
        return 0;
    }

    return 2;
}
