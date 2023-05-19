[1mdiff --git a/src/vivarium_census_prl_synth_pop/constants/data_values.py b/src/vivarium_census_prl_synth_pop/constants/data_values.py[m
[1mindex 82e883f..9211c28 100644[m
[1m--- a/src/vivarium_census_prl_synth_pop/constants/data_values.py[m
[1m+++ b/src/vivarium_census_prl_synth_pop/constants/data_values.py[m
[36m@@ -96,8 +96,8 @@[m [mYEARLY_JOB_CHANGE_RATE = 0.5  # 50 changes per 100 py[m
 [m
 BUSINESS_MOVE_RATE_YEARLY = 0.1  # 10 changes per 100 py[m
 [m
[31m-PERSONAL_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.812309 ** 0.5)[m
[31m-EMPLOYER_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.187691 ** 0.5)[m
[32m+[m[32mPERSONAL_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.812309**0.5)[m
[32m+[m[32mEMPLOYER_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.187691**0.5)[m
 [m
 [m
 class KnownEmployer(NamedTuple):[m
[1mdiff --git a/src/vivarium_census_prl_synth_pop/tools/cli.py b/src/vivarium_census_prl_synth_pop/tools/cli.py[m
[1mindex 43a265c..ffa8216 100644[m
[1m--- a/src/vivarium_census_prl_synth_pop/tools/cli.py[m
[1m+++ b/src/vivarium_census_prl_synth_pop/tools/cli.py[m
[36m@@ -183,7 +183,10 @@[m [mdef make_results([m
         if expected_version_format.match(version):[m
             pass[m
         else:[m
[31m-            raise ValueError(f"{version} is not of correct format.")[m
[32m+[m[32m            raise ValueError([m
[32m+[m[32m                f"'{version}' is not of correct format. "[m
[32m+[m[32m                "Format for version should be 'v#.#.#'"[m
[32m+[m[32m            )[m
     raw_output_dir, final_output_dir = build_final_results_directory(output_dir, version)[m
     cluster_requests = {[m
         "queue": queue,[m
