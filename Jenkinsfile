@Library("vivarium_build_utils") _
reusable_pipeline(scheduled_branches: ["main"],
    test_types: ["unit", "integration"],
    requires_slurm: true, 
    skip_doc_build: true,
    upstream_repos: ["vivarium", "vivarium_inputs", "vivarium_public_health", "vivarium_cluster_tools", "gbd_mapping", "pseudopeople"],
    run_mypy: false,
)
