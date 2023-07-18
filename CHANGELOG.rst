**1.3.0 - 07/18/23**

 - Change w2/1099 column name from "income" to "wages"

**1.2.2 - 07/14/23**

 - Bugfix SSN creation date logic

**1.2.1 - 07/13/23**

 - Copy the CHANGELOG to final results directory
 - Bugfix validation user-provided version label
 - Refactor/bugfix symlink generation
 - Modify behavior to raise even when dropped into debugger with --pdb

**1.2.0 - 07/12/23**

 - Add this CHANGELOG
 - Minor refactor of age perturbation methods for readability
 - Bugfix set the SSA creation event date to date of birth for all simulants

**1.1.0 - 07/05/23**

 - Support for vivarium v1.1.0
 - Include household_id column where appropriate
 - Implement post-processing versioning
 - Implement changes required for pseudopeople noising copy_from_household_member
 - Bugfix increase IndexMap size
 - Remove "Opp-sex" partners from 1040 observer and include relation_to_reference_person in output
 - Bugfix use categorical types for all states
 - Implement nickname proportions for pseudopeople noising use_nickname
 
**1.0.0 - 04/20/23**

 - Initial release for CODS demo