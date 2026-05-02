# PyMOL script: structural visuals for pilot case
# Input structures:
# - WT experimental: 2OV0 chain A
# - Mutant experimental: 2QDV chain A
# - Predicted mutant: rank_001 model from ColabFold

reinitialize
bg_color white
set antialias, 2
set ray_shadows, 0
set orthoscopic, on
set cartoon_fancy_helices, on
set cartoon_sampling, 14

load ../../struct_benchmark_runs/real_colabfold_pilot/run_out/experimental_structures/2ov0_A.pdb, wt_exp
load ../../struct_benchmark_runs/real_colabfold_pilot/run_out/experimental_structures/2qdv_A.pdb, mut_exp
load ../../struct_benchmark_runs/real_colabfold_pilot/run_out/predicted_structures/2ov0_2qdv_M51A/mutant/mut_2qdv_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, mut_pred

hide everything
show cartoon, wt_exp or mut_exp or mut_pred

color gray70, wt_exp
color marine, mut_exp
color salmon, mut_pred

align mut_pred and name CA, mut_exp and name CA
align wt_exp and name CA, mut_exp and name CA

set cartoon_transparency, 0.35, wt_exp
set cartoon_transparency, 0.0, mut_exp
set cartoon_transparency, 0.0, mut_pred

select mut_site_wt, wt_exp and chain A and resi 51
select mut_site_exp, mut_exp and chain A and resi 51
select mut_site_pred, mut_pred and chain A and resi 51

show sticks, mut_site_wt or mut_site_exp or mut_site_pred
set stick_radius, 0.20
color gray30, mut_site_wt
color blue, mut_site_exp
color red, mut_site_pred

label mut_site_exp and name CA, "M51A site"

orient mut_exp
zoom all, 2.0
ray 2200, 1600
png pymol_superposition_overview.png, dpi=300

hide labels
zoom mut_site_wt or mut_site_exp or mut_site_pred, 9.0
set depth_cue, 0
ray 2200, 1600
png pymol_mutation_site_closeup.png, dpi=300

# Confidence-focused predicted structure visual (pLDDT coloring for AF model)
hide everything
show cartoon, mut_pred
spectrum b, blue_white_red, mut_pred, minimum=50, maximum=95
set cartoon_putty, 1, mut_pred
set cartoon_transparency, 0.05, mut_pred
orient mut_pred
zoom mut_pred, 1.8
ray 2200, 1600
png pymol_predicted_confidence_style.png, dpi=300

quit
