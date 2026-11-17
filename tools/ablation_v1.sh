#!/bin/bash
# Experiment 1: where does ReNF's accuracy come from?  (run from the repo root)
# Every arm = official per-horizon config + ONE switch. Same seeds, same early stopping.
#   base      official
#   fb0       stages/losses/widths unchanged, but zeros are fed instead of earlier forecasts
#   nodetach  feedback keeps its gradient (alpha detaches it officially)
#   last      only the final stage is supervised (same total loss weight)
#   uniform   equal stage weights instead of 20/(j+1) (same total)
#   single    one stage (d_layers 1): no cascade at all  -> fewer params, see params_M
#   noema     no EMA weights (r_ema 0: checkpoint = raw weights)
#   l2        MSE training loss instead of L1
# Every run also prints test-time probes (stage average, feedback zeroed / shuffled).
# Usage: DS="etth1 etth2" SEEDS="2021 2022 2023" bash tools/ablation_v1.sh
DS="${DS:-etth1 etth2}"
SEEDS="${SEEDS:-2021 2022 2023}"
B="python tools/renf_jobs.py build --datasets $DS --seeds $SEEDS --append"
$B --tag base
$B --tag fb0      --extra "--fb_mode zero"
$B --tag nodetach --extra "--fb_detach 0"
$B --tag last     --extra "--stage_weight last"
$B --tag uniform  --extra "--stage_weight uniform"
$B --tag single   --extra "--d_layers 1"
$B --tag noema    --extra "--r_ema 0.0"
$B --tag l2       --extra "--time_loss l2"
