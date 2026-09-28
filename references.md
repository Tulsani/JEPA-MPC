# References

Working bibliography for the two-clock hierarchical JEPA project. Each entry
notes why it matters for our positioning. "Verified" means we read the paper or
code during planning (Sep 2026); "to verify" means details come from secondary
sources and must be checked before citing specifics.

## Directly competing: hierarchical planning with JEPA-style latent world models

| Work | Link | Subgoal timing | Relevance | Status |
|---|---|---|---|---|
| **FF-JEPA** — Long-Horizon Planning in World Models with Latent Planners (Masip, Swinnen, Hu, Detry, Tuytelaars) | https://arxiv.org/html/2606.09311v1 | Fixed stride H=25 env steps; replans every 25 steps | Frozen LeWM low level + action-free subgoal predictor (deterministic or diffusion). Push-T d=75: 88.7–91.8% vs flat LeWM 3.5%. Main "fixed stride" foil. | Verified (summary) |
| **HWM** — Hierarchical Planning with Latent World Models | https://arxiv.org/html/2604.03208v1 (v2: https://arxiv.org/html/2604.03208v2) | Waypoints sampled uniformly at random during training (variable-length segments) | High-level model over latent macro-actions; low-dim macro-actions bias toward reachable subgoals. Push-T d=75: 17% → 61%, ~3× less planning compute. Names "feedback across levels" and "uncertainty-aware planning" as open problems. Main "random waypoint" foil. | Verified (summary) |
| **Hi-LeWM** — Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel | https://arxiv.org/html/2607.12547v2 | Fixed stage duration (staged execution) or periodic online replanning | Naive hierarchy *underperforms* flat LeWM (38.7% vs 52.7% at d=50) because CEM exploits OOD macro-actions; fix = empirical-macro CEM around training anchors. Flat LeWM Push-T: 94.0 / 52.7 / 18.0% at d=25/50/75. Also reports OGBCube-v0. Explicitly says subgoal execution timing matters. | Verified (summary) |
| **SAGE** — Subgoal-Conditioned Action Generation for Latent World Model Planning | https://arxiv.org/pdf/2607.17973 | Fixed stride (per quick read) | Push-T + OGBench. | To verify |
| **Temporal-Distance-JEPA** — Plan-Aware Representation Learning for Latent World Model Predictive Control | https://arxiv.org/html/2607.25337v2 | n/a (representation) | Planning-aware latent geometry; relevant to "latent distance is not task cost". | To verify |

## Backbone, data, and evaluation infrastructure

| Work | Link | Relevance |
|---|---|---|
| **LeWorldModel (LeWM)** — Stable End-to-End JEPA from Pixels (Maes et al., 2026) | https://arxiv.org/abs/2603.19312 · code https://github.com/lucas-maes/le-wm | Our frozen encoder. ViT-tiny/14 @224, 192-d CLS embedding + MLP projector, SIGReg regularizer, frameskip 5. Checkpoints: `quentinll/lewm-pusht`, `quentinll/lewm-cube`, `quentinll/lewm-tworooms`, `quentinll/lewm-reacher` (HF). |
| **stable-worldmodel** (v0.1.1) | https://github.com/galilai-group/stable-worldmodel | Environments (`swm/PushT-v1`, `swm/OGBCube-v0`), HDF5 datasets, CEM solver, `World.evaluate` closed-loop harness with dataset goal offsets. |
| LeWM datasets | https://huggingface.co/datasets/quentinll/lewm-pusht · https://huggingface.co/datasets/quentinll/lewm-cube | `pusht_expert_train.h5` (13 GB zst), `cube_single_expert` (46 GB zst). |
| **DINO-WM** — World Models on Pre-trained Visual Features enable Zero-shot Planning (Zhou et al., 2024) | https://arxiv.org/abs/2411.04983 · https://github.com/gaoyuezhou/dino_wm | Frozen DINOv2 features + planning; origin of the Push-T goal-offset protocol. Data: https://osf.io/bmw48/ |
| **PLDM** — planning with latent dynamics models (Sobal et al.) | via LeWM baselines | Reconstruction-free baseline family; includes the two-room environment. |
| **OGBench** (Park et al., 2024) | https://github.com/seohongpark/ogbench | Stretch domain: `cube-single` (manipulation) and `pointmaze` (navigation). |
| Diffusion Policy Push-T data (`pusht_cchi_v7_replay.zarr`) | https://diffusion.cs.columbia.edu/data/training/pusht.zip | Small human-demo set; superseded by LeWM data for comparability. |

## Learned temporal abstraction (closest prior mechanisms)

| Work | Link | Relation to our idea |
|---|---|---|
| **THICK** — Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics (Gumbsch, Sajid, Martius, Butz; ICLR 2024) | https://openreview.net/forum?id=5qappsbO73r · https://github.com/CognitiveModeling/THICK | **Closest prior work.** Sparsely-changing context latents define when the high level ticks; RSSM/Dreamer-based with reconstruction. We differ: reconstruction-free JEPA, gate used at planning time for subgoal persistence and low-level horizon, physical-contact alignment analysis. Must cite prominently. |
| **HM-RNN** — Hierarchical Multiscale Recurrent Neural Networks (Chung, Ahn, Bengio; ICLR 2017) | https://arxiv.org/abs/1609.01704 | Learned boundary detectors between RNN layers; ancestor of our gate. |
| **VTA** — Variational Temporal Abstraction (Kim, Ahn, Bengio; NeurIPS 2019) | https://arxiv.org/abs/1910.00775 | Learned segment boundaries in a sequential latent model. |
| **Clockwork VAE** (Saxena, Ba, Hafner; NeurIPS 2021) | https://arxiv.org/abs/2102.09532 | Fixed multi-rate clocks — the "predefined" version of two clocks. |
| **Clockwork RNN** (Koutník et al.; ICML 2014) | https://arxiv.org/abs/1402.3511 | Fixed-period modules. |
| **Director** — Deep Hierarchical Planning from Pixels (Hafner et al.; NeurIPS 2022) | https://arxiv.org/abs/2206.04114 | Latent goals every fixed K steps. |
| CompILE (Kipf et al.; ICML 2019) | https://arxiv.org/abs/1812.01483 | Unsupervised segmentation of demonstrations into sub-tasks. |

## Foundations

| Work | Link | Relevance |
|---|---|---|
| A Path Towards Autonomous Machine Intelligence (LeCun, 2022) | https://openreview.net/pdf?id=BZ5a1r-kVsf | JEPA and H-JEPA framing. |
| VICReg (Bardes, Ponce, LeCun; ICLR 2022) | https://arxiv.org/abs/2105.04906 | Collapse prevention used in our original wall model. |
| BYOL (Grill et al.; NeurIPS 2020) | https://arxiv.org/abs/2006.07733 | EMA target encoder. |
| Barlow Twins (Zbontar et al.; ICML 2021) | https://arxiv.org/abs/2103.03230 | Alternative collapse prevention. |
| Diffusion Policy (Chi et al.; RSS 2023) | https://arxiv.org/abs/2303.04137 | Origin of the Push-T benchmark data. |
| Categorical reparameterization with Gumbel-Softmax (Jang et al.; ICLR 2017) | https://arxiv.org/abs/1611.01144 | Option for hard boundary sampling. |

## Venue

- CoRL 2026 Workshop: *Bringing Physics Simulation and World Models Together for Robotic Manipulation* — https://corl26ws-physwm.github.io/index.html. Deadline **Sep 30, 2026 23:59 AoE**; ≤4 pages excluding references; CoRL template; double-anonymous; non-archival.
