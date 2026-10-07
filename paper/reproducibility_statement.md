# Reproducibility Statement

The manuscript is tied to the ARIA checkout on `main` at commit `99fd7550a235f97fa950c3037eb4d71d76ffca43`. The current policy schema is `aria-state-v4` with 33 state dimensions; the current belief schema is `belief-v3`; the action space has eight actions.

The production synthetic corpus contains 600 episodes and 9,018 transitions. Its locked report records raw-dataset digest `50f9f5da589b0859667310eeed97717414f80e44fd1ee518ec9f953a978daf2c`, raw-file SHA-256 `e38fbb09470b1d6ec2787e99ea2e9a03dc81faa7b5a0294c4a2f205fc1767a6f`, split-manifest digest `dd1eaf053b0180c6b5972d35695df8cf5001da99ee9c6d5126a2a5375f0121df`, belief-configuration digest `dfbc53ecdd28066f1957eb654612f45a53754ea85f1c5c2467f555890151c27d`, and protocol digest `6d2b26af9d43ec51c67e53cdc525c8b8c43d28404225029d0064604a4b53a64d`.

The IQL checkpoint is `modules/module_07_rl/aria_iql_belief_v7.pth`, SHA-256 `6a9c604bbaf6af987ec631ce27574766b85594af5f248ca88a920fde06c99029`. Its metadata records seed 42, discount 0.99, target-update rate 0.005, expectile 0.8, inverse temperature 3, learning rate 0.0001, patience 10, best epoch 4, and validation objective 4.13864.

The locked report's paired v6-v7 comparison uses the same 91 episodes. V7 increases accuracy by 0.120879 and macro-F1 by 0.115762; paired identity-component percentile-bootstrap 95% intervals are [0.043478, 0.223529] and [0.044336, 0.197551], respectively. These intervals use six identity components and are diagnostic for the synthetic corpus, not population-general confidence intervals.

Focused verification covered 20 Module 7/calibration test files and passed 341 tests. The React/Vite frontend production build passed. An unscoped pytest run did not collect cleanly because it included vendor tests, a stale root test, and modules whose optional dependencies were unavailable. The paper does not label the entire repository test suite as passing.

Reproduction requires preserving model identifiers and prompts for synthetic generation (`qwen2.5:7b` candidate; `gemma3:4b` evaluator), immutable raw outputs, parser/evaluator versions, identity-component assignments, configuration files, and checkpoint metadata. Because model behavior and local runtimes may change, regenerated corpora are new studies rather than interchangeable reproductions of the frozen release.
