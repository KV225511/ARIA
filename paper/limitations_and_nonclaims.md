# Limitations and Nonclaims

1. **Synthetic evidence only.** The locked evaluation covers 91 identity-isolated synthetic episodes; it does not estimate performance on real applicants.
2. **Generated labels are not validated skills.** Low, medium, and high categories verify pipeline behavior but are not job-performance or KSAO criteria.
3. **No learned-policy effect estimate.** The locked test explicitly sets `evaluates_learned_policy` to false. The validation WIS result is diagnostic and its release gate is false.
4. **No fresh rollouts.** The repository contains no controlled held-out rollout comparison against fixed, random-valid, or uncertainty-greedy policies.
5. **Live/offline gap.** The live backend injects fixed semantic and behavioral evidence and cycles three actions; it does not load the frozen IQL checkpoint or execute the full multimodal pipeline.
6. **No hiring-validity claim.** ARIA has not been compared with job performance, structured human interview ratings, or another suitable external criterion.
7. **No fairness or accessibility claim.** No subgroup, disability, neurotype, accent, ASR-error, accommodation, or intersectional evaluation has been run.
8. **No causal claim.** Neither logged-data OPE nor synthetic classification establishes that ARIA causes better interviews, decisions, or outcomes.
9. **No production-readiness claim.** The focused suite passes, but unscoped test collection is not clean and optional dependencies are incomplete.
10. **No exhaustive novelty claim.** The 70-source literature set used Undermind and Scite; Zotero was unavailable, and 2026 work including PolyInterview, recruiting-agent reviews, AI-interviewer user studies, and AVI-Personality substantially narrows the systems gap.
11. **Privacy and governance remain designs, not results.** Consent, retention, encryption, contestability, and human oversight have not been evaluated in deployment.
12. **Historical results are not current evidence.** Older 53-episode, approximately 96% results and 32-feature/belief-v2 descriptions are retained only as project history.
