# Academic Review of the ARIA Manuscript

Review language: English  
Artifact reviewed: `paper/ARIA_complete_research_paper.md`  
Review date: 2026-09-15

## Part I — Reviewer Feedback

### Summary and overall assessment

The manuscript is a careful repository-grounded characterization of an adaptive mock-interview research prototype. Its strongest contribution is epistemic discipline: it separates a locked synthetic belief evaluation, a diagnostic offline-policy estimate, and a simplified live application that does not load the evaluated policy. The paper also replaces stale Project I values with the current 33-dimensional state, belief-v3 lineage, and 91-episode locked test.

The paper is technically credible for an internal report, reproducibility paper, or system-audit submission. The revised version now follows the requested IEEE-style section structure, cites 70 verified and relevant sources, uses three IEEE Access comparators, and integrates eight 2026-dated papers across Related Work, Methodology, and Discussion. It is not yet a convincing empirical policy-learning or automated-selection paper because there are no fresh rollouts, no simple policy baselines, and no human criterion or subgroup study. The title and abstract are appropriately limited to an architecture and do not claim deployment validity.

### Must-fix issues

1. **A learned-policy result is absent.** The IQL checkpoint and WIS diagnostic show implementation and overlap, not improvement. A policy-focused venue would require fresh held-out rollouts against fixed-cycle, random-valid-action, and uncertainty-greedy policies, with uncertainty intervals and failure analysis. The manuscript correctly treats this as a nonclaim; it cannot be repaired by prose.
2. **Real-person validity is absent.** Synthetic low/medium/high labels are not occupational constructs. Any claim about assessment quality, hiring decisions, fairness, or applicant benefit requires job-analysis-grounded criteria and an ethics-reviewed human study. The manuscript correctly blocks those claims.
3. **The live/offline discontinuity must remain prominent.** The current backend uses fixed evidence values and a three-action cycle. Figure 1, the abstract, Methods, Results, Discussion, and Conclusion now state this consistently. Removing that disclosure would materially misrepresent the evaluated system.

### High-leverage improvements

1. Add one frozen rollout study with predeclared outcomes: cumulative reward, skill coverage, repetition, invalid actions, interview length, and belief calibration at termination.
2. Add paired per-identity comparisons and bootstrap or repeated-seed intervals; do not rely on one average reward.
3. Publish a compact model card/data statement linking prompts, model identifiers, raw hashes, split components, calibration config, checkpoint hash, and release decisions.
4. Clarify the intended venue. A machine-learning venue will expect stronger policy evaluation; an HCI venue will expect participant evidence; a systems/reproducibility venue is the best match for the present artifact.
5. Replace the remaining abbreviated configuration/protocol digests in the standalone reproducibility statement with the full values before external archival.

### Easy wins

1. Keep the current title, which says “architecture” rather than “validated system.”
2. Retain the locked-test table and WIS table as separate results; this prevents readers from combining them incorrectly.
3. Retain the failed historical belief audit as calibration lineage, explicitly not as a controlled ablation.
4. Keep “mock interview” and “research prototype” stable throughout the manuscript.
5. Add the exact command list or a release manifest if the paper moves to a venue with artifact evaluation.

### Optional refinements

1. A future two-column venue template would shorten the visual footprint but is unnecessary before venue selection.
2. A second figure could show the 33-feature state in groups, but Table 1 already provides the necessary contract without visual decoration.
3. An appendix could list all eight reward-term definitions; the current paper reports coefficients but avoids unsupported causal interpretation.

### Evidence and claim audit

| Dimension | Judgment |
|---|---|
| Gap–question–method alignment | Pass: all three questions have methods, evidence, answers, and limitations |
| Numerical traceability | Pass: main numbers map to frozen JSON/checkpoint/corpus artifacts |
| Citation relevance | Pass with limits: 70/70 sources verified and cited; eight central papers read in full for the revision; Zotero unavailable |
| Construct validity | Not established and not claimed |
| Learned-policy effectiveness | Not established and not claimed |
| Fairness/accessibility | Not established and not claimed |
| Live-system integration | Incomplete and disclosed |
| Reproducibility | Strong for artifact lineage; weaker for environment-wide dependency recreation |
| Writing and organization | Clear, cumulative, and appropriately scoped |

### Recommendation

**Recommendation: major revision for IEEE Access or another external empirical venue; accept as an internal research and reproducibility report.** The missing experiments are central to policy or hiring claims but do not invalidate the paper’s narrower audit contribution.

## Part II — Readiness Score and Second-Pass Review

### Readiness score

**8/10 for an internal or artifact paper; 5/10 for an IEEE Access systems submission; 4/10 for a policy-learning or hiring-validity submission.**

The manuscript is complete, coherent, citation-checked, and visually deliverable. Its acceptance risk comes from missing primary experiments rather than presentation. A venue that rewards transparent negative evidence and reproducible system characterization is plausible; ML, HCI, or personnel-selection venues would expect different additional evidence.

### Second-pass confirmation after revisions

The exact post-edit candidate was reread in four passes.

1. **Argument and structure:** The requested Abstract, Keywords, I. Introduction, II. Related Work / Literature Review, III. Proposed Model / Methodology, IV. Results and Analysis, V. Conclusion, and References are present. The introduction states three answerable questions. Methods separate intended, offline, and live paths. Results answer belief, policy-diagnostic, and readiness questions in that order. The conclusion preserves the prototype boundary.
2. **Evidence and consistency:** The current schema is 33-dimensional `aria-state-v4`; current beliefs are `belief-v3`; the locked sample is 91; the production corpus is 600 episodes and 9,018 transitions. The locked report is not called a policy evaluation. The WIS estimate is not called causal.
3. **Scholarly prose:** Stock transitions, empty intensifiers, vague novelty language, and unsupported “demonstrates/proves” language were avoided. Technical compounds and repeated terms were retained where they preserve contract identity. The reference list produces unavoidable repeated DOI and journal openings.
4. **Delivery integrity:** Markdown, BibTeX, IEEEtran LaTeX, DOCX, and PDF artifacts exist. The final PDF was visually inspected page by page through a contact sheet; the three figures and four tables are readable and no clipping or overlap was found. The DOCX package was structurally inspected with 148 paragraphs, four tables, three embedded figures, no comments, and no tracked changes. Direct Word/LibreOffice conversion was unavailable because the Word COM shim could not create a session, so exact DOCX pagination remains an environment limitation rather than a passed native render.

### Remaining blockers by intended use

- **External empirical policy paper:** fresh policy rollouts and baselines.
- **Automated selection / personnel assessment paper:** human criterion validation, reliability, subgroup fairness, accessibility, applicant reactions, and governance protocol.
- **Internal research-prototype release:** no S3/S4 blocker remains because the missing experiments are explicit nonclaims; the unscoped pytest discovery issue remains S2.

### Final reviewer judgment

The manuscript is ready to circulate as an honest IEEE-style research-prototype paper, but an IEEE Access submission would still face a major-revision risk because the proposed policy lacks fresh comparative rollouts and the system lacks human validation. It should not be submitted under a title or abstract that implies learned-policy superiority, live end-to-end operation, or validated hiring use. The next scientific priority is a frozen comparative rollout study, not further prose polishing.
