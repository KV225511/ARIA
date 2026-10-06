# Direct policy comparison on frozen ARIA development data

These are controlled synthetic development results. The locked test split was not used. They support comparison of policy behavior under the same ARIA interface, but they do not establish real-candidate performance or a universal state-of-the-art claim.

## Comparator training

| Comparator | Best validation logged-action accuracy |
|---|---:|
| behavior cloning | 0.2975 |
| discrete cql | 0.2776 |
| decision transformer | 0.1516 |

Validation action accuracy measures imitation of logged actions. It is a selection diagnostic, not the rollout outcome metric.

## Matched rollout results

Each cell is accuracy / mean designed reward / mean information gain over 45 matched episodes (15 per synthetic class).

| Policy | Base | Overlap | Low confidence | Positive shift |
|---|---:|---:|---:|---:|
| aria iql v7 | 1.0000 / 32.16 / 17.88 | 0.4667 / 31.09 / 17.26 | 0.9111 / 11.50 / 3.96 | 1.0000 / 31.14 / 17.27 |
| behavior cloning | 1.0000 / 25.90 / 14.02 | 0.5333 / 24.28 / 13.01 | 0.9333 / 8.33 / 2.85 | 1.0000 / 25.08 / 13.47 |
| discrete cql | 1.0000 / 27.54 / 16.72 | 0.5111 / 25.38 / 15.79 | 0.8889 / 5.01 / 3.35 | 1.0000 / 27.31 / 16.55 |
| decision transformer | 1.0000 / 7.22 / 5.13 | 0.4889 / 7.12 / 5.06 | 0.8889 / 2.97 / 1.82 | 1.0000 / 7.04 / 5.01 |

Base accuracy is saturated: multiple learned and non-learned policies reach 1.0 because the evidence centers are widely separated. Accuracy must therefore not be used to rank policies in that condition. Under overlap, IQL does not dominate classification accuracy: behavior cloning reaches 0.5333, discrete CQL 0.5111, Decision Transformer 0.4889, and ARIA IQL 0.4667. Across all four conditions, ARIA IQL has the largest mean designed reward and information gain among the direct learned comparators, while also covering all 17 skills. These outcomes show behavior under ARIA's designed simulator and reward; they do not demonstrate human interview quality.

## Provenance

Comparator training report hash: `5795042b3ae01be877ebab55ebb00d63622a8205b33dc7f122b16834149ddb95`.

- `base`: report hash `ec3b0af141742e2a16b5277789c9293d3f308b2e860a5be18de96a27fe16a045`; source SHA-256 `6d48e8ab7944aa548d87ad8366a85bac321802e6c196633c465156b0d4442abe`
- `overlap`: report hash `1c9933bdd49329ec6a2d2fa1f55db751ba31ed6fa075fcb8ff12615f38eef4b1`; source SHA-256 `14edd5825e47ba59b2bd0787ffb833cb59f22673ddb745f2a458a3562b0132f5`
- `low_confidence`: report hash `4a9b069c326eaf7a13552531571186858b966e655222f0370cfd6cab35a91c92`; source SHA-256 `3c85fe77ac4eabd46f80963b4c81cbfc867099f227072070de89db2590a5cc30`
- `positive_shift`: report hash `8443473247c1ac41988aae85cddf5dba1e389da013e32650869b41dbc61d4cff`; source SHA-256 `f77afd75da519d5af19c5118790a39847c3f12b7973f569f00e3906875c58b87`
