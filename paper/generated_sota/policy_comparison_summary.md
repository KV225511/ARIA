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
| behavior cloning | 1.0000 / 25.90 / 14.02 | 0.5333 / 24.28 / 13.01 | 0.9333 / 8.30 / 2.84 | 1.0000 / 25.08 / 13.47 |
| discrete cql | 1.0000 / 27.54 / 16.72 | 0.5111 / 25.38 / 15.79 | 0.8889 / 5.01 / 3.35 | 1.0000 / 27.31 / 16.55 |
| decision transformer | 1.0000 / 7.22 / 5.13 | 0.4889 / 7.12 / 5.06 | 0.8889 / 2.97 / 1.82 | 1.0000 / 7.04 / 5.01 |

Base accuracy is saturated: multiple learned and non-learned policies reach 1.0 because the evidence centers are widely separated. Accuracy must therefore not be used to rank policies in that condition. Under overlap, IQL does not dominate classification accuracy: behavior cloning reaches 0.5333, discrete CQL 0.5111, Decision Transformer 0.4889, and ARIA IQL 0.4667. Across all four conditions, ARIA IQL has the largest mean designed reward and information gain among the direct learned comparators, while also covering all 17 skills. These outcomes show behavior under ARIA's designed simulator and reward; they do not demonstrate human interview quality.

## Provenance

Comparator training report hash: `1c988a6935e5833aca60805f9a6fdf5277685e46e88786efcf84ad213b4c03c0`.

- `base`: report hash `e6d1a126a02cf151ff3958afef6e8c1184766ec64d7885ef2edd5513a0532594`; source SHA-256 `81a9602afd1903bc7c9663737423f85de2d62439ea00864a36c0e5ef8cc39f53`
- `overlap`: report hash `09ccb855120ac35a0f9bcad42234f678c1a864f63e494571c59a58398bbe68ed`; source SHA-256 `4770b6d733ed5ab7c959178f814dd3597b9f597900d6c53c4d41fe401e355a3a`
- `low_confidence`: report hash `e712784aaed2437fa67a00abdf1d994e84af84936e5df2293e539c2cb2fe9626`; source SHA-256 `c9eb0d8a2c9ce9b87847a9a16daf9e6e7c5de519f7a75cb2945e6341ad350708`
- `positive_shift`: report hash `510b815517ee07483880481c4cf299c9a05e134a59792e964d5a888569e069aa`; source SHA-256 `b4a80a780f228edfcf7c4f1652b957ff240c033a932704e10f71001f3d328dbc`
