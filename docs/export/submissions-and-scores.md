# Submissions & official scores (as of 2026-08-11)

## Official leaderboard results (Development phase)
| upload | zip | size | official final | psnr | ssim | lpips | fscore | miou | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-08-pipeline-v4.zip | 1.51 GB | FAILED | - | - | - | - | - | empty log; timeout or size |
| 2 | v4-mini.zip | 606 MB | 1.262 (#1) | 20.82 | 0.770 | 0.294 | 0.521 | 0.110 | calibrated local scorer exactly |
| 3 | v5-safe.zip | 864 MB | 1.325 (#1) | 20.82 | 0.770 | 0.294 | 0.521 | 0.174 | splatting-depth fusion verified |

## Local-only builds (not uploaded; policy: final phase only)
| zip | size | md5 | expected |
|---|---|---|---|
| v6-safe-q.zip | 0.91 GB | a17c6aceb3a90c9fd6ecb35a10488ed7 | ~1.36 |
| v6-mid.zip | 1.53 GB | - | at known-failed size, shelved |

## Local component scores (held-out where marked)
| component | value | basis |
|---|---|---|
| rendering | 0.6309 (v4, 8-member) / 0.632 (16-member GC) | held-out dev photos |
| geometry | 0.5462 (loose_2x) -> ~0.585 pending (APD-MVS) | dev GT clouds |
| semantics | 0.1846 LOSO held-out (uniform-4M) | leave-one-scene-out |
| final-phase estimate | ~1.40 and climbing | honest basis |

## Raw directory
```
total 12246696
drwxrwxr-x  2 manish_joshi manish_joshi        268 Aug 11 14:06 .
drwxrwxr-x 12 manish_joshi manish_joshi        259 Aug  8 15:59 ..
-rw-rw-r--  1 manish_joshi manish_joshi 1519938836 Aug  9 01:19 2026-08-08-pipeline-v1.zip
-rw-rw-r--  1 manish_joshi manish_joshi 1906419600 Aug  8 23:08 2026-08-08-pipeline-v2.zip
-rw-rw-r--  1 manish_joshi manish_joshi 1506207454 Aug  9 02:04 2026-08-08-pipeline-v4.zip
-rw-rw-r--  1 manish_joshi manish_joshi  211600469 Aug  8 15:38 dummy_official.zip
-rw-rw-r--  1 manish_joshi manish_joshi  920027673 Aug 10 20:35 v4-lite.zip
-rw-rw-r--  1 manish_joshi manish_joshi  605928066 Aug  9 20:57 v4-mini.zip
-rw-rw-r--  1 manish_joshi manish_joshi 1331551672 Aug 10 22:13 v5-best.zip
-rw-rw-r--  1 manish_joshi manish_joshi  864335844 Aug 10 21:50 v5-safe.zip
-rw-rw-r--  1 manish_joshi manish_joshi 1532762636 Aug 11 13:53 v6-mid.zip
-rw-rw-r--  1 manish_joshi manish_joshi  912368813 Aug 11 14:08 v6-safe-q.zip
-rw-rw-r--  1 manish_joshi manish_joshi 1229429790 Aug 11 13:53 v6-safe.zip
```
