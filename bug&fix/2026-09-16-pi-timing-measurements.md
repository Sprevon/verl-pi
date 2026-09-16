# Real Pi trajectory timing

Commit: `92ead41bbeb156ff5d77c57d0a2b7d2badc50f03`

LLM request latency includes RPC, routing, queueing, prefill and decode; it is not GPU kernel time.
GPU readings are sampled utilization, not useful-compute efficiency. Tool time uses interval unions.

| Group | Split | Trajectories | Wall s | LLM union s | Request presence % | Mean sampled GPU % |
|---|---|---:|---:|---:|---:|---:|
| 6087cc95 | test | 1 | 17.110 | 4.143 | 24.21 | 11.44186046511628 |
| 7e20d297 | train | 4 | 17.651 | 8.079 | 45.77 | 10.55056179775281 |

## Session 16c7207d26bb48988345215a48de7a12

Task: `[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]`; split=test; completed=True; turns=6; tokens=107; reward=0; wall=17.110s.
Tool calls=6; tool errors=0; truncated=True.

| Phase | Seconds | Wall % |
|---|---:|---:|
| startup | 12.416295 | 72.57 |
| tokenization | 0.291602 | 1.70 |
| llm_request | 4.142649 | 24.21 |
| parse_and_record | 0.020312 | 0.12 |
| response_send | 0.000536 | 0.00 |
| tool | 0.020545 | 0.12 |
| evaluation | 0.086033 | 0.50 |
| sidecar_close | 0.021408 | 0.13 |
| other | 0.110460 | 0.65 |

| Start offset s | Duration s | Phase / tool | Generation |
|---:|---:|---|---|
| 0.000000 | 12.416295 | startup |  |
| 12.421935 | 0.055644 | tokenization | 1 |
| 12.477937 | 1.237598 | llm_request | 1 |
| 13.715688 | 0.003933 | parse_and_record | 1 |
| 13.724405 | 0.000090 | response_send | 1 |
| 13.726930 | 0.008028 | check_data_restriction_status | 1 |
| 13.747797 | 0.047824 | tokenization | 2 |
| 13.795930 | 0.619925 | llm_request | 2 |
| 14.416024 | 0.003316 | parse_and_record | 2 |
| 14.424040 | 0.000090 | response_send | 2 |
| 14.424930 | 0.003233 | toggle_data_saver_mode | 2 |
| 14.441767 | 0.047454 | tokenization | 3 |
| 14.489538 | 0.573255 | llm_request | 3 |
| 15.062952 | 0.003366 | parse_and_record | 3 |
| 15.071126 | 0.000093 | response_send | 3 |
| 15.071930 | 0.002388 | check_network_status | 3 |
| 15.087076 | 0.047808 | tokenization | 4 |
| 15.135231 | 0.588365 | llm_request | 4 |
| 15.723763 | 0.003233 | parse_and_record | 4 |
| 15.731508 | 0.000093 | response_send | 4 |
| 15.731930 | 0.002524 | check_network_mode_preference | 4 |
| 15.747049 | 0.046950 | tokenization | 5 |
| 15.794315 | 0.575622 | llm_request | 5 |
| 16.370098 | 0.003271 | parse_and_record | 5 |
| 16.378273 | 0.000089 | response_send | 5 |
| 16.378930 | 0.002198 | run_speed_test | 5 |
| 16.393340 | 0.045922 | tokenization | 6 |
| 16.439593 | 0.547884 | llm_request | 6 |
| 16.987632 | 0.003195 | parse_and_record | 6 |
| 16.995215 | 0.000082 | response_send | 6 |
| 16.995930 | 0.002175 | check_network_status | 6 |
| 17.000930 | 0.086033 | evaluation |  |
| 17.088390 | 0.021408 | sidecar_close |  |

## Session 89c4bfb692ae4554a4843a498e65b64b

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`; split=train; completed=True; turns=6; tokens=107; reward=0; wall=16.925s.
Tool calls=6; tool errors=1; truncated=True.

| Phase | Seconds | Wall % |
|---|---:|---:|
| startup | 9.643592 | 56.98 |
| tokenization | 0.333624 | 1.97 |
| llm_request | 6.644207 | 39.26 |
| parse_and_record | 0.024553 | 0.15 |
| response_send | 0.000602 | 0.00 |
| tool | 0.018122 | 0.11 |
| evaluation | 0.113772 | 0.67 |
| sidecar_close | 0.029061 | 0.17 |
| other | 0.117540 | 0.69 |

| Start offset s | Duration s | Phase / tool | Generation |
|---:|---:|---|---|
| 0.000000 | 9.643592 | startup |  |
| 9.650447 | 0.080189 | tokenization | 1 |
| 9.732390 | 3.897132 | llm_request | 1 |
| 13.630243 | 0.008252 | parse_and_record | 1 |
| 13.643275 | 0.000127 | response_send | 1 |
| 13.653955 | 0.007885 | toggle_airplane_mode | 1 |
| 13.673137 | 0.054898 | tokenization | 2 |
| 13.728207 | 0.565252 | llm_request | 2 |
| 14.293640 | 0.003426 | parse_and_record | 2 |
| 14.301545 | 0.000091 | response_send | 2 |
| 14.301955 | 0.003380 | check_network_status | 2 |
| 14.318310 | 0.056179 | tokenization | 3 |
| 14.374732 | 0.622378 | llm_request | 3 |
| 14.997275 | 0.003378 | parse_and_record | 3 |
| 15.005039 | 0.000091 | response_send | 3 |
| 15.006955 | 0.000227 | check_data_restrictions (error) | 3 |
| 15.019280 | 0.050489 | tokenization | 4 |
| 15.070085 | 0.546897 | llm_request | 4 |
| 15.617132 | 0.003164 | parse_and_record | 4 |
| 15.624561 | 0.000079 | response_send | 4 |
| 15.624955 | 0.002188 | check_data_restriction_status | 4 |
| 15.638687 | 0.047114 | tokenization | 5 |
| 15.685925 | 0.510981 | llm_request | 5 |
| 16.197058 | 0.003179 | parse_and_record | 5 |
| 16.205304 | 0.000127 | response_send | 5 |
| 16.205955 | 0.002106 | check_network_mode_preference | 5 |
| 16.219647 | 0.044755 | tokenization | 6 |
| 16.264702 | 0.501568 | llm_request | 6 |
| 16.766424 | 0.003153 | parse_and_record | 6 |
| 16.774426 | 0.000087 | response_send | 6 |
| 16.774955 | 0.002336 | run_speed_test | 6 |
| 16.780955 | 0.113772 | evaluation |  |
| 16.895952 | 0.029061 | sidecar_close |  |

## Session 99a179831dc4498186f7066cf130c475

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`; split=train; completed=True; turns=6; tokens=107; reward=0; wall=17.173s.
Tool calls=6; tool errors=1; truncated=True.

| Phase | Seconds | Wall % |
|---|---:|---:|
| startup | 10.426320 | 60.71 |
| tokenization | 0.317801 | 1.85 |
| llm_request | 6.076340 | 35.38 |
| parse_and_record | 0.040657 | 0.24 |
| response_send | 0.000343 | 0.00 |
| tool | 0.018153 | 0.11 |
| evaluation | 0.118458 | 0.69 |
| sidecar_close | 0.056117 | 0.33 |
| other | 0.118980 | 0.69 |

| Start offset s | Duration s | Phase / tool | Generation |
|---:|---:|---|---|
| 0.000000 | 10.426320 | startup |  |
| 10.430053 | 0.074001 | tokenization | 1 |
| 10.504218 | 3.207148 | llm_request | 1 |
| 13.711498 | 0.012455 | parse_and_record | 1 |
| 13.726302 | 0.000044 | response_send | 1 |
| 13.729657 | 0.008333 | toggle_airplane_mode | 1 |
| 13.765925 | 0.056574 | tokenization | 2 |
| 13.822668 | 0.635182 | llm_request | 2 |
| 14.457973 | 0.008912 | parse_and_record | 2 |
| 14.470203 | 0.000059 | response_send | 2 |
| 14.470657 | 0.002945 | check_network_status | 2 |
| 14.485350 | 0.050937 | tokenization | 3 |
| 14.536598 | 0.646185 | llm_request | 3 |
| 15.182897 | 0.010097 | parse_and_record | 3 |
| 15.195460 | 0.000046 | response_send | 3 |
| 15.197657 | 0.000113 | check_data_restrictions (error) | 3 |
| 15.205981 | 0.041866 | tokenization | 4 |
| 15.247971 | 0.535755 | llm_request | 4 |
| 15.783784 | 0.001599 | parse_and_record | 4 |
| 15.787490 | 0.000048 | response_send | 4 |
| 15.787657 | 0.002192 | check_data_restriction_status | 4 |
| 15.811438 | 0.048156 | tokenization | 5 |
| 15.859762 | 0.528044 | llm_request | 5 |
| 16.387947 | 0.004558 | parse_and_record | 5 |
| 16.397321 | 0.000093 | response_send | 5 |
| 16.397657 | 0.002087 | check_network_mode_preference | 5 |
| 16.414003 | 0.046267 | tokenization | 6 |
| 16.460397 | 0.524025 | llm_request | 6 |
| 16.984532 | 0.003036 | parse_and_record | 6 |
| 16.991471 | 0.000052 | response_send | 6 |
| 16.991657 | 0.002482 | check_wifi_status | 6 |
| 16.997657 | 0.118458 | evaluation |  |
| 17.117017 | 0.056117 | sidecar_close |  |

## Session d7c4cff6e7e24a659b45ca8b796692f5

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`; split=train; completed=True; turns=6; tokens=109; reward=0; wall=17.465s.
Tool calls=6; tool errors=1; truncated=True.

| Phase | Seconds | Wall % |
|---|---:|---:|
| startup | 9.594718 | 54.94 |
| tokenization | 0.334100 | 1.91 |
| llm_request | 6.802285 | 38.95 |
| parse_and_record | 0.050680 | 0.29 |
| response_send | 0.000328 | 0.00 |
| tool | 0.017739 | 0.10 |
| evaluation | 0.115324 | 0.66 |
| sidecar_close | 0.420456 | 2.41 |
| other | 0.129359 | 0.74 |

| Start offset s | Duration s | Phase / tool | Generation |
|---:|---:|---|---|
| 0.000000 | 9.594718 | startup |  |
| 9.597244 | 0.076142 | tokenization | 1 |
| 9.673678 | 3.899416 | llm_request | 1 |
| 13.573279 | 0.017571 | parse_and_record | 1 |
| 13.593941 | 0.000044 | response_send | 1 |
| 13.601023 | 0.008179 | toggle_airplane_mode | 1 |
| 13.629978 | 0.052255 | tokenization | 2 |
| 13.682379 | 0.610399 | llm_request | 2 |
| 14.292943 | 0.014535 | parse_and_record | 2 |
| 14.312185 | 0.000079 | response_send | 2 |
| 14.314023 | 0.002996 | check_network_status | 2 |
| 14.335829 | 0.057497 | tokenization | 3 |
| 14.393704 | 0.651864 | llm_request | 3 |
| 15.045707 | 0.003051 | parse_and_record | 3 |
| 15.052473 | 0.000060 | response_send | 3 |
| 15.054023 | 0.000196 | check_data_restrictions (error) | 3 |
| 15.070911 | 0.052836 | tokenization | 4 |
| 15.123874 | 0.562603 | llm_request | 4 |
| 15.686618 | 0.002909 | parse_and_record | 4 |
| 15.693607 | 0.000063 | response_send | 4 |
| 15.694023 | 0.002473 | check_data_restriction_status | 4 |
| 15.708051 | 0.046005 | tokenization | 5 |
| 15.754233 | 0.506454 | llm_request | 5 |
| 16.260823 | 0.010673 | parse_and_record | 5 |
| 16.275185 | 0.000048 | response_send | 5 |
| 16.276023 | 0.002033 | check_network_mode_preference | 5 |
| 16.293270 | 0.049364 | tokenization | 6 |
| 16.342749 | 0.571550 | llm_request | 6 |
| 16.914353 | 0.001941 | parse_and_record | 6 |
| 16.918023 | 0.001862 | check_wifi_calling_status | 6 |
| 16.918067 | 0.000034 | response_send | 6 |
| 16.922023 | 0.115324 | evaluation |  |
| 17.044430 | 0.420456 | sidecar_close |  |

## Session da799d7996a243d8a84bb8890c95467e

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`; split=train; completed=True; turns=6; tokens=109; reward=0; wall=17.182s.
Tool calls=6; tool errors=1; truncated=True.

| Phase | Seconds | Wall % |
|---|---:|---:|
| startup | 8.836045 | 51.43 |
| tokenization | 0.353433 | 2.06 |
| llm_request | 7.696867 | 44.80 |
| parse_and_record | 0.030285 | 0.18 |
| response_send | 0.000437 | 0.00 |
| tool | 0.017318 | 0.10 |
| evaluation | 0.113621 | 0.66 |
| sidecar_close | 0.021058 | 0.12 |
| other | 0.113245 | 0.66 |

| Start offset s | Duration s | Phase / tool | Generation |
|---:|---:|---|---|
| 0.000000 | 8.836045 | startup |  |
| 8.838489 | 0.098948 | tokenization | 1 |
| 8.937656 | 4.820185 | llm_request | 1 |
| 13.758040 | 0.013595 | parse_and_record | 1 |
| 13.774976 | 0.000079 | response_send | 1 |
| 13.778252 | 0.008042 | toggle_airplane_mode | 1 |
| 13.804596 | 0.052430 | tokenization | 2 |
| 13.857182 | 0.620206 | llm_request | 2 |
| 14.477559 | 0.004385 | parse_and_record | 2 |
| 14.486253 | 0.000082 | response_send | 2 |
| 14.487252 | 0.003070 | check_network_status | 2 |
| 14.506383 | 0.052806 | tokenization | 3 |
| 14.562979 | 0.659696 | llm_request | 3 |
| 15.222849 | 0.003134 | parse_and_record | 3 |
| 15.230268 | 0.000077 | response_send | 3 |
| 15.231252 | 0.000234 | check_data_restrictions (error) | 3 |
| 15.246006 | 0.046931 | tokenization | 4 |
| 15.293149 | 0.543171 | llm_request | 4 |
| 15.836386 | 0.002356 | parse_and_record | 4 |
| 15.840850 | 0.000041 | response_send | 4 |
| 15.841252 | 0.002082 | check_data_restriction_status | 4 |
| 15.855247 | 0.052506 | tokenization | 5 |
| 15.907875 | 0.509721 | llm_request | 5 |
| 16.417734 | 0.003113 | parse_and_record | 5 |
| 16.425550 | 0.000083 | response_send | 5 |
| 16.426252 | 0.002056 | check_network_mode_preference | 5 |
| 16.438983 | 0.049811 | tokenization | 6 |
| 16.489006 | 0.543888 | llm_request | 6 |
| 17.033054 | 0.003703 | parse_and_record | 6 |
| 17.041088 | 0.000075 | response_send | 6 |
| 17.042252 | 0.001833 | check_wifi_calling_status | 6 |
| 17.046252 | 0.113621 | evaluation |  |
| 17.161116 | 0.021058 | sidecar_close |  |
