# Real Pi startup breakdown

Inclusive spans nest: extension load is inside reload; Python startup/environment is inside bindExtensions.
Do not sum inclusive durations. Exclusive time subtracts the union of contained spans across processes.
First LLM submitted means the Worker entered its server_manager.generate await (not GPU kernel start).
Prompt preparation means session.prompt entry to the first provider callback, including skill expansion.
Canonical prompt publication is a marker, not a measurement of the ticket formatter's execution time.
Tau2 solo agent/user simulator init_state: not called in this Pi bridge path.
CPU preflight sessions are excluded. All listed requests belong to real model rollout traces.

## Session 485685bf984e465692ac40468ad623aa (test)

Task: `[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]`
First LLM submit: 12.090402 s; uncovered: 0.016643 s.

| Start offset s | Inclusive s | Exclusive s | Process | Phase |
|---:|---:|---:|---|---|
| 0.000114 | 0.259461 | 0.259461 | worker | sidecar_start |
| 0.258391 | 0.000000 | 0.000000 | node | node.ready |
| 0.261022 | 1.103745 | 1.103745 | node | pi.sdk_import |
| 1.364972 | 0.001436 | 0.001436 | node | pi.resource_loader_ctor |
| 1.366478 | 0.027630 | 0.016178 | node | pi.resource_loader_reload |
| 1.371051 | 0.011452 | 0.011452 | node | pi.extension_load |
| 1.394177 | 0.018469 | 0.018469 | node | pi.model_runtime_create |
| 1.413291 | 0.008877 | 0.008877 | node | pi.create_agent_session |
| 1.422566 | 10.573148 | 0.089601 | node | pi.bind_extensions |
| 1.486108 | 9.642981 | 9.642981 | python | tau2.package_import |
| 11.129176 | 0.001745 | 0.001745 | python | tau2.bridge_module_import |
| 11.130976 | 0.000073 | 0.000073 | python | tau2.install_timing_hooks |
| 11.131086 | 0.054948 | 0.000078 | python | tau2.bridge_ctor |
| 11.131091 | 0.054869 | 0.054859 | python | tau2.get_environment |
| 11.137936 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 11.186112 | 0.059505 | 0.059505 | python | tau2.describe_tools |
| 11.267315 | 0.724277 | 0.001513 | python | tau2.load_task |
| 11.267323 | 0.443015 | 0.443015 | python | tau2.task_catalog_load |
| 11.711606 | 0.231055 | 0.231044 | python | tau2.get_environment |
| 11.718327 | 0.000011 | 0.000010 | python | tau2.environment_ctor |
| 11.942748 | 0.000403 | 0.000177 | python | tau2.set_state |
| 11.942762 | 0.000094 | 0.000094 | python | tau2.initialization_action |
| 11.942898 | 0.000033 | 0.000033 | python | tau2.initialization_action |
| 11.942958 | 0.000021 | 0.000021 | python | tau2.initialization_action |
| 11.943002 | 0.000061 | 0.000061 | python | tau2.initialization_action |
| 11.943098 | 0.000017 | 0.000017 | python | tau2.initialization_action |
| 11.943227 | 0.048291 | 0.048291 | python | tau2.describe_tools |
| 11.995035 | 0.000018 | 0.000018 | node | pi.canonical_prompt_published |
| 11.995801 | 0.014561 | 0.014561 | node | pi.prompt_to_first_provider |
| 12.010489 | 0.000503 | 0.000503 | node | pi.provider_context_conversion |
| 12.018454 | 0.000000 | 0.000000 | worker | generation_request_received |
| 12.024123 | 0.065930 | 0.065930 | worker | tokenization |
| 12.090402 | 0.000000 | 0.000000 | worker | llm_request_submitted |

## Session 669fa39ef527452fae8e40e3efcf4efe (train)

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`
First LLM submit: 8.743118 s; uncovered: 0.034834 s.

| Start offset s | Inclusive s | Exclusive s | Process | Phase |
|---:|---:|---:|---|---|
| 0.000086 | 0.357920 | 0.357919 | worker | sidecar_start |
| 0.356848 | 0.000000 | 0.000000 | node | node.ready |
| 0.360909 | 1.192525 | 1.192525 | node | pi.sdk_import |
| 1.553646 | 0.001470 | 0.001470 | node | pi.resource_loader_ctor |
| 1.555197 | 0.029677 | 0.017642 | node | pi.resource_loader_reload |
| 1.560632 | 0.012036 | 0.012036 | node | pi.extension_load |
| 1.584948 | 0.022858 | 0.022858 | node | pi.model_runtime_create |
| 1.608456 | 0.008832 | 0.008832 | node | pi.create_agent_session |
| 1.617641 | 7.010249 | 0.090099 | node | pi.bind_extensions |
| 1.682575 | 5.970452 | 5.970452 | python | tau2.package_import |
| 7.653104 | 0.001820 | 0.001820 | python | tau2.bridge_module_import |
| 7.654977 | 0.000072 | 0.000072 | python | tau2.install_timing_hooks |
| 7.655087 | 0.055213 | 0.000072 | python | tau2.bridge_ctor |
| 7.655092 | 0.055141 | 0.055132 | python | tau2.get_environment |
| 7.662011 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 7.710373 | 0.062349 | 0.062349 | python | tau2.describe_tools |
| 7.793563 | 0.830230 | 0.001714 | python | tau2.load_task |
| 7.793571 | 0.714627 | 0.714627 | python | tau2.task_catalog_load |
| 8.509521 | 0.051077 | 0.051067 | python | tau2.get_environment |
| 8.516432 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 8.560734 | 0.001343 | 0.000557 | python | tau2.set_state |
| 8.560775 | 0.000278 | 0.000278 | python | tau2.initialization_action |
| 8.561146 | 0.000316 | 0.000316 | python | tau2.initialization_action |
| 8.561522 | 0.000031 | 0.000031 | python | tau2.initialization_action |
| 8.561661 | 0.000123 | 0.000123 | python | tau2.initialization_action |
| 8.561862 | 0.000038 | 0.000038 | python | tau2.initialization_action |
| 8.562243 | 0.061469 | 0.061469 | python | tau2.describe_tools |
| 8.627315 | 0.000014 | 0.000014 | node | pi.canonical_prompt_published |
| 8.627972 | 0.012865 | 0.012865 | node | pi.prompt_to_first_provider |
| 8.640966 | 0.000600 | 0.000600 | node | pi.provider_context_conversion |
| 8.661488 | 0.000000 | 0.000000 | worker | generation_request_received |
| 8.665450 | 0.071288 | 0.071289 | worker | tokenization |
| 8.743118 | 0.000000 | 0.000000 | worker | llm_request_submitted |

## Session 88703adf48e345d082a5b9de56c60cff (train)

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`
First LLM submit: 9.779503 s; uncovered: 0.015594 s.

| Start offset s | Inclusive s | Exclusive s | Process | Phase |
|---:|---:|---:|---|---|
| 0.000084 | 0.346825 | 0.346825 | worker | sidecar_start |
| 0.345730 | 0.000000 | 0.000000 | node | node.ready |
| 0.349581 | 1.171349 | 1.171349 | node | pi.sdk_import |
| 1.521144 | 0.001439 | 0.001439 | node | pi.resource_loader_ctor |
| 1.522654 | 0.027635 | 0.016322 | node | pi.resource_loader_reload |
| 1.527236 | 0.011313 | 0.011313 | node | pi.extension_load |
| 1.550358 | 0.018708 | 0.018708 | node | pi.model_runtime_create |
| 1.569696 | 0.008876 | 0.008876 | node | pi.create_agent_session |
| 1.578928 | 8.113519 | 0.091816 | node | pi.bind_extensions |
| 1.645659 | 7.048485 | 7.048485 | python | tau2.package_import |
| 8.694224 | 0.001768 | 0.001768 | python | tau2.bridge_module_import |
| 8.696053 | 0.000071 | 0.000071 | python | tau2.install_timing_hooks |
| 8.696162 | 0.054743 | 0.000060 | python | tau2.bridge_ctor |
| 8.696167 | 0.054683 | 0.054673 | python | tau2.get_environment |
| 8.703094 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 8.750976 | 0.059039 | 0.059039 | python | tau2.describe_tools |
| 8.830751 | 0.857580 | 0.001516 | python | tau2.load_task |
| 8.830768 | 0.746566 | 0.746566 | python | tau2.task_catalog_load |
| 9.578609 | 0.050593 | 0.050581 | python | tau2.get_environment |
| 9.585549 | 0.000011 | 0.000011 | python | tau2.environment_ctor |
| 9.629268 | 0.000414 | 0.000166 | python | tau2.set_state |
| 9.629280 | 0.000089 | 0.000089 | python | tau2.initialization_action |
| 9.629406 | 0.000087 | 0.000087 | python | tau2.initialization_action |
| 9.629521 | 0.000020 | 0.000020 | python | tau2.initialization_action |
| 9.629563 | 0.000037 | 0.000037 | python | tau2.initialization_action |
| 9.629630 | 0.000017 | 0.000016 | python | tau2.initialization_action |
| 9.629761 | 0.058491 | 0.058491 | python | tau2.describe_tools |
| 9.691801 | 0.000016 | 0.000016 | node | pi.canonical_prompt_published |
| 9.692530 | 0.013600 | 0.013600 | node | pi.prompt_to_first_provider |
| 9.706256 | 0.000553 | 0.000553 | node | pi.provider_context_conversion |
| 9.713780 | 0.000000 | 0.000000 | worker | generation_request_received |
| 9.717769 | 0.061406 | 0.061406 | worker | tokenization |
| 9.779503 | 0.000000 | 0.000000 | worker | llm_request_submitted |

## Session 951f9761a84447a5a45027a94fb66c3a (train)

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`
First LLM submit: 8.288040 s; uncovered: 0.013732 s.

| Start offset s | Inclusive s | Exclusive s | Process | Phase |
|---:|---:|---:|---|---|
| 0.000087 | 0.215498 | 0.215498 | worker | sidecar_start |
| 0.214375 | 0.000000 | 0.000000 | node | node.ready |
| 0.216898 | 1.194780 | 1.194780 | node | pi.sdk_import |
| 1.411889 | 0.001427 | 0.001427 | node | pi.resource_loader_ctor |
| 1.413385 | 0.027462 | 0.016175 | node | pi.resource_loader_reload |
| 1.417954 | 0.011286 | 0.011286 | node | pi.extension_load |
| 1.440915 | 0.017992 | 0.017992 | node | pi.model_runtime_create |
| 1.459550 | 0.008862 | 0.008862 | node | pi.create_agent_session |
| 1.468765 | 6.695347 | 0.092745 | node | pi.bind_extensions |
| 1.535067 | 5.983773 | 5.983773 | python | tau2.package_import |
| 7.518896 | 0.001035 | 0.001035 | python | tau2.bridge_module_import |
| 7.519965 | 0.000043 | 0.000043 | python | tau2.install_timing_hooks |
| 7.520029 | 0.031981 | 0.000038 | python | tau2.bridge_ctor |
| 7.520032 | 0.031944 | 0.031938 | python | tau2.get_environment |
| 7.524040 | 0.000006 | 0.000006 | python | tau2.environment_ctor |
| 7.552055 | 0.034294 | 0.034294 | python | tau2.describe_tools |
| 7.608293 | 0.551461 | 0.003134 | python | tau2.load_task |
| 7.608300 | 0.481795 | 0.481795 | python | tau2.task_catalog_load |
| 8.093001 | 0.031520 | 0.031499 | python | tau2.get_environment |
| 8.097477 | 0.000021 | 0.000021 | python | tau2.environment_ctor |
| 8.124586 | 0.000316 | 0.000122 | python | tau2.set_state |
| 8.124609 | 0.000082 | 0.000082 | python | tau2.initialization_action |
| 8.124717 | 0.000067 | 0.000067 | python | tau2.initialization_action |
| 8.124799 | 0.000012 | 0.000012 | python | tau2.initialization_action |
| 8.124824 | 0.000022 | 0.000021 | python | tau2.initialization_action |
| 8.124866 | 0.000012 | 0.000012 | python | tau2.initialization_action |
| 8.124992 | 0.034697 | 0.034697 | python | tau2.describe_tools |
| 8.163479 | 0.000015 | 0.000015 | node | pi.canonical_prompt_published |
| 8.164196 | 0.013999 | 0.013999 | node | pi.prompt_to_first_provider |
| 8.178345 | 0.000698 | 0.000698 | node | pi.provider_context_conversion |
| 8.184641 | 0.000000 | 0.000000 | worker | generation_request_received |
| 8.189448 | 0.098242 | 0.098242 | worker | tokenization |
| 8.288040 | 0.000000 | 0.000000 | worker | llm_request_submitted |

## Session d25c53262d3c4e0caf070e3aa67a659b (train)

Task: `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]`
First LLM submit: 8.865074 s; uncovered: 0.035202 s.

| Start offset s | Inclusive s | Exclusive s | Process | Phase |
|---:|---:|---:|---|---|
| 0.000058 | 0.356213 | 0.356213 | worker | sidecar_start |
| 0.355178 | 0.000000 | 0.000000 | node | node.ready |
| 0.359173 | 1.158998 | 1.158998 | node | pi.sdk_import |
| 1.518409 | 0.001607 | 0.001607 | node | pi.resource_loader_ctor |
| 1.520089 | 0.036175 | 0.023335 | node | pi.resource_loader_reload |
| 1.525047 | 0.012840 | 0.012840 | node | pi.extension_load |
| 1.556343 | 0.019537 | 0.019536 | node | pi.model_runtime_create |
| 1.576697 | 0.009209 | 0.009209 | node | pi.create_agent_session |
| 1.586285 | 7.162452 | 0.090199 | node | pi.bind_extensions |
| 1.651121 | 6.131698 | 6.131698 | python | tau2.package_import |
| 7.782904 | 0.001767 | 0.001767 | python | tau2.bridge_module_import |
| 7.784726 | 0.000073 | 0.000073 | python | tau2.install_timing_hooks |
| 7.784836 | 0.054811 | 0.000074 | python | tau2.bridge_ctor |
| 7.784841 | 0.054736 | 0.054726 | python | tau2.get_environment |
| 7.791714 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 7.839723 | 0.062030 | 0.062030 | python | tau2.describe_tools |
| 7.923212 | 0.821861 | 0.001559 | python | tau2.load_task |
| 7.923222 | 0.705293 | 0.705293 | python | tau2.task_catalog_load |
| 8.629835 | 0.050610 | 0.050600 | python | tau2.get_environment |
| 8.636735 | 0.000010 | 0.000010 | python | tau2.environment_ctor |
| 8.680512 | 0.000418 | 0.000169 | python | tau2.set_state |
| 8.680525 | 0.000090 | 0.000090 | python | tau2.initialization_action |
| 8.680652 | 0.000088 | 0.000088 | python | tau2.initialization_action |
| 8.680768 | 0.000019 | 0.000019 | python | tau2.initialization_action |
| 8.680810 | 0.000035 | 0.000035 | python | tau2.initialization_action |
| 8.680877 | 0.000017 | 0.000017 | python | tau2.initialization_action |
| 8.681007 | 0.063982 | 0.063982 | python | tau2.describe_tools |
| 8.748199 | 0.000014 | 0.000014 | node | pi.canonical_prompt_published |
| 8.748810 | 0.012771 | 0.012771 | node | pi.prompt_to_first_provider |
| 8.761722 | 0.000688 | 0.000688 | node | pi.provider_context_conversion |
| 8.781363 | 0.000000 | 0.000000 | worker | generation_request_received |
| 8.785639 | 0.072221 | 0.072221 | worker | tokenization |
| 8.865074 | 0.000000 | 0.000000 | worker | llm_request_submitted |
