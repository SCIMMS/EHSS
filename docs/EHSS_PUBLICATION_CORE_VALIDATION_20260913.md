# EHSS 현재 구현 검증 범위

2026-09-13. 현재 worktree에서 72개 테스트가 통과했다. 실패·오류·skip은 0개다. 실행 로그는 JUnit XML로 보존했다.

| Test module | Passed cases |
|---|---:|
| tests.test_ehss_reference | 9 |
| tests.test_ehss_static | 19 |
| tests.test_ehss_cpu | 11 |
| tests.test_ehss_native_ablation | 3 |
| tests.test_ehss_publication_invariance | 4 |
| tests.test_ccs_cli | 26 |

검증 범위는 단일 구 해석값, impact parameter 반사, 최근접 충돌 순서, 작은 양의 gap, grazing 경계, 내부 trapped 경로와 tail, cap에 따른 잔여 기여, static 준비와 기존 경로 일치, thread/backend 일치, native flat/BVH, XYZ CLI 입력과 오류 처리다.

추가 publication invariant 검사는 비대칭 4구 입력과 같은 4,096개 입사 ray를 함께 회전·평행 이동·크기 배율 0.01/1/100으로 옮겨 collider history, 방향, CCS의 길이 제곱 scaling을 대조했다. Atom 순서를 바꾼 뒤 원래 ID로 history를 환산한 결과도 일치했다.

Co-moving ray의 공변성 검사는 구현의 기하 성질을 확인한다. 고정 입사 sampler에 대한 ensemble 회전 불변성 또는 IMoS의 평균 정의 검증을 대신하지 않는다. 별도 Stage B 회전 진단은 완료됐다. rod9·plate9·asym4 각각 12회전 × 8seed, 총 288개 paired jobs에서 방법별 36개 조건에 대한 사전 Welch/Holm 규칙의 경고는 없었다. 이는 관측된 회전 편차가 검출 기준을 넘지 않았다는 뜻이며 회전 불변성의 증명이나 통계적 동등성 판정은 아니다. 원자료 및 reference hash를 재확인했다. 근거: output/imos113_stage_b_20260913/rotation_summary.json 및 returned/results/rows.json.

검사된 입력과 표본에 대한 결과이며 모든 grazing·접촉·긴 반사 경로에 대한 수학적 증명은 아니다. 새 18분자 reference의 별도 collision kernel 검산과 cap 보완 결과를 함께 사용한다.

Command: `.venv/Scripts/python.exe -m pytest tests/test_ehss_reference.py tests/test_ehss_static.py tests/test_ehss_cpu.py tests/test_ehss_native_ablation.py tests/test_ehss_publication_invariance.py tests/test_ccs_cli.py -q --junitxml=output/ehss_publication_validation_20260913/core_tests.xml`

Evidence: `output/ehss_publication_validation_20260913/core_tests.xml`.
