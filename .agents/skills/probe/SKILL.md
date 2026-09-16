---
name: probe
description: Explore hypotheses, conduct empirical scratch experiments, and establish high-reasoning design rationale before contracting.
---

# Probe Protocol

High-reasoning exploratory protocol to formulate hypotheses, uncover failure modes, and empirically validate solutions via scratch experiments before freezing contracts.

## High-Reasoning Exploration Philosophy

As the high-reasoning architect, your cognitive budget is 100% dedicated to **causal root-cause discovery, autonomous alternative generation, and falsification-driven stress testing**.
Do NOT write `contract.json` or full test suite skeletons in this phase.
Focus on: *What is the fundamental causality behind this state? What competing architectures could resolve it? Under what realistic boundary conditions does each approach break?*

## Directives

1. **Focused Scope & Context Alignment (No Over-Exploration)**:
   - Do NOT run broad, unbounded repository scans. Confine code inspection to the target module and its immediate 1-depth callers or fixtures using targeted `rg`/`view_file`.
   - Collect domain references and historical decisions when needed:
     ```bash
     uv run python tools/agent_skills/spec_init.py --feature <feature_name> --domain <domain> --query <keyword>
     ```
   - **Challenge Prior Assumptions (No Sacred Cows)**: Historical ADRs and caveats are past context, not infallible dogma. If current empirical evidence contradicts a past decision, actively challenge and falsify the old assumption rather than anchoring to it.

2. **Autonomous Alternative Generation (No Anchoring Bias)**:
   - Do NOT settle on the first plausible fix. Formulate competing architectural approaches driven strictly by the inherent nature of the problem.
   - Evaluate engineering trade-offs (complexity vs. safety vs. blast radius vs. maintainability) without forcing predetermined dichotomies.

3. **Falsification-Driven Empirical Probing**:
   - Do NOT merely verify that a hypothesis works under happy paths; actively identify under what domain-specific conditions it **fails**.
   - Interrogate implicit assumptions regarding data distribution, state transitions, timing, concurrency, and boundary constraints specific to this system.
   - Run lightweight scratch probes:
     - Create a temporary probe script: `scratch/probe_<topic>.py`
     - Run via `uv run python scratch/probe_<topic>.py`
     - Measure real values, state transformations, or execution bottlenecks directly on actual or synthetic workloads.
     - Never rely on speculative assumptions when an empirical command can prove or falsify them.

4. **Invariants & Performance Budget Formulation**:
   - Define strict Fail-Closed invariants and domain boundaries (.agents/rules/quant.md, performance.md, code-style.md).
   - If touching backtest, training, or bulk I/O, draft a realistic `performance_budget`:
     - `{ expected_data_scale, memory_target_mb, storage_format, dtype_precision, chunking_strategy }`
     - Do NOT introduce artificial truncation or shortened windows.

5. **Seamless In-Session Transition to `/spec`**:
   - `probe` and `spec` run in the same model and session. Do NOT serialize intermediate JSON files to disk.
   - The diagnosis, invariants, and architectural decisions established here flow directly through the conversation context into `/spec`.

## Chat Output Format

Keep chat response clear, intuitive for humans, and token-efficient. Strictly avoid dense jargon walls or redundant data dumps.

**Output Directives:**
- **Human-Friendly & Intuitive Context**: Explain the problem, root cause, and solution in clear, natural Korean. Use plain analogies or everyday phrasing so any stakeholder can grasp the situation in seconds without needing clarification.
- **Token Efficiency & Anti-Redundancy**: Do NOT repeat the same numbers or conclusions across both narrative text and markdown tables. Avoid decorative/redundant tables unless presenting raw multidimensional data.
- **High-Scannability Structure**: Keep each bullet to 1-2 focused lines. Separate bold core takeaway from supporting details.
- **Zero Typing Next Step**: Point directly to `/spec` without requiring the user to type `--feature` or other CLI arguments.
- **Language Requirement**: All output rendered to the user MUST be written in Korean (한국어).

---

### 🔬 [PROBE] <기능/토픽 제목>

#### 💡 한눈에 이해하는 문제와 해법
- 🔍 **상황**: <전문 용어 난사 대신, 직관적인 비유나 일상 언어로 어떤 결함/증상인지 1-2줄 요약>
- ⚙️ **근본 원인**: <숨겨진 제약이나 데이터 흐름상의 진짜 원인 1-2줄>
- 🛠️ **해결 방식**: <왜 다른 대안 대신 이 방식을 택했는지 핵심 근거 1-2줄>
- 🎯 **기대 효과**: <시스템과 사용자에게 주는 실질적 개선 수치/효과 1줄>

#### 🛡️ 주의할 점 (Caveats)
- ⚠️ <다음 작업자가 오해하거나 놓치기 쉬운 전제조건 또는 경계 규칙 1-2줄>

---
👉 다음 단계: `/spec`
