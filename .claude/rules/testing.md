---
trigger:
  - on_file_path_regex: "tests/.*test_.*\\.py"
  - on_file_path_regex: "src/.*\\.py"
priority: 8
---

# Testing Directives & Test Quality Standards

This document defines testing directives focusing on observable behavior, interface contracts, and reliable verification.

---

## 1. Test Architecture & Design
- **Behavior-Driven Mapping:** Organize tests by component behavior and logical boundary, rather than enforcing rigid 1:1 file mirroring for every utility module.
- **Observable Behavior:** Test observable outcomes, return contracts, and state mutations rather than internal implementation details.
- **AAA Pattern:** Structure test cases clearly using Arrange, Act, and Assert steps.

---

## 2. Test Execution & Database Strategy
- **Realistic Engine Testing:** Use the production database engine (or test containers matching production SQL dialects) when SQL dialect behavior or query optimization matters.
- **Mocking Boundaries:** Limit mocking to external network boundaries, third-party APIs, clock interfaces, and hardware I/O.
- **Stable Semantics over String Matching:** Verify exception types and key semantic phrases rather than relying on brittle, full string error message matching.

---

## 3. Review Signals & Retry Safety
- **Coverage as a Signal:** Treat test coverage of modified code as a quality review signal, not an absolute numerical metric that replaces test depth.
- **Targeted Paths:** Focus test creation in order of priority:
  1. Changed core execution paths
  2. Boundary values and failure modes
  3. High-risk regression paths
  4. Line coverage metrics
- **Retry Budget Boundary:** If the automated fix budget is exhausted after test failures, STOP and report diagnostics to the user. **NEVER commit failing code or broken tests automatically.**