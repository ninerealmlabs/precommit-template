# Instructions

Revise/Expand **tests only**. Critically review the associated source code and any existing tests to ensure the tests validate **intended behavior and contracts**, not merely the current implementation.

## Scope and Constraints

- **You may only change tests. Do not change production/source code.**
- Prefer small, incremental, testable edits.
- If the source code's intent is unclear or underspecified, ask targeted questions **before** making assumptions that would lock in incorrect behavior.

## Process

1. **Determine intent and contracts**

   - Infer the intended behavior from names, docstrings, type hints, usage patterns, and existing tests.
   - If intent is ambiguous, explicitly call out assumptions and ask clarifying questions.

2. **Validate tests against intent (not implementation)**

   - Ensure tests verify observable outcomes and stable contracts.
   - Avoid assertions coupled to internal details (private functions, exact log lines, internal data structures), unless those details are explicitly part of the public contract.

3. **Identify coverage gaps**

   - Missing edge cases and boundary conditions
   - Negative/error paths and exception types/messages (when part of contract)
   - Incorrect or weak assertions (false positives)
   - Tests that fail to distinguish between correct and incorrect behavior
   - Flaky tests (time, randomness, ordering, concurrency)

4. **Optional: highlight code issues, but do not modify code**

   - If you notice source changes that would improve clarity/testability, list them as recommendations only.
   - Do not implement them.

5. **Plan the changes**

   - Provide a step-by-step todo list in **markdown**, wrapped in triple backticks.
   - Each step should be small and independently verifiable.

6. **Implement incrementally**

   - Apply changes in small steps, keeping tests readable and focused.
   - After changes, provide a concise summary of what improved and why.

## Deliverables

- A todo plan (markdown checklist in triple backticks)

- Updated unit tests (only)

- A brief summary of:

  - what behaviors/contracts are now covered
  - what gaps remain (if any)
  - any notable risks/assumptions

## Testing Guidelines

### Framework and Structure

- Use **pytest**.
- Follow the **Arrange–Act–Assert** pattern.
- Prefer **plain test functions**; use test classes only when they add structure (shared setup via fixtures is usually better than class state).
- Keep one behavior per test; avoid multi-assert "kitchen sink" tests unless the assertions are tightly related.

### Naming and Organization

- **Unit tests mirror source modules**: one source module → one unit test module (parallel structure)

- **Integration tests** (if/when present) are grouped by **scenario / contract / intent**, not source structure.

- Test naming:

  - `test_<unit>_<scenario>_<expected_result>`
  - Prefer describing the behavior, not the implementation.

### Unit vs Integration Boundaries

- **Unit tests**

  - Validate a module's public behavior in isolation
  - Avoid network, filesystem, real DB, real time/sleep
  - Dependencies should be mocked/stubbed/faked at module boundaries

- **Integration tests**

  - Validate interactions across modules/boundaries
  - Use real infrastructure only when explicitly intended and controlled

### Isolation, Mocks, and Patching

- Mock external dependencies and I/O.
- Patch at the **import location used by the unit under test** (avoid patching the "definition site" when the unit imported it elsewhere).
- Prefer pytest's `monkeypatch` for simple patching; use `unittest.mock` for call assertions and more complex mocking.
- Avoid mocking internal implementation details; mock only at boundaries.

### Coverage and Scenarios

- Cover:

  - happy path(s)
  - boundary conditions
  - invalid inputs / error paths
  - exceptions (type and message only when message is part of the contract)

- Prefer table-driven coverage with `pytest.mark.parametrize` for input variation.

### Assertions and Clarity

- Assertions should clearly express \*why- the behavior matters.

- Prefer precise assertions over vague checks:

  - assert returned value, state change, and/or emitted events
  - avoid "it didn't crash" tests unless explicitly valuable

- Add assertion context only when it materially improves diagnosis (don't over-annotate).

### Fixtures and Test Data

- Use fixtures to:

  - share setup
  - remove duplication
  - keep tests focused on the act/assert parts

- Keep fixtures:

  - small
  - explicit
  - scoped appropriately (`function` by default)

- Use realistic, meaningful test data; avoid overly synthetic values unless targeting a boundary.

- Avoid logging inside tests; rely on assertion clarity.

### Determinism and Reliability

- Tests must be deterministic:

  - no reliance on wall-clock time, random ordering, global state, or test execution order
  - if time/randomness is part of the code, freeze/patch it
