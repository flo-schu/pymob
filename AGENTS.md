# AI Guidelines for software development

This file is split into two parts:

1. **General Guidance**: project-independent rules. Can be copied to other repositories as is.
2. **Package specific guidance**: rules, commands and contracts specific to this repository.

If a rule conflicts with an explicit instruction from the user, the user's instruction wins.
Log the deviation.

## General Guidance

### Rules

- Assume that the package was written by scientists, not professional software engineers. This means code should remain understandable.
- Human-maintainability is of utmost importance. This means that newly implemented code should be similar in style to the existing code. If unsure, fall back to established style guides (e.g. PEP8).
- Keep the changes minimal. Re-use existing code patterns.
- Use private methods/functions (`_name`) for internal functionality.
- On implementing new functionality, prefer established, maintained packages over custom implementations.
- No hardcoded configuration (paths, credentials, magic options). Pass it as arguments or config.
- It is okay to raise errors. Always ask when creating fallback options/defaults that swallow errors.
- No sensitive information (data, passwords, tokens, API keys) in code, logs or commits.
- Never directly upload sensitive datasets to AI. Always require user confirmation before reading raw data.
- Use typing where possible, but favor readability over complex (and cryptic) type annotations.
- Avoid changing public API signatures. Always ask before doing so.
- Ask before adding new dependencies or changing version pins.
- Ask before pushing, merging, tagging or bumping versions.
- Do not reformat code that is unrelated to the task (mass reformatting makes diffs hard to review).
- Back claims about code behavior by reading or running the code. Do not guess.

### Logging

Document important aspects of the code and AI work in an `ai-logs` directory with timestamped subfolders
(`ai-logs/<YYYY-MM-DD_HHMM>_<short-topic>/`).
This folder will be a public record of AI decisions.

**Make sure that sensitive information (e.g., data, passwords, tokens, API keys) is never logged.**
Describe the shape and structure of data instead of its contents.

In timestamped folders, note:

- Assumptions. What are they backed by? Noted in summary.md
- Design decisions. Taken on which grounds? What was the reasoning? Noted in summary.md
- Raised questions and user answers. Noted in summary.md
- User approved changes of existing tests. Changes of existing tests without explicit user consent are not permitted. Noted in summary.md
- Scratch files for code exploration. Summary of behavioral insights noted in summary.md
- AI Plans (separate document, summary noted in summary.md)
- Raw test results (no warnings), one file per run (e.g. `tests_<phase>.txt`). Summary noted in summary.md

Maintain the following separate documents in the main ai-logs folder:

- `package_map.md`: a working document that maps the structure and behavior of the package for later consultation by yourself and other AI models. Update it with new insights from the current task.
- `bugs.md`: Discovered bugs. Keep a running list of open and closed bugs. Also note known bugs in inline comments (`# BUG:`). Also note in timestamped folder's logs.
- `tasks.md`: Open tasks. Keep a running list of open and closed tasks. Also note specific code tasks in inline comments (`# TODO:`). Also note in timestamped folder's logs.

### Code documentation

- Write numpy style docstrings for each function.
- Keep docstrings of private methods concise, writing only minimal docstrings or omitting docstrings
  altogether if the functions are self-explanatory, or can be explained sufficiently well with
  inline comments.
- Write comprehensive docstrings for public methods, understandable also by inexperienced users. Keep parameter descriptions concise. Add an `Examples` section to explain the methods.
- Document new features in jupyter notebooks. Either integrate the description of a new feature into an existing notebook if it fits, or create a new notebook for it.
- Update README file with relevant changes.
- Maintain a CHANGELOG file ([Keep a Changelog](https://keepachangelog.com) format, section `[Unreleased]`).

### Testing

- pytest is used as a testing framework.
- Follow behavioral testing. This means that we do not test the implementation, which can change, but we test the expected behavior, which should remain constant.
- Never change existing tests. If you have absolute certainty that the test is wrong, present the user with proof and a suggested fix. Always log.
- Favor lean, explicit test suites over complex and large test suites.
- Reuse existing fixtures instead of writing new setup code.
- Mark new long-running tests with the existing markers (e.g. slow, integration).
- Run tests with silenced warnings. Log test output in ai-logs.
- Fast tests exclude tests marked as slow or integration. Long-running tests are only run in final testing; for parameterized long-running tests (> 1 min per test) a representative subset is enough.

### Commits

- Small, focused commits with an imperative summary line (e.g. `Add input validation to data reader`).
- Mark incomplete or broken states with a `(wip)` prefix.
- Don't commit generated results, caches, large data files or notebook outputs.

### Bugs

- Note bugs when discovered.
- Only fix on explicit user approval.

