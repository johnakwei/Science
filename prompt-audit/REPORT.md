# Prompt audit: `arxiv_quantum_agent.py`

## Assumptions

- **Scope:** the whole repository. The only prompt surface is the five LLM prompts and the request code in `arxiv_quantum_agent.py`. The `.Rmd` and `.html` files contain no prompts. `Chiral_Graviton_Modes_in_R.Rmd:450` mentions a `Claude_AI_Science_Chats.txt` file, but that file isn't in the repo.
- **Provider:** the code calls Google Gemini (`google.generativeai`, lines 108–118). It does not use the Anthropic SDK. This audit covers only the prompting patterns that apply to any provider. The Claude-specific checks (prefill 400s, `budget_tokens`, adaptive thinking, refusal fallbacks) don't apply to this code, and nothing here proposes switching providers.
- **Target model:** `gemini-2.5-flash`, the newest model the code references (line 404).
- **Provenance:** the file arrived in a single upload (`6cfe55c`, 2026-01-01), so `git blame` can't explain any line. The findings below date the patterns by their idioms and by how the code uses each prompt.
- **Out-of-band dependency:** `AgentEvaluator.evaluate_summary_completeness` (lines 773–784) searches the summary for the strings `executive summary`, `findings`, `trends` and `recommendations`. The proposed diff keeps those section names.

## Summary

| Group | Findings (high / medium / low-flag) |
|---|---|
| 1 – Dated prompt text | 1 / 3 / 0 |
| 2 – Skill files | none (the repo has none) |
| 3 – Tool descriptions | none (the code defines no tools) |
| 4 – Request config & architecture | 0 / 2 / 3 |

**Highest impact:**

1. **The summary prompt asks for content it never receives (F1).** `SummaryGeneratorAgent.process` takes `analyses` but never uses it. Its prompt still asks for "Extracted Equations and their significance". The only input is three abstracts cut to 200 characters, so the model has to invent equations or produce an empty section. This also means the output of the AbstractAnalyzer and MathIdentifier calls never reaches the summary. Those calls cost API quota and change nothing in the result.
2. **Stack of workarounds to force JSON (F2).** The scorer prompt says "Only return the JSON, no other text", and the code strips ```` ```json ```` fences before parsing. Gemini's JSON mode (`response_mime_type: "application/json"`) returns bare JSON, so both the instruction and the fence stripping can go.
3. **Analysis runs on the wrong papers (F5).** The abstract and math analysis run on `papers[:3]`, the first three results from arXiv. Relevance scoring runs afterwards, so the analysis doesn't cover the three papers the summary actually uses.

## Findings

| # | Location | Evidence | Pattern | Why it's obsolete or wrong | Confidence | Action |
|---|---|---|---|---|---|---|
| F1 | `arxiv_quantum_agent.py:598-621` | `analyses: dict` never used; `Abstract: {paper.abstract[:200]}...`; `3. Extracted Equations and their significance` | Keep-list 1: context is never cruft. 1d: unenforced instruction. Also a contract/input mismatch | The prompt asks for equation analysis but receives no equations. It also asks for a "comprehensive" summary of abstracts cut to 200 characters. When context is missing, the model fills the gap with generic or invented text. | **High** | **add**: pass the abstract analysis, the equations and the math analysis into the prompt, send full abstracts, and limit the equations section to the equations provided |
| F2 | `:555-570` | `Only return the JSON, no other text.` + the ```` ```json ```` fence-stripping block | 1b: JSON-forcing scaffold replaced by an API feature | The API's structured-output mode replaces both the prompt instruction and the regex/fence cleanup. The keyword-scoring fallback stays as the error path. | **Medium** (this is Gemini's JSON mode, not Claude structured outputs; check it against your `google-generativeai` version) | **replace-with-API-feature**: `generation_config={"response_mime_type": "application/json"}`, a plain `json.loads`, and `generate_content` accepts a `generation_config` argument |
| F3 | `:484` | `Format as JSON with paper numbers as keys.` | 1d: unenforced instruction | No code parses this output. It goes into `analyses` as a raw string, and (after F1) into a prose prompt. The JSON requirement shapes the output for no consumer. | **Medium** | **rewrite**: `For each paper, give its main research contribution, key methodology, and primary findings.` |
| F4 | `:616` | `1. Executive Summary (2-3 sentences)` | 1f: numeric output ceiling | A sentence count doesn't describe the goal, which is a short opening a researcher can skim. The evaluator only checks that the heading exists. | **Medium** | **rewrite**: `Executive Summary - a short opening paragraph a researcher can skim` |
| F5 | `:716-727` | Analyze steps run before `# Step 4: Score relevance` | Group 4: pipeline wiring | The analyzer and math agents look at the first three arXiv results. The summary covers the top three *scored* papers, so after F1 the summary would get notes on the wrong papers. | **Medium** | **rewrite**: score first, then pass the papers in score order to the analyzer and math agents |
| F6 | `:403-424` | Fallback list including `gemini-1.5-pro`, `gemini-2.0-flash` | 1d: model-version fossil, Group 4 | `genai.GenerativeModel(name)` makes no API call, so the first entry always "succeeds" and the fallback loop never runs. The older entries are dead code pointing at retired models. | **Medium** | **rewrite**: one `MODEL_NAME` (overridable with `GEMINI_MODEL`) |
| F7 | `:471`, `:545` | `paper.abstract[:500]`, `paper.abstract[:300]` | Keep-list 1: context | The analyzer and scorer judge truncated abstracts. arXiv abstracts are around 1–2k characters, so sending them whole costs very little. | **Medium** | **add**: send full abstracts (in the F2/F3 hunks) |
| F8 | `:1-28` (module docstring) | `Built-in tools (Google Search, Code Execution)`, `LoggingPlugin`, `InMemorySessionService`, `python arxiv_quantum_agent_v2.py` | Out of scope (documentation, not a prompt) | None of these exist in the code, and the run command names the wrong file. | Low | **flag** |
| F9 | `:266` | `if "429" in error_str or "quota" in error_str.lower()` | Out of scope (error handling) | The code detects rate limits by matching error strings, so the retry logic can break silently if the error text changes. The SDK has typed exceptions (`google.api_core.exceptions.ResourceExhausted`) for this. | Low | **flag** |
| F10 | `:514-518` | Prompt for the MathIdentifier agent | Group 4: check each model-call site | The equations come from regex (`LaTeXParser`), which is deterministic. The LLM step that explains why they matter is real judgment, so this call should stay. No change. | Low | **flag** (reviewed, keep) |

## Proposed diff

See [`proposed.patch`](proposed.patch). It applies cleanly with `git apply prompt-audit/proposed.patch` and keeps the file's CRLF line endings. Hunks map to findings:

| Hunk (orig. lines) | Finding |
|---|---|
| `@@ -391` + `@@ -399` | F6 |
| `@@ -434` | F2 (adds `generation_config` pass-through) |
| `@@ -468` + `@@ -476` | F7, F3 |
| `@@ -542` + `@@ -552` | F7, F2 |
| `@@ -604` | F1, F4 |
| `@@ -714` | F5 |

I compiled the patched file with `py_compile`. I didn't run it end to end, because `google-generativeai` isn't installed here and the run needs a Gemini API key.

## Verifying the changes

- Run one query before and after the patch and compare the "Extracted Equations" section. Before, it should be invented or empty. After, it should discuss only the regex-extracted equations or say that none were found.
- Check that the relevance scores parse without falling back to keyword scoring. Before the patch, the `Could not parse scores` warning in the logs shows how often the fallback fires.
- Check that the `AgentEvaluator` completeness score doesn't drop, since the section headings are unchanged.
