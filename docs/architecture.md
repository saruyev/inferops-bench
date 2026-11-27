# paribus - Architecture (Stage 1)

**paribus** (from *ceteris paribus*, "all else being equal"): an agent that investigates quality and cost problems in an AI pipeline by looking at what actually goes into it. It takes a symptom ("the transcripts hallucinate", "we pay too much", "p99 is bad on some calls"), finds the input slices responsible, prices them, and proves each proposed fix with a controlled experiment.

Stage 1 covers speech-to-text (ASR) pipelines.

Status: design, 2026-11-27. Stage 1 is not built yet.

---

## 1. Purpose

A team runs an ASR pipeline. Something is wrong: invented text, high cost, slow tails, unhappy customers. The usual path is days of manual work: listen to bad examples, guess, write a script for distributions, find that a large part of the input is silence, music or noise, try a filter, hope it did not cut real speech.

paribus does that loop:

1. **Symptom**: a description plus a sample of real inputs and the pipeline's outputs.
2. **Signals**: it decides, per sample, whether the symptom is present, using the best quality signal available (section 4).
3. **Features**: it computes interpretable features of each input and output (speech ratio, music, noise, language, duration, …).
4. **Slices**: it finds slices of the input where the symptom concentrates, and prices them: share of traffic, share of the symptom, money, latency.
5. **Fixes**: it proposes interventions (drop, trim, preprocess, reroute, change model or parameters) and verifies each with a controlled experiment, including the damage the fix causes.
6. **Report**: every claim cites slices and experiments, and every number is computed by deterministic code.

### Why this does not exist yet
- ASR vendors give per-request signals (event tags, word confidence, empty output on silence), not population-level analysis of the customer's input. They bill silence like speech, so they have no reason to tell you half your traffic is garbage.
- Voice-agent evaluation platforms (Hamming, Coval, Cekura, Future AGI) score, alert and run regressions; slicing by audio condition is left to the user.
- Research systems for slice discovery and verified repair (EvalRx, HiBug2, CB-SLICE) work on labelled evaluation sets and repair the model, not the inputs of a deployed pipeline.
- Open-source code has WER calculators and hand-written VAD filters inside individual pipelines, which is evidence of the pain, not a tool for it.

### Why an agent
Feature computation, statistics and experiments are deterministic code. The agent does what that code cannot: read the symptom, choose which signals and features matter, form hypotheses, look at examples, decide which slices are worth testing, design the experiment for each fix, and explain the result. It never produces a number.

### Non-goals
- Not an ASR model and not a VAD. paribus ships no segmentation or filtering algorithm of its own; fixes use public components (for example Silero VAD) or the user's own.
- Not a monitoring SaaS. It runs locally, on the user's data, on demand.
- Not a labelling tool, although it can ask for a small, well-chosen set of labels.
- No production data, configs, algorithms or architecture from any employer. Examples and tests use public datasets and synthetic corruption.

---

## 2. User journey

1. **Try it**: `uvx paribus demo` replays a recorded investigation on the bundled example (public speech with injected silence, music and noise). No key, no download.
2. **Point at data**: `paribus init` writes `paribus.yaml`: where the inputs are (a folder or a manifest), where the existing outputs are (if any), which ASR adapter runs the pipeline for experiments, and what the cost model is.
3. **Profile**: `paribus profile` computes features for a sample (default 2,000 inputs) and shows the input landscape: duration, speech ratio, music, noise, language, duplicates. This step alone often answers the question.
4. **Investigate**: `paribus investigate "transcripts contain text that was never said"` runs the loop: signals → slices → fixes → experiments.
5. **Label on demand**: when the signals disagree or confidence is low, paribus asks for labels on a small, chosen set (`paribus label`: plays audio, shows the transcript, one key per verdict).
6. **Report**: `paribus report` writes markdown with the slices, their price, the verified fixes and their side effects, and the experiments behind each claim.
7. **Resume**: long sessions are checkpointed; `paribus resume <session>` continues after a crash.

Example finding:

> **Slice S2**: inputs with speech ratio < 10% (VAD). 41% of traffic, 78% of hallucination flags (lift 4.3, 95% CI 3.8–4.9).
> **Fix F1** (Silero VAD trim, then drop if no speech): hallucination flags −71% (CI −76…−65%), audio minutes billed −38%, WER on the gold subset unchanged (+0.2 pp, n.s.), speech wrongly removed: 0.6% of words (CI 0.3–1.1%). Evidence: e7, e9, e12.

---

## 3. Concepts

| Concept | Meaning |
|---|---|
| **Sample** | one input (audio file or segment) with metadata, and the pipeline output for it, if known |
| **Symptom** | the problem in the user's words, mapped to one or more signals |
| **Signal** | a per-sample judgement that the symptom is present, with its source and reliability (section 4) |
| **Feature** | an interpretable value computed from an input or an output |
| **Slice** | a rule over features (`speech_ratio < 0.1`, `music_prob > 0.5 and duration > 30s`) |
| **Fix** | an intervention: filter, preprocessing, routing, model or parameter change |
| **Experiment** | a fix applied to a held-out sample, compared with the unchanged pipeline on the same inputs |
| **Finding** | a claim plus the slice and experiment ids and the statistics that support it |

---

## 4. Quality signal ladder

Most real pipelines have no reference transcripts. paribus uses the best signal available and climbs when it is not confident enough.

| Rung | Signal | Cost | Notes |
|---|---|---|---|
| 1 | **Output heuristics**: text where VAD finds no speech, repetition loops, compression ratio, words per second of speech out of range, known filler phrases, empty output | free | model-agnostic |
| 1 | **Model-specific confidence**: e.g. `no_speech_prob`, `avg_logprob` for Whisper-family models, word confidences from APIs | free | declared by the ASR adapter profile |
| 2 | **Cross-model disagreement**: run a second ASR model, preferably from another family, and measure disagreement (WER between the two) | compute | strong without labels; the two models' errors should not be correlated |
| 3 | **Gold labels**: reference transcripts if the user has them, or a small set paribus asks for | human time | paribus chooses which samples to label (disagreements, slice boundaries), not a random sample |

Every signal has a measured reliability: once some gold labels exist, paribus reports how well each cheaper signal agrees with gold, and weights them accordingly. A finding states which rung it rests on.

---

## 5. System overview

```
 paribus CLI ──► Session (LangGraph, SQLite checkpointer)
                   │  frame → profile → signal → slice → fix → experiment → report
                   │
                   │ MCP tools (typed, validated)
                   ▼
 ┌─────────────────────── Analysis Engine (deterministic, no LLM) ──────────────────────────┐
 │ Data        manifest loader, sampling, held-out splits, content hashes                     │
 │ Features    feature packs (audio, ASR output, model-specific); cached per content hash      │
 │ Signals     ladder rungs 1-3, reliability vs gold                                            │
 │ Slices      rule search over features, lift/support/CI, multiple-testing correction         │
 │ Impact      share of traffic, share of symptom, cost, latency per slice                     │
 │ Experiments paired A/B of fixes on held-out samples, side-effect metrics, statistics        │
 └───────────┬─────────────────────────────────────────────────┬───────────────────────────┘
             │ ASR adapters / fix components                   │
             ▼                                                 ▼
  local models │ API models │ user pipeline           Store (SQLite + feature cache + raw results)
                                                               │
 LLM layer (provider-neutral, profiles, record/replay) ◄── Session    ▼
                                                        Report (markdown + plots)
```

Rules that hold the design together:
- **Numbers come only from the engine.** The report reads the store. A finding whose statistics the store does not support is rejected by the claim verifier.
- **The agent never touches raw audio.** It sees features, aggregates, and short transcript excerpts (which can be turned off, section 12).
- **Actions are tool calls with validated arguments.** No shell, no arbitrary code on the user's data in Stage 1.
- **No vendor SDK in the core**, for the agent's LLM and for ASR models alike: both are adapters.

---

## 6. Feature packs

A feature pack declares features, how to compute them, their type and a human-readable description the agent can read. Packs are registered by name; a new pack is a class plus a registration.

**`audio` (Stage 1)**: duration, sample rate, channels, codec; loudness, peak, clipping ratio; estimated SNR; speech ratio, number of speech segments and longest non-speech gap (from a public VAD); music and noise probability (from a public audio-tagging model); spoken-language probability (from a public language-ID model); near-duplicate fingerprint.

**`asr_output` (Stage 1)**: empty output; words, words per second of detected speech; repetition ratio; compression ratio; non-speech-text ratio (text during VAD silence, using timestamps when available); language of the text vs detected spoken language.

**`asr_model` (Stage 1, per adapter)**: confidence-type values the model exposes (e.g. `no_speech_prob`, `avg_logprob`, word confidence).

Feature extractors use public models and libraries only; the concrete choices are pinned in `config/features/*.yaml` with their licenses, and checked to run on a laptop (developed on an M1 Pro, 32 GB). Features are cached by content hash, so re-running an investigation does not recompute them.

---

## 7. Slices and impact

The engine searches for slices; the agent steers the search and interprets it.

- **Candidates** come from three sources: the agent's hypotheses ("silence produces hallucinations"), single-feature thresholds scanned by the engine, and shallow decision-tree rules (depth ≤ 3) fitted to separate symptomatic from healthy samples.
- **Statistics per slice**: support (share of traffic), symptom rate inside and outside, lift, confidence intervals (bootstrap), and multiple-testing correction across all slices tried.
- **Impact per slice**: share of the symptom it explains, audio minutes and money (from the cost model), processing time.
- **Overlap**: slices are reported with their overlaps, so the report does not double count.
- A slice is a candidate, not a cause. Only an experiment turns it into a finding about what to change.

---

## 8. Fixes and experiments

| Fix type | Examples |
|---|---|
| filter | drop inputs in a slice (no speech, duplicates, wrong language) |
| preprocess | trim non-speech with a public VAD, resample, loudness-normalise, denoise |
| route | send a slice to another model or path |
| model / parameters | another ASR model, decoding parameters exposed by the adapter |
| user-defined | a Python callable the user registers as a fix |

**Experiment protocol**
- Held-out sample, never used to find the slice.
- Paired design: the same inputs through the unchanged pipeline and the fixed one.
- Primary metric: the symptom rate on the chosen signal rung.
- **Side effects are first-class**: WER on the gold subset if any, speech wrongly removed (measured with a reference model or gold on what the fix cut), inputs wrongly dropped, latency, cost.
- Statistics: paired bootstrap CIs, effect sizes, multiple-testing correction across fixes.
- Repeat runs where the pipeline is non-deterministic; timing metrics use warmup and interleaving.
- When several fixes combine, leave-one-out shows what each contributes.

A fix is recommended only if it improves the symptom significantly and its side effects stay within limits the user set (default: no significant WER increase on gold, speech removed < 1%).

---

## 9. Agent

### 9.1 Graph (LangGraph)

```
frame → profile → signal → slice → fix → experiment → report
                    ▲                         │
                    └── (budget / confidence)◄┘
```

- **frame**: turn the symptom into signals and hypotheses; decide the sample size.
- **profile**: read the feature landscape; note anomalies (huge silence share, mixed languages).
- **signal**: pick signal rungs; ask for labels when rungs disagree.
- **slice**: steer slice search, read results, inspect example excerpts of top slices.
- **fix**: propose fixes per slice, with the expected effect and the side effect to watch.
- **experiment**: run, read statistics, accept or reject.
- **report**: findings only from verified experiments, plus open questions.

### 9.2 State

```python
class SessionState(TypedDict):
    session_id: str
    symptom: Symptom
    sample: SampleRef  # manifest hash, split ids
    signals: list[SignalRef]
    hypotheses: list[Hypothesis]  # statement, status, evidence ids
    slices: list[SliceRef]
    experiments: list[ExperimentRef]
    findings: list[Finding]
    budget: Budget  # wall, compute minutes, llm_usd, labels requested, used_*
    status: Literal["running", "done", "aborted"]
```

### 9.3 Tools (MCP)

| Tool | Purpose |
|---|---|
| `describe_data` | sample size, available outputs, gold coverage, adapters |
| `profile(features?)` | feature distributions |
| `compute_signal(rung, params)` | per-sample symptom flags and reliability |
| `request_labels(selection, n)` | queue samples for human labelling |
| `search_slices(target_signal, features?, constraints?)` | ranked slices with statistics |
| `slice_examples(slice_id, n)` | feature rows and short transcript excerpts |
| `slice_impact(slice_id)` | traffic, symptom share, cost, latency |
| `run_fix_experiment(fix, slice_id?, n?)` | paired experiment on held-out data |
| `compare(experiment_ids)` | statistics between experiments |
| `record_finding(claim, refs)` | checked by the claim verifier |

### 9.4 Guardrails
- Budget caps (wall time, compute, LLM spend, number of labels asked); on exhaustion the session reports what it has.
- Claim verifier: a finding must cite slices or experiments whose stored statistics support it (direction, significance, the metric named).
- Loop guard: repeated identical tool calls or stalled hypotheses force a strategy change, then a report.
- Context cap per LLM request from the model profile (default 100K tokens); the agent sees summaries and small excerpts only.

---

## 10. ASR adapters

```python
class ASRAdapter(Protocol):
    profile: ASRProfile                      # name, signals it exposes, cost model, limits
    def transcribe(self, audio: AudioRef, params: Params) -> Transcript  # text, segments, timestamps, confidences
```

- Model choice is configuration. Stage 1 ships two local adapters that run on a laptop (one current open model as the default, one from a different family as the reference for disagreement), chosen after checking what runs well on Apple Silicon; a Whisper-family adapter, because many deployed pipelines use it; and one API adapter as an example of the remote path.
- **User pipeline adapter**: a Python callable `transcribe(audio, params) -> Transcript`, so the user's real pipeline (with its own preprocessing) can be the system under test.
- **Offline mode**: if the user already has outputs (production transcripts), profiling, signals and slices work without re-running anything. Only experiments need an adapter.
- Model-specific signals are declared by the adapter profile and become `asr_model` features.

---

## 11. LLM layer

The agent's own LLM goes through one internal protocol; providers are adapters, and no vendor SDK is in the core.

```python
class ChatModel(Protocol):
    profile: ModelProfile

    def complete(self, request: ChatRequest) -> ChatResponse: ...
    def count_tokens(self, request: ChatRequest) -> int: ...
```

Adapters: `anthropic`, `openai_compat` (OpenAI, Ollama, vLLM, OpenRouter, …), `replay`, `fake`. Model profiles (`config/models/*.yaml`, pydantic-validated) hold context limits, tiered pricing, capabilities and strategy names (`structured_output`, `context_trim`, `cost`). A vendor changing terms means editing a profile. A rule DSL is a possible extension, added only if data plus named strategies stop being enough.

Planned agent profiles: Haiku 5.5 (main), Sonnet 5.5 (comparison), a local model via Ollama (free, fully offline).

---

## 12. Privacy

Audio and transcripts are the user's data, often personal.
- Everything runs locally; features and statistics never leave the machine.
- The agent's LLM receives feature summaries, slice statistics and, by default, short transcript excerpts. `privacy: no_text` removes transcript text from everything sent to a remote LLM; `privacy: local_only` refuses any non-local LLM profile.
- Transcript excerpts pass through a redaction step (emails, phone numbers, long digit strings) before they reach a remote model.
- Audio is never sent to the agent's LLM.

---

## 13. Reproducibility

- **Analysis** (features, signals, slices, statistics, reports) is a pure function of the stored data and cached features. `paribus report --session <id>` recomputes it exactly.
- **Agent decisions** are reproduced by record/replay: each LLM response is stored in a cassette keyed by the hash of the normalised request plus provider, model and profile hash; in replay a miss is a hard error.
- **Experiments** that call ASR models store their outputs, so re-analysis does not re-transcribe. Re-running them is a new experiment, compared honestly.
- `paribus demo` and CI run in replay mode: no key, no network, no model download.

---

## 14. Test suite and benchmark

paribus must prove it finds real causes and does not invent them. The test suite builds datasets where the truth is known:

- **Corrupted public speech**: clean speech from public corpora, with injected silence padding, music beds, noise at controlled SNR, codec degradation, language mixing and duplicates, each in a known share of samples. Licenses are checked and recorded per dataset.
- **WildASR** and similar public ASR robustness sets for realistic conditions.

Measured:
- **Slice recovery**: precision and recall of the true corrupted slices; false slices reported.
- **Fix verification**: does an experiment accept the fixes that truly help and reject the ones that do not? Is side-effect damage (speech removed) detected?
- **Signal reliability**: how well rungs 1-2 track gold.
- **Baseline without agent**: compute all features, fit a decision tree, report its rules. paribus has to beat it, or match it with fewer false findings and a better explanation; if it does not, the report says so.
- **Resume**: kill a session at a random step, resume in replay, require the same report.

---

## 15. Configuration and secrets

- Layered settings via pydantic-settings: code defaults → `config/base.yaml` → `config/<env>.yaml` (`PARIBUS_ENV`: dev, ci, live) → `PARIBUS_*` env vars.
- `paribus.yaml` describes one dataset and pipeline; it is the user's file.
- Secrets: profiles name a secret, never a value. Resolution: env var → `*_FILE` → OS keychain (`[keychain]` extra). Held as `SecretStr`, no `.env` autoload; a test asserts that no secret reaches cassettes, logs, results or spans.

---

## 16. Observability
- OpenTelemetry spans for every graph node, LLM call, tool call and experiment, exported to Arize Phoenix (`[tracing]` extra).
- Every session records its config hash: `paribus.yaml`, profiles, feature-pack versions, prompt versions.

---

## 17. Repo layout

```
paribus/
  src/paribus/
    agent/        graph.py  state.py  guards.py  prompts/
    engine/       data.py  features/  signals.py  slices.py  impact.py  experiments.py  verifier.py
    adapters/     asr/  (local models, whisper-family, api, user pipeline)
    fixes/        filters.py  preprocess.py  routing.py
    tools/        mcp_server.py
    llm/          protocol.py  factory.py  profiles.py  strategies/  providers/  cassettes.py
    store/        results.py  feature_cache.py
    report/       report.py  plots.py
    label/        cli_labeler.py
    settings/     app.py  secret_providers.py
    cli.py
  config/         base.yaml  dev.yaml  ci.yaml  live.yaml  models/  features/  asr/
  datasets/       build scripts for the corrupted public test sets (data itself is not committed)
  examples/       demo dataset manifest + recorded session
  cassettes/
  docs/           architecture.md  sessions/   (annotated example investigations)
  tests/          unit/paribus/<package>/test_<file>.py  integration/
  Makefile  pyproject.toml  README.md
```

---

## 18. Stage 1 scope (≈2 weeks) - definition of done

- [ ] Data loading from a folder or manifest; sampling and held-out splits.
- [ ] Feature packs `audio`, `asr_output`, `asr_model`, cached by content hash, running on a laptop.
- [ ] Signal ladder rungs 1-3, including cross-model disagreement and the labelling CLI.
- [ ] Slice search with statistics and impact (traffic, symptom share, cost).
- [ ] Fix experiments: filter, VAD trim, route; paired design; side-effect metrics.
- [ ] ASR adapters: two local models, a Whisper-family adapter, the user-pipeline adapter.
- [ ] LangGraph agent with checkpointed state, budget, claim verifier, loop guard.
- [ ] LLM layer: protocol, profiles, strategies; `anthropic`, `openai_compat`, `replay`, `fake`.
- [ ] CLI: `demo`, `init`, `profile`, `investigate`, `label`, `report`, `resume`.
- [ ] Test suite on corrupted public speech, including the no-agent baseline.
- [ ] `demo` and CI in replay mode.
- [ ] README: the problem, the demo, one annotated investigation.

**Cut line**: if features and slices are not reliable on the corrupted test set by day 5, drop the API adapter and routing fixes, and keep profiling + slices + filter/trim experiments solid.

## 19. Stage 2
- More fixes (denoise, loudness, chunking strategies via public components) and parameter search for fixes.
- Streaming and segment-level inputs, not only whole files.
- `paribus mcp` for interactive use from Claude Code and similar clients.
- Feature packs for other modalities (text pipelines, vision) through the same engine.
- Agent model comparison across profiles.

---

## 20. Decisions

| Decision | Choice |
|---|---|
| Name | paribus |
| Stage 1 domain | ASR pipelines |
| Agent framework | LangGraph (checkpointer required for long sessions) |
| Numbers | computed only by the engine; claims verified against the store |
| Models | ASR models and agent LLMs are both adapters; no vendor lock-in |
| Fix components | public libraries/models or user-provided only |
| Data | public datasets and synthetic corruption only |
| Reproducibility | analysis recomputable from stored data; agent decisions via record/replay |
| Agent tracing | Arize Phoenix |
| Packaging | uv, src layout, Python ≥ 3.13 (dev on 3.14) |

Open:
1. Which two local ASR models (checked on Apple Silicon for speed and quality).
2. Which public VAD, audio-tagging and language-ID models (license, laptop speed).
3. Default thresholds for "acceptable side effects".
4. Whether slice search also uses embeddings (clusters of similar audio) in Stage 1, or only interpretable features.
