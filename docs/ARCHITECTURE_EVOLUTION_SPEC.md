# HOPE Stethoscope — Architectural Evolution Prompt

> **Provenance note (added by the Phase 0/1 audit-and-implementation pass, 2026-09-01):**
> This file is a verbatim copy of the task specification supplied for the
> architectural evolution of this repository. It is saved here, unedited, so
> that the requirements survive independently of any particular chat session
> or sandbox. See `docs/ARCHITECTURE_DELTA_REPORT.md` for the Phase 0 audit
> performed against this spec, and `TASKS.md` at the repo root for live
> status and resumption notes. Everything below this line is the original
> document, reproduced without modification.

---

You are working on the existing **HOPE Stethoscope** repository.

## Confidentiality and project boundaries

Treat everything in this repository and everything discussed in the context of this project as confidential project material.

Do not publish, reproduce, redistribute, or disclose the repository's source code, implementation details, research ideas, architectural decisions, experimental findings, prompts, or discussions outside the work required here.

Do not replace existing project-specific ideas with generic implementations merely because a generic library exists. Understand the existing architecture first and evolve it without destroying the work already done.

The repository itself is the source of truth for the current implementation. Inspect the code, tests, documentation, experiment records, and existing architecture before making changes.

---

# 1. Project objective

The long-term goal of **HOPE Stethoscope** is:

> Build an instrumentation and intervention system that allows us to inspect, monitor, understand, experimentally modify, and causally analyse the internal computation of an LLM/Transformer to the greatest technically practical extent.

The initial implementation concentrated heavily on the **KV cache**, including real Hugging Face model experimentation such as Google Gemma.

That work is valuable and should be preserved.

However, the project must now evolve from:

> **KV-cache observability/intervention**

into:

> **Model-wide Transformer observability + intervention + experimentation + causal analysis.**

The desired end state is not merely a debugging framework.

It should become a practical **neural-network stethoscope** for Transformer-based models.

The system should eventually let an investigator ask questions such as:

* What is happening inside this Transformer?
* What components exist?
* What tensors/state exist at every meaningful architectural boundary?
* How does a token representation evolve through the layers?
* Which attention heads are actually contributing?
* Which neurons activate for a given token/concept?
* How does the KV state evolve?
* What does each layer appear to be contributing?
* What would happen if I zeroed a head?
* What would happen if I silenced a neuron?
* What happens if I perturb a residual stream?
* What happens if I modify Q/K/V?
* Where does an intervention first propagate?
* Which downstream representations change?
* How does the intervention affect logits and final output?
* Can an observed correlation be distinguished from a causal effect?
* Can we compare two model states or two experiments?
* Can we reproduce an experiment exactly?

---

# 2. Critical architectural principle

Do **not** build separate, unrelated systems for:

* KV inspection
* attention inspection
* neuron inspection
* residual inspection
* weight inspection
* interventions
* HOPE compression
* experiment logging

Instead establish a common architecture:

```text
                    HOPE STETHOSCOPE
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
     Observation      Intervention       Experiment
       Layer             Layer             Layer
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                  Measurement / Analysis
                           │
                           ▼
                  Evidence / Provenance
                           │
                           ▼
               Visualization / Reporting
```

The fundamental abstraction should be:

```text
COMPONENT
    ↓
DISCOVER
    ↓
OBSERVE
    ↓
OPTIONALLY INTERVENE
    ↓
MEASURE
    ↓
COMPARE
    ↓
EXPLAIN / REPORT
```

The existing KV-cache implementation should become the first mature implementation of this general abstraction rather than remaining the architectural centre of the entire system.

---

# 3. Current implementation assessment

Preserve and build upon the following capabilities already present in the repository.

## Strong / relatively mature

* KV cache observation
* Actual Hugging Face cache integration
* `CacheTap` interception around the real cache update path
* KV statistics
* KV inspection
* KV visualisation
* KV modifiers/interventions
* storage of experimental information
* experiment/debugging infrastructure
* HOPE-related weight/parameter manipulation
* Gemma-oriented experimentation
* tests around several of the above

The discovery that module-name-based attention identification is insufficient for multimodal architectures such as Gemma 3 is especially important.

Do not regress from the lesson learned there.

Where the framework exposes an authoritative semantic interface, prefer that interface over fragile inference from arbitrary module naming.

For example:

```text
Cache.update()
```

is semantically more meaningful for cache instrumentation than trying to infer cache behaviour from module names.

Apply the same principle to the rest of the Transformer.

---

# 4. Target architecture

Build the following conceptual layers.

## 4.1 Model Adapter Layer

Introduce a model/framework abstraction that allows the Stethoscope core to work across multiple Transformer implementations.

Conceptually:

```text
                    Stethoscope Core
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
     HuggingFace       PyTorch             Future
       Adapter           Adapter            Adapter
        │
   ┌────┼───────┐
   │    │       │
 Gemma Llama  Qwen/...
```

The core must not become hardcoded to Gemma.

The model adapter should be responsible for discovering:

* model architecture
* layers
* attention
* Q/K/V/O
* MLP components
* normalization
* residual boundaries
* embeddings
* output/LM head
* cache
* multimodal components
* modality boundaries where applicable

Do not assume every Transformer follows exactly the same implementation structure.

---

# 5. `ModelManifest`

Create a canonical machine-readable architecture representation.

For example:

```python
ModelManifest(
    model_id=...,
    framework=...,
    architecture=...,
    modalities=...,
    layers=...,
    attention=...,
    mlp=...,
    normalization=...,
    embeddings=...,
    output_head=...,
    cache=...,
)
```

It should capture, where determinable:

### Global

* model identifier
* framework
* architecture type
* parameter count
* dtype
* device
* context length
* tokenizer
* modalities

### Attention

* MHA / MQA / GQA
* number of Q heads
* number of KV heads
* head dimension
* projection dimensions
* RoPE or alternative positional mechanism
* attention masking
* sliding-window behaviour
* grouped heads where applicable

### MLP

* architecture/type
* hidden dimensions
* activation function
* gate/up/down projections
* gated vs non-gated

### Normalization

* LayerNorm / RMSNorm / other
* pre-norm / post-norm structure
* epsilon
* parameterisation

### Embedding/output

* embedding dimensions
* vocabulary size
* tied/untied embeddings
* LM head

### Cache

* cache implementation
* layer count
* KV shape
* cache strategy
* static/dynamic structure where detectable

The manifest must be generated from actual model structure/configuration whenever possible rather than guessed.

---

# 6. `ComponentRegistry`

Create a canonical registry of semantically meaningful model components.

The registry should be able to identify components such as:

```text
Model
Embedding
Block
Norm
Residual
Attention
Attention.Q
Attention.K
Attention.V
Attention.Scores
Attention.Probability
Attention.Context
Attention.O
KVCache
MLP
MLP.Gate
MLP.Up
MLP.Activation
MLP.Down
Neuron
LMHead
Logits
```

Each component should have a stable identity.

For example:

```text
model/layer/17/attention
model/layer/17/attention/q
model/layer/17/attention/k
model/layer/17/attention/v
model/layer/17/mlp
model/layer/17/mlp/neuron/183
model/layer/17/residual/post_attention
```

Do not make identifiers dependent solely on Python object names.

---

# 7. Generic observation API

Create a model-independent observation mechanism.

Conceptually:

```python
with stethoscope.observe(
    model,
    components=[
        "attention",
        "mlp",
        "residual",
        "kv_cache",
    ],
):
    output = model(...)
```

Observation must support at least:

```text
INPUT
PRE_FORWARD
INTERNAL_STATE
OUTPUT
POST_FORWARD
```

Where technically possible also support:

```text
BACKWARD
GRADIENT
PARAMETER_UPDATE
```

The observer should allow selective capture.

Do not automatically capture every tensor in the model.

---

# 8. Semantic boundaries rather than operation-level noise

Do NOT build the system by recording every PyTorch operation.

Avoid creating enormous event streams consisting of:

```text
reshape
view
transpose
cast
slice
matmul
...
```

Those are implementation details and usually not useful to an investigator.

Prefer semantic architectural boundaries:

```text
Embedding
Attention.Q
Attention.K
Attention.V
Attention.Scores
Attention.Softmax
Attention.Context
Attention.Output
MLP.Gate
MLP.Up
MLP.Activation
MLP.Down
Residual
Norm
KVCache
LMHead
Logits
```

The system should expose lower-level operations only when explicitly requested for deep debugging.

---

# 9. Standard observation event

Introduce a standardized event/data structure.

Conceptually:

```python
ObservationEvent(
    experiment_id=...,
    model_id=...,
    component_id=...,
    component_type=...,
    layer=...,
    token_positions=...,
    timestep=...,
    phase=...,
    tensor_metadata=...,
    tensor_reference=...,
    statistics=...,
    provenance=...,
)
```

Do not necessarily store every full tensor permanently.

Support configurable:

```text
full tensor
sampled tensor
summary statistics
selected tokens
selected layers
selected heads
selected neurons
time window
```

This is essential for memory efficiency.

---

# 10. Residual-stream instrumentation

This is one of the highest-priority capabilities.

The system must be able to track the residual representation as it evolves through the model.

Conceptually:

```text
R0
 ↓
R1
 ↓
R2
 ↓
...
 ↓
Rn
 ↓
LM Head
```

Support observation of:

* residual vector
* vector norm
* change from previous layer
* cosine similarity
* distance metrics
* dimensional statistics
* selected-token trajectories
* token-to-token similarity
* selected direction projections

For a selected token, the investigator should eventually be able to see:

```text
token
    ↓
embedding
    ↓
layer 0 representation
    ↓
layer 1 representation
    ↓
...
    ↓
layer N representation
    ↓
logits
```

This should become one of the defining features of the Stethoscope.

---

# 11. Logit lens / vocabulary lens

At selected layers, derive the model's intermediate prediction where mathematically and architecturally meaningful.

Conceptually:

```text
Residual[layer]
       ↓
appropriate normalization
       ↓
LM Head
       ↓
vocabulary logits
       ↓
top-k tokens
```

Allow comparison such as:

```text
Layer 0: ...
Layer 4: ...
Layer 8: ...
Layer 12: ...
Layer 20: ...
Layer N: ...
```

Record:

* top-k token predictions
* logits
* probabilities
* entropy
* rank of target token
* probability trajectory
* logit deltas between layers

Do not claim that intermediate logit predictions literally represent "what the model thinks." Describe them as intermediate readouts.

---

# 12. Attention instrumentation

Expand existing attention support into a complete semantic system.

Where technically available, capture:

```text
Q
K
V
QKᵀ scores
mask effects
softmax probabilities
context
O projection
```

Support investigation by:

* layer
* head
* token
* token pair

Calculate useful derived metrics:

* attention entropy
* concentration
* top attended tokens
* attention distance
* local/global distribution
* attention mass by token class where labels are available
* Q/K/V norms
* head-level statistics

The existing KV instrumentation must remain fully supported.

---

# 13. KV cache becomes a first-class component

Retain and generalize the existing KV implementation.

The KV layer should support:

```text
observe
summarize
visualize
compare
perturb
replace
mask
scale
noise
swap
decay
low-rank modification
```

The existing `CacheTap` mechanism should remain the primary proven path for Hugging Face cache-aware models where appropriate.

Do not replace working KV functionality merely to make it "generic."

Generalize it behind the common component/intervention architecture.

---

# 14. MLP / neuron instrumentation

Add systematic MLP introspection.

For gated or non-gated architectures, expose meaningful intermediate states.

Potential signals:

```text
MLP input
gate
up projection
activation
down projection
MLP output
```

At neuron level support:

```text
activation value
activation frequency
activation magnitude
maximum activation
distribution
selected-token activation
top activating tokens
```

Where feasible, support identification of:

* inactive/dead neurons
* highly active neurons
* highly selective neurons
* broadly active neurons
* neurons correlated with particular token categories

Do not make semantic claims merely from high activation. Treat activation as evidence requiring causal testing.

---

# 15. HOPE integration

HOPE must become a first-class intervention/analysis capability, not a separate universe.

The relationship should be:

```text
Stethoscope
    │
    ├── observes neuron
    │
    ├── measures neuron behaviour
    │
    ├── performs intervention
    │
    └── evaluates consequence
             │
             ▼
          HOPE tools
```

HOPE should therefore be usable to investigate questions such as:

```text
Which neurons can be pruned?
Which neurons can be merged?
What capacity does a neuron appear to carry?
What happens when the neuron is removed?
How much output distortion occurs?
```

The Stethoscope should provide the experimental instrumentation around those operations.

Do not redefine the Stethoscope as a HOPE compression framework.

---

# 16. Generic intervention framework

Create a first-class intervention abstraction.

Conceptually:

```python
Intervention(
    component=...,
    location=...,
    operation=...,
    parameters=...,
    scope=...,
)
```

Examples:

```text
Zero
Scale
Mask
Noise
Swap
Clamp
Patch
Project
Quantize
Replace
```

Apply interventions to semantic components.

Examples:

```text
Zero attention head 3 at layer 17

Scale V by 0.5 at layer 12

Zero neuron 183 at layer 19

Add vector Δ to residual stream after attention at layer 10

Replace K state with baseline K

Patch a specific token's residual representation

Mask selected KV positions
```

---

# 17. Intervention safety model

Separate interventions into categories.

## Non-destructive

```text
observe
measure
profile
```

## Reversible

```text
zero
scale
noise
mask
swap
clamp
patch
project
```

## Structural

```text
remove neuron
remove head
merge neuron
remove layer
change parameter topology
```

## Learned / optimisation-based

```text
HOPE
DEFT
LoRA
fine-tuning
gradient modification
```

The framework must know which type it is dealing with.

Every intervention must record:

* target
* operation
* parameters
* original state where relevant
* modified state where relevant
* reversibility
* experiment ID

---

# 18. Causal experiment engine

This is the long-term differentiating capability.

The system should support:

```text
BASELINE
   ↓
INTERVENTION
   ↓
RE-RUN
   ↓
COMPARE
   ↓
CAUSAL REPORT
```

Record differences across:

### Output

* output tokens
* sequence probability
* target-token probability
* logits
* loss
* perplexity where meaningful

### Internal state

* residual changes
* attention changes
* MLP changes
* neuron activation changes
* KV changes
* downstream propagation

### Temporal/layer propagation

Determine:

* first layer affected
* maximum divergence
* recovery/convergence if any
* final divergence

Example conceptual report:

```text
Experiment #184

Model:
Gemma ...

Input:
...

Intervention:
Layer 17
Attention Head 3
V scaled to 0

Output changed:
YES

First downstream divergence:
Layer 18

Maximum residual divergence:
Layer 23

Largest logit change:
...

Top affected tokens:
...

Final output:
...
```

Do not claim causality merely because two signals correlate.

Causal claims should be tied to explicit interventions.

---

# 19. Baseline vs intervention protocol

Every causal experiment should be reproducible.

Capture:

```text
model identifier
model revision/version
weights/source
tokenizer
prompt/input
generation configuration
random seeds where applicable
dtype
device
framework versions
intervention
observation configuration
software revision / git commit
```

Baseline and intervention runs must be directly comparable.

Where possible:

```text
same input
same generation parameters
same seed
same hardware/software configuration
```

---

# 20. Experiment provenance

Use the existing SQLite/event/knowledge-graph infrastructure, but expand the ontology.

A useful conceptual graph is:

```text
Model
 ├── Layer
 │    ├── Attention
 │    │    ├── Head
 │    │    ├── Q
 │    │    ├── K
 │    │    ├── V
 │    │    └── O
 │    │
 │    ├── MLP
 │    │    └── Neuron
 │    │
 │    ├── Norm
 │    └── Residual
 │
 ├── Token
 ├── Parameter
 └── Experiment
```

Relationships:

```text
Token ──represented_by──> Activation

Head ──attended_to──> Token

Neuron ──activated_by──> Token

Layer ──transforms──> Residual

Parameter ──belongs_to──> Component

Experiment ──modified──> Component

Experiment ──caused_change_in──> Component

Experiment ──changed──> Output
```

The knowledge graph should eventually permit questions about relationships between observations and interventions.

---

# 21. Visualization strategy

Do not optimise for pretty dashboards first.

Optimise for useful scientific inspection.

The visualization system should eventually support:

### Architecture

```text
model → layers → components
```

### Layer timeline

```text
layer 0 ── layer 1 ── layer 2 ── ... ── layer N
```

### Residual trajectory

```text
token → representation through layers
```

### Attention

```text
token × token
layer × head
```

### KV

```text
layer × token × head
```

### MLP

```text
layer × neuron
```

### Causal propagation

```text
intervention
    ↓
layer 17
    ↓
layer 18
    ↓
layer 19
    ↓
...
    ↓
output
```

Prefer visualizations that answer a question.

---

# 22. Memory/performance architecture

This system can easily become unusable if it stores every tensor.

Design capture modes:

```text
NONE
SUMMARY
SAMPLED
SELECTED
FULL
```

Allow selection by:

```text
layer
component
head
token
neuron
timestep
experiment
```

For large tensors, use:

* lazy storage
* memory mapping
* chunking
* compression where appropriate
* sampled recording
* derived statistics
* on-demand materialisation

The observer must add minimal overhead when disabled.

---

# 23. Do not instrument everything by default

The default mode should be lightweight.

For example:

```python
stethoscope.attach(
    model,
    observe=["architecture"]
)
```

or:

```python
stethoscope.attach(
    model,
    observe=["residual"],
    layers=[10, 20, 30],
    tokens=[5, 10],
    capture="summary",
)
```

Deep instrumentation should be explicitly requested.

---

# 24. Real-model validation

A major project rule:

> Synthetic correctness is not sufficient.

For each important capability use the progression:

```text
Unit test
    ↓
Synthetic Transformer
    ↓
Small real Transformer
    ↓
Real Hugging Face model
    ↓
Causal intervention
    ↓
Independent verification
```

The Gemma experiment should remain an important reference validation.

Where applicable, test against more than one architecture.

At minimum, design the system so it can eventually support different families rather than silently assuming Gemma's structure.

---

# 25. Do not overclaim

The repository currently demonstrates some things well and others only partially.

Maintain this distinction.

Documentation must explicitly distinguish:

```text
Implemented
Tested
Validated on synthetic model
Validated on real model
Partially validated
Not yet validated
```

Do not report toy-model results as evidence of production-scale behaviour.

Do not use misleading reconstruction or compression metrics.

Do not claim that an intervention is causal unless the experiment actually performs an intervention and establishes the relevant comparison.

---

# 26. Required architectural phases

Implement this evolution in phases.

## Phase 0 — Repository audit

Before modifying code:

1. map the current repository
2. identify existing subsystems
3. identify entry points
4. identify tests
5. identify Gemma integration
6. identify `CacheTap`
7. identify KV modifiers
8. identify experiment persistence
9. identify HOPE machinery
10. identify duplicated or conflicting abstractions
11. identify current limitations
12. document architectural risks

Do not rewrite anything before understanding the existing system.

---

## Phase 1 — Common component architecture

Implement:

```text
ModelAdapter
ModelManifest
ComponentRegistry
ComponentID
ObservationEvent
```

without breaking existing KV behaviour.

---

## Phase 2 — Generic observation layer

Implement:

```text
Observer
CapturePolicy
ObservationEvent
Storage backend
```

Migrate KV instrumentation onto this abstraction while preserving compatibility.

---

## Phase 3 — Residual instrumentation

Implement residual-stream capture and analysis.

This is a priority capability.

---

## Phase 4 — Attention instrumentation

Generalise attention observability:

```text
Q
K
V
scores
probabilities
context
O
```

while retaining dedicated KV cache support.

---

## Phase 5 — MLP/neuron instrumentation

Implement:

```text
MLP stages
neuron activations
neuron statistics
neuron selection
```

---

## Phase 6 — Generic intervention framework

Migrate KV modifiers into the generic intervention abstraction.

Then add:

```text
head interventions
neuron interventions
residual interventions
attention interventions
```

---

## Phase 7 — Causal experiment engine

Implement:

```text
baseline
intervention
comparison
propagation analysis
causal report
```

---

## Phase 8 — HOPE integration

Connect HOPE operations to the common intervention and measurement infrastructure.

---

## Phase 9 — Architecture portability

Test the architecture across multiple Transformer families and remove assumptions that only hold for Gemma.

---

# 27. API quality requirements

The external API should be simple enough that a researcher can write something conceptually like:

```python
with stethoscope.experiment("head_ablation"):

    stethoscope.observe(
        model,
        components=["residual", "attention", "mlp"],
        layers=[10, 17, 24],
        capture="summary",
    )

    stethoscope.intervene(
        component="layer.17.attention.head.3",
        operation="zero",
    )

    output = model(...)
```

and subsequently:

```python
report = stethoscope.compare(
    baseline=baseline,
    intervention=experiment,
)
```

The complexity of hooks, framework-specific details, storage and tensor handling should remain below this API.

---

# 28. Testing requirements

Every new component needs:

### Functional tests

* component discovery
* observer registration
* observer removal
* capture correctness
* intervention correctness

### Numerical tests

Where applicable:

* tensor shapes
* expected values
* no-intervention equivalence
* intervention effect

### Causal tests

Construct deliberately simple models where the effect of an intervention is known.

Example:

```text
zeroing neuron X must change output Y
```

### Regression tests

Existing:

* KV cache tests
* Gemma tests
* HOPE tests
* experiment persistence tests

must continue passing.

---

# 29. Performance requirements

Measure the overhead introduced by the Stethoscope.

Benchmark:

```text
baseline inference
observer disabled
summary observation
selected observation
full observation
intervention
```

Report:

* latency
* throughput
* memory
* storage generated

Do not optimise prematurely, but do not allow instrumentation to silently become unusable.

---

# 30. Failure handling

The framework must clearly distinguish:

```text
unsupported architecture
unsupported component
unavailable internal tensor
incompatible model implementation
intervention not safely applicable
capture failed
partial observation
```

Never silently fabricate data.

When an internal value cannot be observed because of an implementation detail, report:

```text
UNAVAILABLE
```

rather than guessing.

---

# 31. Documentation requirements

Create documentation that explains the conceptual architecture.

It should answer:

### What is HOPE Stethoscope?

### What can it currently inspect?

### What can it modify?

### What has been experimentally validated?

### What remains experimental?

### How does it work with Hugging Face?

### How does it discover architecture?

### How does `CacheTap` work?

### How are interventions represented?

### How are experiments recorded?

### How do I instrument a real model?

### How do I run a causal experiment?

### What are the limitations?

Do not overstate scientific conclusions.

---

# 32. Important design philosophy

The system is intended to support **investigation**, not merely monitoring.

The hierarchy should therefore be:

```text
Observe
   ↓
Measure
   ↓
Hypothesise
   ↓
Intervene
   ↓
Measure consequence
   ↓
Compare
   ↓
Refine hypothesis
```

This is much more important than building a collection of dashboards.

---

# 33. Definition of success

The project should eventually make workflows like these straightforward.

### Example 1 — Attention head

```text
Find unusual head
    ↓
inspect Q/K/V
    ↓
inspect attention pattern
    ↓
zero head
    ↓
measure output change
    ↓
trace downstream propagation
```

### Example 2 — Neuron

```text
Find selective neuron
    ↓
inspect activating tokens
    ↓
inspect activation magnitude
    ↓
zero neuron
    ↓
measure residual/logit change
    ↓
determine functional importance
```

### Example 3 — Residual stream

```text
Select token
    ↓
trace representation through all layers
    ↓
identify largest representation changes
    ↓
inspect responsible attention/MLP components
    ↓
intervene
    ↓
verify causal effect
```

### Example 4 — KV cache

```text
select layer/head/token
    ↓
inspect KV state
    ↓
modify KV
    ↓
run generation
    ↓
compare baseline
    ↓
trace effect
```

### Example 5 — HOPE

```text
identify candidate neurons
    ↓
estimate capacity/redundancy
    ↓
prune/merge
    ↓
measure behavioural distortion
    ↓
compare against baseline
```

The user should experience these as variations of the **same experimental framework**, not five different tools.

---

# 34. What NOT to do

Do not:

* rewrite the repository from scratch
* throw away existing KV work
* hardcode Gemma-specific assumptions into the core
* build everything around `named_modules()`
* capture every PyTorch operation
* store every tensor by default
* build a dashboard before the instrumentation architecture is sound
* introduce arbitrary abstractions without corresponding tests
* claim interpretability from correlation alone
* claim causality without intervention
* optimise for synthetic models at the expense of real models
* replace proven mechanisms with generic abstractions merely for aesthetic consistency

---

# 35. Deliverables

At the end of the architectural evolution, produce:

## A. Architecture

A documented system architecture showing:

```text
Model Adapter
     ↓
Model Manifest
     ↓
Component Registry
     ↓
Observation Layer
     ↓
Intervention Layer
     ↓
Experiment Engine
     ↓
Measurement / Analysis
     ↓
Persistence / Knowledge Graph
     ↓
Visualization / Reports
```

## B. Core implementation

A reusable component instrumentation framework.

## C. KV integration

Existing KV functionality migrated into the framework without regression.

## D. Transformer instrumentation

At least:

```text
Embedding
Attention
Q/K/V
KV Cache
MLP
Neuron
Norm
Residual
LM Head
Logits
```

where technically possible for the target models.

## E. Intervention framework

Support at least:

```text
KV
attention head
neuron
residual
```

interventions.

## F. Causal experiments

A reproducible baseline-vs-intervention experiment mechanism.

## G. Real-model validation

Demonstrate the framework against real Hugging Face Transformer models, including the existing Gemma work.

## H. Documentation

Clearly document capabilities, limitations and validation status.

---

# 36. First task before coding

Do not immediately start implementing.

First inspect the entire repository and produce an **architecture delta report** with:

```text
1. Current architecture
2. Existing components
3. Existing KV architecture
4. Existing HOPE architecture
5. Existing experiment infrastructure
6. Existing tests
7. What should be preserved
8. What should be refactored
9. What is missing
10. Proposed target architecture
11. Migration plan
12. Risks
13. Minimal first implementation milestone
```

Then implement the first milestone incrementally.

Every architectural change must preserve existing functionality unless there is a documented technical reason to change it.

The objective is not to make the repository larger.

The objective is to make it possible to **open up a Transformer, listen to its internal computation, selectively modify components, observe the consequences, and accumulate reproducible evidence about what happened.**

That is the target architecture for HOPE Stethoscope.
