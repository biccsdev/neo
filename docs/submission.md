# Release and submission checklist

This file prepares the public release. It does not assert that external steps have happened.

## Package identity

- Plugin: `neo`
- Version: `0.1.0`
- Repository: `https://github.com/biccsdev/neo.git`
- Licence: MIT
- Minimum Hermes: `0.22.6`
- Initially validated platform: macOS

## Local release gates

- [x] Full automated system suite passes.
- [x] API, presentation and security suite passes with development dependencies.
- [x] Hermes Plugin Doctor passes.
- [x] Hermes package validation and security scan pass.
- [x] Local installation and enablement pass in an isolated Hermes home.
- [x] End-to-end operator acceptance suite passes with controlled executors.
- [ ] Live Codex task and exact-session follow-up.
- [ ] Live Claude task and exact-session follow-up.
- [ ] Live official-Blender scene, preview and artifact.
- [ ] Live Desktop mission-control render and approval interaction.
- [ ] Live microphone/voice turn.
- [ ] Screenshots of mission control, live preview and approval state.
- [ ] Demo video with a real result and a real failure or approval.

A pending live gate must not be described as passing in release material.

## External release gates

- [ ] Push the reviewed commit to the public repository.
- [ ] Verify the exact 40-character public commit loads in a clean Hermes home.
- [ ] Open the Hermes plugin-catalog pull request.
- [ ] Publish the demo post on X tagging `@NousResearch`.
- [ ] Submit the verified pull-request and demo-post URLs before the hackathon deadline.

## Catalog pull-request draft

**Title:** Add NEO — conversational computer operations and durable task control

**Summary:**

NEO adds a conversational mission-control interface to Hermes. It orchestrates parallel coding specialists, official-Blender workers and visible browser navigation with durable SQLite state, exact-session follow-ups, task-scoped previews, approval boundaries, restart recovery and verified automatic result presentation.

The package includes seven conversational tools, a read-only task-context hook, a FastAPI backend, a Desktop mission-control page, bundled operator/worker skills and extensive automated acceptance coverage.

**Validation:**

- Full system suite: 672 tests passing, 12 environment-dependent API skips under system Python.
- API/presentation/security suite: 115 tests passing with development dependencies.
- `hermes plugins doctor . --ci`: seven tools and one hook.
- `hermes plugins validate . --install-deps`: passed, including security scan and Desktop SDK surface checks.
- Clean local install and enablement in an isolated Hermes home: passed.

**Platform and limitations:**

macOS is the initially validated platform. Live Codex, Claude, Blender, Desktop and microphone acceptance require their respective credentials/capabilities and are tracked in `docs/demo.md`. Browser approval never publishes or sends content. Blender fails closed unless the official pinned plugin and installed revision can be verified.

## Demo-post draft

> NEO for Hermes Agent: one conversational mission-control interface for coding specialists, Blender, browser workflows and reusable capabilities. Durable parallel tasks, exact-session follow-ups, task-scoped previews, approval boundaries and verified automatic results. Open source: https://github.com/biccsdev/neo @NousResearch

Attach the actual demo video before posting. Do not post this draft without the video or claim live scenarios that have not been demonstrated.
