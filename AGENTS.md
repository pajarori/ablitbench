# Development workflow

- Use the existing `.venv`. Do not create environments or image tags for individual checkpoints.
- Run API and static source checks without Docker. Use containers only for source inference, real tools, or a specific research engine that requires isolation.
- Start with focused tests. Run the broader regression gate after the implementation is coherent.
- Keep one current worker image per required engine profile. Build only the profile needed for the current verification.
- Measure available disk and the expected build peak before downloading ML dependencies or model artifacts.
- Keep hashes, logs, and small fixtures as evidence. Do not duplicate research repositories, model files, environments, or wheels for every proof.
- Clean task-owned temporary directories in `finally`, including read-only source staging. Verify that no task processes or containers remain.
- Remove superseded task-owned images and cache records after verification. Never use global Docker pruning or delete unrelated volumes.
- Preserve original research, Git history, dirty changes, and unrelated services.
- Never add code comments, docstrings, or commit co-author trailers.
