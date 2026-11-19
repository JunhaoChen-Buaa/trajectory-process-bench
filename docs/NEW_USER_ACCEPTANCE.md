# Prior new-user acceptance and publication-copy checks

Before this publication copy was prepared, the delivery was extracted from ZIP into two isolated Python 3.13 environments. A stateful custom agent was connected through `--agent-factory`, and real task execution proceeded through automatic process scoring and report generation. Setup and compatibility issues found during those runs were fixed in the delivered software.

The custom agent maintained its own conversation state and generated its own next response. It reused the configured model transport, not the public LLMAgent workflow. The user and scorer were cloud services for these software acceptance runs; those settings differ from the paper's original 540 trajectories.

These checks demonstrate that the software path was exercised. They are not new benchmark-performance measurements, and they do not establish that all automatic judgments are correct.

This public copy retains one smaller real replay at `examples/replay/retail100`. Redundant installation outputs, local environment paths, and machine-specific acceptance metadata are excluded. The reference collection, runtime scorer/rubric, custom-agent example, and source-review behavior are retained.

For the exact checks rerun on the publication copy, see [VERIFICATION.md](VERIFICATION.md). No new paid model call is needed to verify packaging, reference aggregation, or saved-report reproduction.
