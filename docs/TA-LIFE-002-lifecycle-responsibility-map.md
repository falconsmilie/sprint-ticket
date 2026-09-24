# TA-LIFE-002 lifecycle responsibility map

This map records the owner of every responsibility that existed in
`ticket_automation.workflow` before the stage split.

| Responsibility | Current location before the split | Intended owner |
| --- | --- | --- |
| Run setup and trusted policy reload | `run_ticket_lifecycle`, `_run_ticket_lifecycle_locked` | lifecycle controller |
| Resume setup and trusted state reload | `resume_ticket_lifecycle`, `_resume_ticket_lifecycle_locked` | lifecycle controller plus `application.lifecycle.resume` policy |
| Repository lock acquisition and updates | run/resume wrappers, `_update_repository_lock` | lifecycle controller using `locking.RepositoryRunLock` |
| Initial preflight and snapshot creation | `create_run_snapshot` call | existing run snapshot/preflight service, coordinated by controller |
| Attempt creation and completion | preparation/implementation/verification/review/correction stage services and inline reporting branch | `LifecycleController` through the public `attempts` services; handlers receive an immutable identity view of the exact controller-created attempt |
| Stage dispatch | `_drive_lifecycle` state `if` chain | explicit immutable state-to-`StageHandler` mapping owned by `application.lifecycle.handlers` |
| Stage invocation | `_complete_preparation` and `_drive_lifecycle` | focused preparation, implementation, verification, review, correction, and reporting handlers |
| Result interpretation | `_drive_lifecycle` and stage-specific stop-category helpers | focused stage handlers returning `StageDecision` |
| Transition choice | `_record_after_stage_outcome` | focused stage handlers; controller remains the only transition validator/committer |
| Stop classification | `_record_after_stage_outcome`, `_mark_human_required`, stage-specific helpers | focused handlers for expected stops; controller failure classifier for unexpected exceptions |
| Run-record persistence | `_persist_requested_transition`, `_persist_stage_outcome`, terminal helpers | lifecycle controller |
| Final handoff | inline `REPORTING` branch | reporting/handoff handler using `HandoffAcceptanceService` |
| Attempt result JSON writing/loading | `_write_attempt_result`, `_attempt_result` | `persistence_codecs`; the controller coordinates the reporting message write after transition persistence and application/presentation consumers receive decoded values |
| Resume artifact and workspace checks | `_resume_preflight_problem`, `_resume_problem`, `_require_current_workspace_matches_fingerprint` | `application.lifecycle.resume` |
| Report-context creation | `_safe_report_context` and `collect_report_context` calls | `presentation.reporting` |
| Console formatting | `format_lifecycle_result` and helpers | `presentation.lifecycle` |
| Terminal report rendering | direct `generate_terminal_report_best_effort` / `run_report_stage` calls | `presentation.reporting`, invoked exactly once by the controller from persisted terminal state; the package entry boundary supplies the default adapter for compatible public calls |

## Future stage checklist

- [ ] Add the typed workflow state/phase and legal domain transitions.
- [ ] Add one focused handler that returns a `StageDecision` without persisting
      the run record.
- [ ] Add the state explicitly to `build_active_stage_handlers` construction.
- [ ] Inject only the ports and immutable context required by the handler.
- [ ] Add handler tests plus controller dispatch and illegal-transition tests.
- [ ] Construct decisions only through the lifecycle factories and derive attempt
      completion fields from the typed stage result.
- [ ] Confirm the handler package imports neither transition persistence nor
      concrete provider adapters.
- [ ] Extend characterization coverage for the complete transition sequence and
      terminal outcome.
- [ ] Confirm interrupted writable work remains non-resumable unless trusted
      evidence proves it safe.
