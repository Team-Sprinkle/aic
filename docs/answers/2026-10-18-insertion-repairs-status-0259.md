# Status — October 18, 02:59 UTC

- **Training is finished:** all current fits reached validation plateaus after learning-rate reductions, rather than being cut short. The new SC grasp-pose models still failed their accuracy gates; none was promoted.
- **Learned BC comparison: 0/8 successes.** Including public controller-error feedback did not solve insertion. Runs stop before the opening with lateral drift; SC also retains orientation error.
- **Explicit visual controller: 6/10 combined successes**, with 5/5 for SFP port 0 and 1/5 for port 1. The toolkit/terminal-geometry disagreement on the four port 1cases remains unresolved. This controller is not learned BC.
- **Backoff comparison is running: 2/12 scored, 0 successes so far.** The completed no-backoff and local-backoff cases both stopped short; the enabled local mode did not actually trigger a retreat. Thus there is no evidence yet that backoff helps this failure.
- **No new complete sample score:** the last full sample remains 165.42/300.

The current bottlenecks are learned insertion command selection and SC grasp-orientation accuracy. Backoff cannot help if the policy stops short without triggering the blockage detector. Let the frozen 12-run comparison finish; then diagnose its traces alongside the visual controller's port 1scoring discrepancy. No active jobs were interrupted and final scenes remain sealed.
