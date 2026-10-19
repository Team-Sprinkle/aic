# Names for the two handoff directions

I suggest these names:

| Direction | Short name | Meaning |
|---|---|---|
| Previous | **Transport-first handoff** | A task-coded transport actor moves toward the port using its image/state history. Near the expected arrival, coarse perception checks the requested port, a high-resolution refiner estimates its pose, and the controller decides whether to hand off to insertion. Precise target identity enters control late. |
| New | **Board-anchored target-aware transport** | Perception first establishes a board origin and numbered slot map when observable, tracks it through motion, and selects the requested slot. Transport uses that explicit observation-derived target cue. Near the port, the refiner and safety check still authorize insertion. |

For plots and checkpoint names, use **transport-first** and **board-anchored**. “Board-anchored” describes the intended source of card identity; it does not imply the new model has passed its identity gate. The current board-slot attempts failed, so this remains a research direction, not a deployed pipeline.

The key distinction is **when the actor learns which physical card is requested**. Both directions can retain the same near-port refiner, insertion actor and recovery logic. The new direction can include a generic safe inspection move before target-specific transport if the board origin is not visible at startup.
