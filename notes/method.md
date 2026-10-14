# Why three selection strategies, not one

Could just run farthest-point sampling (FPS) in SOAP space on the
whole window and call it done. Don't, for two reasons.

**FPS misses the actual extremes.** "Far apart in SOAP space" isn't
the same as "physically extreme" - a frame with one atom under a huge
force can still look ordinary in an averaged, per-species descriptor
if the rest of the structure is typical. Those frames matter for a
robust potential, so I pull them in first with independent
extreme-value stats (robust z-score on energy/atom, max force, stress
norm, volume), before FPS runs.

**FPS can also cluster in time** if SOAP-diversity happens to peak in
one narrow part of the window. Doesn't matter as much scientifically
(time-adjacent frames are redundant anyway, that's the whole premise)
but a selection visibly spread across the window is a useful sanity
check in `selection.csv` later - so a few slots are reserved for even
time coverage too.

Both tail and temporal picks are seeded into FPS, which then fills the
rest maximizing distance from everything already chosen - not three
independent lists stacked on top of each other.

## Why species-balanced SOAP

A SOAP descriptor averaged over all atoms is weighted by how many
atoms of each species there are. For Mg32 Si32 O96, O would dominate
the descriptor 3:1 just by atom count, even if Mg or Si is what's
actually changing. Averaging per species first, then concatenating
and renormalizing, puts every species on equal footing.

## Why window at all (`--last`/`--stride`)

SOAP on every raw frame of a multi-thousand-step trajectory is slow -
windowing keeps it tractable. More importantly: the early part of a
run is still equilibrating from whatever you started it at, and you
don't want the training set biased toward that transient - which is
why this windows to the *end* of the trajectory, not a uniform sample
of the whole thing.
