# Blind-judged crops: WITH agent vs WITHOUT agent (SAM3)

Panels in each PNG: Real photo | WITHOUT agent (SAM3) | WITH agent. Same held-out camera pose.
3 independent judges saw these crops blind (A/B order shuffled per judge). Votes = judges preferring each side.

| Crop | View | Agent votes | SAM3 votes | Visibility | Likely cause |
|---|---|---|---|---|---|
| C01 | DJI_0002 | 3 | 0 | noticeable, noticeable, noticeable | random (close-up object, poorly constrained in both) |
| C02 | DJI_0067 | 1 | 1 | subtle, subtle, subtle | - |
| C03 | DJI_0443 | 3 | 0 | noticeable, noticeable, noticeable | random (no-sky view: both trained on identical images) |
| C04 | DJI_0139 | 3 | 0 | noticeable, noticeable, noticeable | random (both break down in this corner) |
| C05 | DJI_0571 | 3 | 0 | noticeable, noticeable, noticeable | random colour floater (agent has its own just outside crop) |
| C06 | DJI_0555 | 2 | 0 | subtle, subtle, subtle | unclear, subtle |
| C07 | DJI_0059 | 0 | 2 | subtle, subtle, none | mask edge (subtle) |
| C08 | DJI_0011 | 3 | 0 | noticeable, noticeable, noticeable | random floater column |
| C09 | DJI_0580 | 0 | 3 | noticeable, obvious, noticeable | mostly random, horizon poorly constrained |
| C10 | DJI_0211 | 0 | 3 | obvious, obvious, obvious | random blotches + agent skyline fringe |
| C11 | DJI_0235 | 0 | 3 | noticeable, noticeable, noticeable | random tear + agent skyline fringe |
| C12 | DJI_0075 | 0 | 2 | subtle, subtle, subtle | random (subtle) |
| C13 | DJI_0195 | 0 | 3 | noticeable, noticeable, noticeable | MASK: agent bright skyline fringe |
| C14 | DJI_0067 | 0 | 3 | noticeable, noticeable, noticeable | MASK: agent bright skyline fringe |

Bottom line: the 5 crops where the agent clearly looks better (C01, C03, C04, C05, C08) are random training artifacts, not caused by the masks.
The only mask-caused difference is at the skyline: the agent keeps ~4 px more terrain than SAM3's 15 px dilated mask but leaves a bright 1-2 px edge line (32/34 sky views), which judges preferred to avoid (C13, C14).
