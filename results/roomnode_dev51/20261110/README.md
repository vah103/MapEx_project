# RoomNode DEV51 — image gallery S0 versus S1a

This is a visualization-only replay of the **51 locked DEV snapshots**,
comparing original frozen RoomNode L1 v2 (S0) with RoomNode R1 (S1a) on
identical observed and LaMa-mean inputs.

- Source: vah103/newlab branch chatgpt/roomnode-r1-p1-dev51-20261110,
  evaluation commit 0e2c44c.
- Independent review: claude/roomnode-r1-p1-dev51-review, commit 5b08a4e.
- Locked L1 SHA256:
  7aba149a4391b1045efdbf4ad2c8f4d81e0c4fddc256a093bf320a47ae794a19.
- DEV inputs: /home/dell/DELL_NATIVE2D/build/topo002/eval_dev via
  eval_dev.load_frames(). Reference: /tmp/roomnode_p1_dev51_20261110/per_frame.jsonl.
- Every pair's input SHA256, output-door count and region count was checked
  against the locked scorer: **102/102 match**.
- Total doors: S0 337, S1a 398.
- Unmatched to any GT doorway cell within 1 m: S0 49, S1a 61.
  Nonmatching annotation is not proof of a physically fake door.
- GT was used only to annotate nonmatches, never to run S0 or S1a.
- Six DEV maps, 51 snapshots, 102 WebP images. No KTH or closed-loop runs.

## Reading the images (v2: semantic colors and full skeleton)

- **Peach:** RoomNode classifies this region as a **room** (P#).
- **Teal:** RoomNode classifies this region as a **corridor** (HL#).
- **Yellow:** RoomNode classifies this region as a **hall** (S#).
- **Violet line with white halo:** all `skeleton_rc` pixels from the frozen
  RoomNode algorithm, *not* the sparse `backbone_mask` seen in gallery v1.
- The labels/classes are **predictions**, not ground-truth labels; all
  regions of the same class have the same color, separated by dark boundaries.
- Slightly stronger tint: UNKNOWN regions predicted free by LaMa.
- Dark navy: observed obstacles.
- Gold dots: predicted door centers in observed free space.
- Purple dots: predicted door centers in UNKNOWN.
- Red X: predicted doors with no GT door within 1 m.
- Yellow dot: robot; thin dark lines: predicted region boundaries.

The two panels of each decision use precisely the same map crop.
The included build_gallery.py regenerates the visual-only replay on Dell,
without modifying source runs or algorithm code.
