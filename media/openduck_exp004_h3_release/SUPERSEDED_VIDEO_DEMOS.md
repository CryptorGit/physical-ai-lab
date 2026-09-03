# Superseded video demonstrations

Only `openduckmini_h3_motion_verified_showcase_v2_final.mp4` is the current
motion demonstration.

- `openduckmini_h3_release_showcase.mp4`
  (`d7395e1a37aa248f9b8a6389d01b2eb0d1b9ecf208cb706f81496cc84b369bbe`)
  is invalid as motion evidence. Its three-field schedule omitted the frozen
  physical-to-policy observation overrides, its tracking camera hid world
  translation, and its audit checked safety but not motion progress.
- `openduckmini_h3_motion_verified_showcase_v2.mp4`
  (`c97dd1f3656cd6f8118f4b0ebe105606056c7beb5cc4038a5279b0746ef68837`)
  passed the corrected physics checks but is a pre-QA render with long title
  overlap. It is retained only as build history.
- `openduckmini_h3_motion_verified_showcase_v2_final.mp4`
  (`1b2c2eb046dfbea4b3519acf40796432864c7fc6ce7856c627d8feb97b615fbc`)
  uses the exact formal policy-observation mappings, independent exact-home
  resets, a fixed world camera, a top-view trace, and central motion acceptance
  for all twelve moving cases. It remains simulation-only; hardware deployment is
  prohibited.
