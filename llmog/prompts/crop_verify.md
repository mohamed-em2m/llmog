---
description: Crop verification — binary YES/NO presence check for a single object label within a cropped image region
---
Examine this image crop carefully.

**Your task:** determine whether a **"{{ label }}"** is clearly and genuinely present inside this crop.

## Decision rules
- Answer **YES** only if the object is unambiguously present and recognisable as a "{{ label }}".
- Answer **NO** if the crop shows only background, an ambiguous texture, a different object entirely, or if you are uncertain.
- Do **not** guess. When in doubt, answer **NO**.

## Output format
Respond with **exactly one** of the following — no other text:

<present>YES</present>

or

<present>NO</present>
