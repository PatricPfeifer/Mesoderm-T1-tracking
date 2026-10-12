# Design brief: feature help images for "Export Cell Props"

## What this is for

The napari plugin has an **Export Cell Props** tab. Each cell feature (Area, Perimeter, ...) has a checkbox. When the user hovers over a checkbox, a small tooltip shows **one image** that explains what the feature measures. We need one image per feature.

The audience is biologists looking at epithelial cell shapes in a time-lapse movie (cells are roughly polygons, 4-8 sides, tightly packed). They know what a cell is, but not necessarily what "solidity" or "eccentricity" means. The image should make the meaning clear in about two seconds.

## Technical specs

| | |
|---|---|
| Format | PNG with transparent background (JPG also works, but PNG is preferred) |
| Size | 320 x 240 px, landscape (max 400 x 300). Tooltips can't scroll or resize, so keep it small |
| Theme | napari is **dark** (background about #262930, text about #d6d0ce). Design for dark, and make sure it stays readable on light too if possible |
| Text | Feature name as a title, **one short line** of definition, optionally the formula. Min font about 12 px. No paragraphs |
| Files | One file per feature, named exactly with the key below, e.g. `area.png` |
| Where | `plugins/napari_patric/napari_patric/feature_help/` (they load automatically on the next napari start) |

Keys and display names (use exactly these file names):

`area`, `perimeter`, `circularity`, `shape_index`, `eccentricity`, `solidity`, `major_axis`, `minor_axis`, `orientation`, `centroid`, `touches_edge`

## Visual language (same for all 11 images)

- **One cell** as the main object: a slightly irregular hexagon/pentagon (like a real epithelial cell), flat fill in a muted teal or blue-grey, thin lighter outline. Same cell colour across all images so they read as a set.
- **One accent colour** (e.g. warm orange or yellow) for *the thing being measured*: the outline, the axis, the point, the filled area. Everything else stays neutral.
- Lines and labels only. No gradients, shadows, 3D or decoration.
- Where the idea is a comparison ("low vs high"), show **two small cells side by side** with a short label under each (e.g. "round: 1.0" / "elongated: 0.6"), not a long explanation.
- Use symbols from the formula in the image (A, P, a, b) so users can match them to the docs.
- Consistent layout: title top-left, schematic centre, one-line definition at the bottom.

## What each image should show

For each feature: what it measures, the minimal schematic, and the one comparison worth showing.

### 1. `area`: Area
- **Measures:** the size of the cell, the number of pixels it covers (shown as px^2, or um^2 if pixel size is set).
- **Schematic:** one cell with the **whole interior filled** in the accent colour, labelled "A".
- **Comparison:** none needed. Optionally a small cell vs a big cell.
- **Line:** "Size of the cell: pixels inside the outline."

### 2. `perimeter`: Perimeter
- **Measures:** the length of the cell outline.
- **Schematic:** one cell with only the **outline** in the accent colour (thick), interior neutral, labelled "P". A small bracket or arrow along one edge suggests "length along the boundary".
- **Line:** "Length of the cell boundary."
- **Note:** estimated from the pixel mask, so it is slightly rough (not a polygon fit).

### 3. `circularity`: Circularity
- **Formula:** 4 * pi * A / P^2. Range 0 to 1; **1 = perfect circle**, lower = more irregular or elongated.
- **Schematic:** two cells side by side: a **round** cell (value 1.0) and an **elongated or spiky** one (value about 0.6). Outline of each in the accent colour, with a circle of equal area drawn dashed around/inside the round one to make the idea of "how close to a circle" obvious.
- **Line:** "How close to a circle: 1 = round, lower = elongated or irregular."

### 4. `shape_index`: Shape index
- **Formula:** P / sqrt(A). The perimeter divided by the square root of the area, so it doesn't depend on cell size.
- **Reference values:** circle = 3.54, regular hexagon = 3.72, **larger = more elongated or irregular**. In the cell-jamming literature about 3.81 is a threshold between solid-like (lower) and fluid-like (higher) tissue, so a small marker or a tick on a scale for "about 3.8" is worth showing if there is room.
- **Schematic:** a horizontal scale/bar from about 3.5 to 4.5 with three tiny cells placed on it: a round/hexagonal one on the left, a more elongated one in the middle, a strongly elongated one on the right.
- **Line:** "Perimeter / sqrt(area): higher = more elongated or irregular."
- **Note:** the same cell shape gives the same value at any size.

### 5. `eccentricity`: Eccentricity
- **Measures:** how stretched the cell is. Based on the ellipse that has the same second moments as the cell. Range 0 to 1; **0 = circle, close to 1 = line**.
- **Schematic:** one cell with its **best-fit ellipse** drawn over it in the accent colour, both axes visible. Beside it a small row of three ellipses, from nearly round (0) through stretched (0.8) to very flat (0.97).
- **Line:** "0 = round, 1 = line: how stretched the cell is."

### 6. `solidity`: Solidity
- **Formula:** cell area / convex hull area. Range 0 to 1; **1 = fully convex**, lower = has dents or concave parts.
- **Schematic:** a cell with a **dent** (concave) and its **convex hull** drawn as a dashed rubber-band outline. The **gap between the cell and the hull** is filled in the accent colour. Next to it a fully convex cell with no gap (1.0).
- **Line:** "Cell area / area of its convex hull: 1 = no dents."

### 7. `major_axis`: Major axis length
- **Measures:** the length of the long axis of the best-fit ellipse (not the longest chord).
- **Schematic:** one cell with its best-fit ellipse (faint) and the **long axis** as a thick accent-coloured line with end caps, labelled "a".
- **Line:** "Length of the long axis of the best-fit ellipse."

### 8. `minor_axis`: Minor axis length
- **Measures:** the length of the short axis of the best-fit ellipse (perpendicular to the major axis).
- **Schematic:** the same cell as `major_axis`, but now the **short axis** is the accent-coloured thick line labelled "b", and the long axis is shown thin and grey for reference.
- **Line:** "Length of the short axis of the best-fit ellipse."
- **Tip:** use the same cell and ellipse as image 7 so the two read as a pair.

### 9. `orientation`: Orientation
- **Measures:** the angle of the cell's long axis, in radians, range -pi/2 to +pi/2. (scikit-image convention: angle between the vertical (row) axis of the image and the major axis, counter-clockwise.)
- **Schematic:** one cell with its long axis, a **reference line** (vertical) and an **arc with an angle label** between them, in the accent colour. Optionally a small dial showing 0, +pi/2 and -pi/2.
- **Line:** "Direction of the long axis, in radians (-pi/2 to +pi/2)."
- **Note for the designer:** an elongated cell at +pi/2 and one at -pi/2 point in the same direction. The values wrap, which is why it is coloured with a cyclic colour map.

### 10. `centroid`: Centroid (x, y)
- **Measures:** the centre of mass of the cell, as x and y position in the image.
- **Schematic:** one cell with a **dot** at its centre in the accent colour, and thin dashed lines to the image's x and y axes (small axes in a corner, origin at the **top-left**, y pointing **down**, as in image coordinates).
- **Line:** "Centre of the cell: x, y position in the image."

### 11. `touches_edge`: Touches image edge
- **Measures:** 1 if the cell touches the border of the image (so it is cut off), else 0.
- **Schematic:** a small rectangle (the image frame) with a few cells inside: **two cells cut by the frame border** drawn in the accent colour with a clear "cut" at the border, the interior cells neutral. Labels "1" on the cut ones and "0" on the interior ones.
- **Line:** "1 = cell is cut off by the image border: its shape values are unreliable."

## Deliverable and review

1. Eleven PNGs named as above, plus a short note on fonts and colours used so we can match them later.
2. Check each image at its real size (about 320 x 240 px) on the dark napari background. If the text is unreadable at that size, simplify the image, not the font.
3. Before finalising, check that the set looks like **one family**: same cell, same accent colour, same layout.
