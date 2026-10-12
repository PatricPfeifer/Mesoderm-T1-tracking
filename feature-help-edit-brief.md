# Edit brief: feature help cards (round 2)

Edits to the existing cards in `plugins/napari_patric/napari_patric/feature_help/` (editable sources in `svg/`, shared spec in `README.md` there, original brief in `feature-help-design-brief.md`). Keep the shared style (same cell polygons, accent #ff563c, dark theme, 320 x 240 px, Archivo, 3 px halo on text). Only the changes below.

## 0. Layout change for all cards (point 2)

**Formulas move onto the cards and out of the checkbox labels** (the code change for the labels is done; the checkboxes now just say "Circularity", "Shape index", ...).

- The **top-right slot** now always shows the **formula** (previously some cards showed units there):

| Card | Top-right formula |
|---|---|
| area | `A` |
| perimeter | `P` |
| circularity | `4πA / P²` |
| shape_index | `P / √A` |
| eccentricity | `√(1 − b²/a²)` |
| solidity | `A / A(hull)` |
| major_axis | `a` |
| minor_axis | `b` |
| orientation | `θ` |
| centroid | `(x, y)` |
| touches_edge | `1 or 0` |

- **Units** that used to sit top-right (area: px² or µm², perimeter / major / minor: px or µm, centroid: px or µm) move to the **end of the definition line**, e.g. "Size of the cell: pixels inside the outline (px² or µm²)."
- Cards that get a "How it is calculated" block (circularity, shape index, eccentricity; see below) may use **up to 4 lines of bottom text** at 12 px. Shrink the schematic height to make room, but keep the title/rule position (y 28 / 38) identical across all cards.

## 1. `perimeter`: show the whole outline, not one edge

Problem: the card draws a bracket along **one edge** labelled "P", so it reads as "P = the length of one side".

Fix:
- Draw the **entire closed outline** of the cell in accent (4 px, as now).
- Remove the single-edge bracket.
- Show that P is the **sum of all sides**: mark the six corners with small dots and add a small arrow going **once around the cell** (a circular arrow just outside the outline), labelled "P". Alternative if cleaner: split the outline into its six edges with thin alternating-tone segments and write `P = e₁ + e₂ + … + e₆` below the cell.
- Definition line: "Length of the whole cell boundary (px or µm)."
- Optional second line: "Estimated from the pixel mask, so slightly rough."

## 2. `eccentricity`: say how it is calculated

Today the card only says "0 = round, 1 = line", which is not intuitive.

How the plugin computes it (scikit-image `regionprops`): fit the **ellipse that has the same second moments (spread of the pixels) as the cell**, take its long and short half-axes **a** and **b**, then eccentricity = √(1 − b²/a²). It is purely about the **aspect ratio b/a**:

| b/a (short / long) | eccentricity |
|---|---|
| 1.0 (circle) | 0 |
| 0.8 | 0.60 |
| 0.5 | 0.87 |
| 0.24 | 0.97 |

Edits:
- Keep the cell + best-fit ellipse with both axes labelled **a** (long) and **b** (short) in accent.
- **Redraw the row of three small ellipses so their shapes match the numbers**. The current ones (0, 0.8, 0.97) look too similar. Use b/a ratios 1.0, 0.6, 0.24 for e = 0, 0.8, 0.97, and print the ratio under each value (e.g. "0 · b/a = 1", "0.8 · b/a = 0.6", "0.97 · b/a = 0.24").
- Bottom text (max 3 lines):
  - "How stretched the cell is: 0 = round, 1 = a line."
  - "Fit an ellipse to the cell; compare its short axis b to its long axis a."
- Note for the designer: eccentricity is **not** the same as elongation ratio. It grows fast at first (b/a = 0.8 already gives 0.6), so most real cells fall between 0.3 and 0.9.

## 3. `circularity`: say how it is calculated

How: **circularity = 4πA / P²**, with A = pixel area and P = estimated perimeter. Intuition to show: for a given area, **a circle has the shortest possible perimeter** (P_circle = 2√(πA)). So circularity = (P_circle / P)², i.e. how much longer the real outline is than the best possible one, squared. Reference values: circle 1.0, regular hexagon 0.91, square 0.79, elongated cell about 0.6. It is capped at 1.0 (pixel rounding can exceed it slightly).

Edits:
- Keep the round (1.0) and elongated (0.6) cells with the dashed equal-area circle. Add the **regular hexagon 0.91** as a third, small reference cell if space allows, since real cells are hexagon-like and this tells the user "0.9 is already near-perfect for a tissue cell".
- Bottom text (max 3 lines):
  - "How close to a circle: 1 = round, lower = elongated or irregular."
  - "Compares the outline P to the shortest possible outline for the same area (a circle)."

## 4. `shape_index`: formula and how it is calculated

How: **shape index = P / √A**. Perimeter divided by the square root of the area, so it has no units and does not depend on cell size (doubling a cell's size doubles P and multiplies √A by 2, so the ratio stays the same). It is a closely related measure to circularity: **shape index = 3.54 / √circularity**. Reference values: circle 3.54, regular hexagon 3.72, square 4.0; about **3.81** is the threshold used in cell-jamming studies between solid-like (below) and fluid-like (above) tissue.

Edits:
- Keep the 3.5 to 4.5 scale with the three cells and the accent tick at 3.81. Add small reference marks on the scale for **circle 3.54** and **hexagon 3.72**.
- Show a **tiny size-independence hint**: the same hexagon at two sizes, both labelled 3.72. Only if space allows; otherwise write it in the text.
- Bottom text (max 3 lines):
  - "Perimeter / √area: higher = more elongated or irregular."
  - "Has no units, so it does not depend on cell size."

## Checklist

1. Formula top-right on **all 11 cards**; units moved to the definition line.
2. Perimeter shows the full outline.
3. Eccentricity, circularity and shape index carry the short "how calculated" lines above.
4. Layout and style identical across the set (title/rule position, accent colour, fonts).
5. Re-export PNGs at 320 x 240, update the `svg/` sources and the card descriptions in `README.md`.
