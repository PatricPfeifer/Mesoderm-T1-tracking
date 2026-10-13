# Feature help images: Export Cell Props

Tooltip images for the Export Cell Props tab. PNGs are final; `svg/` holds editable sources.

## Shared spec

- Size: 320 × 240 px, transparent PNG, designed for napari dark (#262930).
- Font: Archivo (Google Fonts). Title 16px / 800, formula 13px / 400, labels 12–12.5px / 400. The SVGs reference Archivo by name and do not embed it; install it before rasterising.
- Layout: title top-left (x 14, y 28), formula/unit top-right (x 306, right-aligned), 2px rule at y 38, schematic centre, definition at the bottom (y 210 / 226).
- Colours: accent #ff563c (the measured thing); cell fill #7d7979 @ 50%, outline #bab6b6; text #eae7e7 (title), #d7d3d3 (labels), #bab6b6 (secondary); rule #605d5d; text on accent #201e1d.
- All text has a 3px #262930 halo (paint-order: stroke), so it stays readable on light themes.
- Base cell polygon: `-50,-30 -8,-56 44,-38 56,12 14,52 -42,36`. Elongated cell: `-58,-10 -26,-34 24,-32 60,-6 40,28 -30,30` with best-fit ellipse rx 54 / ry 30.

## Cards

### `area` · Area
- File: `area.png` (source `svg/area.svg`)
- Formula / unit: px² or µm²
- Definition: Size of the cell: pixels inside the outline.
- Drawn: One cell, whole interior filled accent, labelled "A".

### `perimeter` · Perimeter
- File: `perimeter.png` (source `svg/perimeter.svg`)
- Formula / unit: px or µm
- Definition: Length of the cell boundary.
- Drawn: One cell, outline in accent (4px), interior neutral. Bracket along one edge labelled "P". Value is estimated from the pixel mask.

### `circularity` · Circularity
- File: `circularity.png` (source `svg/circularity.svg`)
- Formula / unit: 4πA / P²
- Definition: How close to a circle: 1 = round, lower = elongated or irregular.
- Drawn: Two cells: round (1.0) with dashed equal-area circle, elongated (0.6). Outlines in accent.

### `shape_index` · Shape index
- File: `shape_index.png` (source `svg/shape_index.svg`)
- Formula / unit: P / √A
- Definition: Perimeter / sqrt(area): higher = more elongated or irregular.
- Drawn: Scale 3.5–4.5 with three cells (hexagon ≈3.72, elongated ≈4.0, strongly elongated ≈4.4). Accent tick at 3.81, "solid-like / fluid-like" either side. Size-independent.

### `eccentricity` · Eccentricity
- File: `eccentricity.png` (source `svg/eccentricity.svg`)
- Formula / unit: 0 to 1
- Definition: 0 = round, 1 = line: how stretched the cell is.
- Drawn: Cell with best-fit ellipse and both axes in accent; row of three ellipses labelled 0, 0.8, 0.97.

### `solidity` · Solidity
- File: `solidity.png` (source `svg/solidity.svg`)
- Formula / unit: A / A(hull)
- Definition: Cell area / area of its convex hull: 1 = no dents.
- Drawn: Dented cell (0.84) with dashed convex hull, gap filled accent; convex cell (1.0) next to it.

### `major_axis` · Major axis length
- File: `major_axis.png` (source `svg/major_axis.svg`)
- Formula / unit: a
- Definition: Length of the long axis of the best-fit ellipse.
- Drawn: Elongated cell (rotated −20°), faint dashed ellipse, long axis thick accent with end caps, labelled "a". Pairs with minor_axis.

### `minor_axis` · Minor axis length
- File: `minor_axis.png` (source `svg/minor_axis.svg`)
- Formula / unit: b
- Definition: Length of the short axis of the best-fit ellipse.
- Drawn: Same cell and ellipse as major_axis; short axis thick accent labelled "b", long axis thin grey.

### `orientation` · Orientation
- File: `orientation.png` (source `svg/orientation.svg`)
- Formula / unit: radians
- Definition: Direction of the long axis, in radians (−π/2 to +π/2).
- Drawn: Cell with long axis in accent, dashed vertical reference, arc labelled θ. Dial: 0 at top, ±π/2 at both ends (values wrap; cyclic colour map).

### `centroid` · Centroid (x, y)
- File: `centroid.png` (source `svg/centroid.svg`)
- Formula / unit: image coordinates
- Definition: Centre of the cell: x, y position in the image.
- Drawn: Cell with accent dot at centre; dashed lines to x and y axes. Origin top-left, y points down.

### `touches_edge` · Touches image edge
- File: `touches_edge.png` (source `svg/touches_edge.svg`)
- Formula / unit: 1 or 0
- Definition: 1 = cell is cut off by the image border: its shape values are unreliable.
- Drawn: Image frame with three interior cells (0, neutral) and two cells cut by the border (1, accent fill, dashed ghost outside the frame).
