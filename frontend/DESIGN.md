---
name: Deep-Sea Tactical
colors:
  surface: '#05034f'
  surface-dim: '#05034f'
  surface-bright: '#2f3174'
  surface-container-lowest: '#020045'
  surface-container-low: '#0f1056'
  surface-container: '#141559'
  surface-container-high: '#1f2164'
  surface-container-highest: '#2a2d6f'
  on-surface: '#e1e0ff'
  on-surface-variant: '#bcc9c7'
  inverse-surface: '#e1e0ff'
  inverse-on-surface: '#26286b'
  outline: '#869391'
  outline-variant: '#3d4947'
  surface-tint: '#69d8cd'
  primary: '#69d8cd'
  on-primary: '#003733'
  primary-container: '#36ada3'
  on-primary-container: '#003b37'
  inverse-primary: '#006a63'
  secondary: '#a5c8ff'
  on-secondary: '#00315e'
  secondary-container: '#204a7c'
  on-secondary-container: '#95baf3'
  tertiary: '#bbc3ff'
  on-tertiary: '#1d2a6d'
  tertiary-container: '#8c98e1'
  on-tertiary-container: '#222e71'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#86f5ea'
  primary-fixed-dim: '#69d8cd'
  on-primary-fixed: '#00201d'
  on-primary-fixed-variant: '#00504b'
  secondary-fixed: '#d4e3ff'
  secondary-fixed-dim: '#a5c8ff'
  on-secondary-fixed: '#001c3a'
  on-secondary-fixed-variant: '#1d487a'
  tertiary-fixed: '#dee0ff'
  tertiary-fixed-dim: '#bbc3ff'
  on-tertiary-fixed: '#031158'
  on-tertiary-fixed-variant: '#354185'
  background: '#05034f'
  on-background: '#e1e0ff'
  surface-variant: '#2a2d6f'
  deep-navy: '#121358'
  royal-blue: '#232F72'
  steel-blue: '#2F578A'
  teal-accent: '#36ADA3'
typography:
  headline-lg:
    fontFamily: Oswald
    fontSize: 64px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Oswald
    fontSize: 40px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Oswald
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: 0.02em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.15em
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0em
spacing:
  unit: 4px
  safe-margin: 24px
  gutter: 12px
  panel-gap: 8px
  container-padding: 16px
---

## Brand & Style

The brand personality is **Industrial Tactical**, specifically refined for high-stakes maritime or deep-space environments where deep shadows and neon clarity are paramount. It targets specialized operators who manage complex systems through a digital-first lens.

The design style is **Brutalism-Glassmorphism**. It combines the raw, unyielding structure of industrial hardware with the sophisticated visual layering of advanced optics. The interface must feel heavy yet precise—like a ruggedized submarine console or a high-altitude cockpit. Visual impact is achieved through the contrast between the "void" of the Deep Navy background and the "glow" of the Teal Accent telemetry, creating a sense of immense depth and focus.

## Colors

The palette is optimized for a strictly dark environment, utilizing a monochromatic blue range to recede into the background while the chromatic teal provides high-visibility feedback.

- **Primary Surface (Deep Navy):** `#121358`. The foundation of the HUD. It acts as the "true dark" to minimize eye strain.
- **Secondary Surface (Royal Blue):** `#232F72`. Used for nested containers and active panel backgrounds to provide subtle tonal separation.
- **Accents/Borders (Steel Blue):** `#2F578A`. The color of the "metal"—used for structural lines, inactive UI states, and HUD brackets.
- **Primary Action (Teal Accent):** `#36ADA3`. The "neon highlight." Reserved for critical data, active selections, and interactive controls to ensure they pierce through the blue atmosphere.

For data readability, all text on surfaces must maintain a contrast ratio of at least 7:1 against the Deep Navy background.

## Typography

This typography strategy centers on **Industrial Authority** and **Functional Precision**.

- **Oswald:** Used in its condensed form for all major headings. It should almost always be set in all-caps to reinforce the military-spec aesthetic.
- **Inter:** The secondary workhorse for descriptions and paragraph text where maximum legibility is required within tight containers.
- **JetBrains Mono:** Dedicated exclusively to technical data, labels, and telemetry. The monospaced nature ensures that numeric values (coordinates, depth, time) don't jump as they update in real-time.

Use `label-caps` for all interactive element labels to distinguish UI controls from static information.

## Layout & Spacing

The layout follows a **Fixed HUD Frame** model, where the viewport is treated as a tactical display.

- **Safe Zones:** A 24px margin is maintained around the edges of the screen. Primary navigation and system health are anchored to these corners.
- **Modular Panels:** Content is grouped into logical modules. Use the 8px `panel-gap` for horizontal and vertical spacing between distinct UI widgets.
- **Information Density:** High density is a feature, not a bug. Use the 4px unit for fine-tuning the positioning of telemetry within panels, keeping internal padding at 16px.
- **Responsive Behavior:** On mobile, the corner-anchored HUD elements reflow into a vertically stacked list of modules. The `headline-lg-mobile` ensures the industrial style persists without dominating the limited vertical space.

## Elevation & Depth

Depth is established through **Spectral Layering** and **Atmospheric Blurs** rather than traditional drop shadows.

- **The Void:** The base layer is `deep-navy`. All other elements sit "above" this depth.
- **HUD Glass:** Panels use `royal-blue` at 60-80% opacity with a high-intensity backdrop blur (24px) to simulate information being projected onto semi-reflective surfaces.
- **Neon Luminosity:** Interactive elements use an inner and outer glow in `teal-accent` to signify life. Active buttons should appear as if they are backlit LEDs.
- **Scanning Borders:** Use 1px `steel-blue` outlines for all containers. For high-priority modules, use a "pulsing" 1px `teal-accent` border to draw the operator's eye.

## Shapes

The shape language is strictly **Sharp (0px)**, reflecting the unyielding nature of industrial hardware.

- **90-Degree Precision:** All corners for buttons, panels, and input fields must be perfectly square.
- **Tactical Chamfers:** For the most prominent actions (e.g., "LAUNCH," "ENGAGE"), apply an 8px 45-degree chamfer on the top-right corner to break the grid and signal importance.
- **Data Brackets:** Use corner-only borders (L-shapes) to frame critical telemetry blocks instead of fully enclosing them, maintaining an open and breathable UI despite the high density.

## Components

- **Action Buttons:** Primary buttons are solid `teal-accent` with black text. Secondary buttons are ghost-style with a 1px `steel-blue` border that turns `teal-accent` on hover.
- **Telemetry Chips:** Small containers with `royal-blue` backgrounds and `data-mono` typography in `teal-accent`.
- **Lists:** Data lists should use alternating row backgrounds of `deep-navy` and `royal-blue` (at 30% opacity) for horizontal scanning.
- **Input Fields:** Styled as "Field Terminals"—use a solid bottom border in `steel-blue`. On focus, the border glows `teal-accent` and the background becomes a slightly brighter `royal-blue`.
- **Status Indicators:** Represented as vertical segmented bar graphs. Each segment is a 4x2px rectangle.
- **Crosshairs:** A persistent 1px `teal-accent` reticle should be used in the center of the viewport for systems requiring targeting or directional focus.