"""UI Design Rules Knowledge Base.

Each rule is a concept-level design directive — opinionated, modern, and specific.
Rules are organised into 10 categories. The retriever picks one rule per category
per build, producing a unique design brief for every project.
"""

UI_RULES: list[dict] = [

    # ── COLOR ──────────────────────────────────────────────────────────────────

    {
        "id": "color_warm_minimal",
        "category": "color",
        "name": "Warm Minimal",
        "rule": (
            "Build around a warm off-white base (cream or stone-50) with a single deep accent — "
            "forest green, terracotta, or ink navy. Use the accent sparingly: primary CTA, "
            "active states, and key labels only. Every other element stays in warm neutrals."
        ),
        "tailwind_hints": "bg-stone-50, text-stone-900, accent emerald-700 or amber-700",
        "vibe": ["minimal", "warm", "editorial", "premium", "clean", "subtle"],
    },
    {
        "id": "color_high_contrast_pop",
        "category": "color",
        "name": "High Contrast + One Pop",
        "rule": (
            "Near-black (#0A0A0A) backgrounds and white text for primary surfaces. "
            "Introduce one electric accent — electric violet, neon lime, or hot coral — "
            "exclusively for interactive elements and highlights. Everything else is strict black/white."
        ),
        "tailwind_hints": "bg-zinc-950, text-white, accent violet-500 or lime-400 or rose-500",
        "vibe": ["bold", "dark", "tech", "startup", "modern", "developer"],
    },
    {
        "id": "color_jewel_tones",
        "category": "color",
        "name": "Jewel Tones",
        "rule": (
            "Use rich saturated jewel colors as the palette foundation: deep sapphire, emerald, "
            "amethyst, or ruby. Apply the jewel color to the hero/header surface. Keep content "
            "areas white or near-white. Use the jewel tone for accents and iconography throughout."
        ),
        "tailwind_hints": "bg-violet-900 or bg-emerald-800 for hero, bg-white for content, text-violet-700 for accents",
        "vibe": ["rich", "premium", "sophisticated", "luxury", "brand", "deep"],
    },
    {
        "id": "color_sunset_gradient",
        "category": "color",
        "name": "Gradient Spectrum",
        "rule": (
            "Use a carefully chosen 2-stop gradient as the brand signature — coral→violet, "
            "amber→rose, or sky→indigo. Apply it to hero backgrounds, primary buttons, and "
            "selected states. Use white/light gray for all content surfaces. "
            "Never apply gradients to text or small icons."
        ),
        "tailwind_hints": "bg-gradient-to-br from-orange-400 to-pink-600 on hero + CTAs only",
        "vibe": ["vibrant", "energetic", "consumer", "app", "colorful", "modern", "marketing"],
    },
    {
        "id": "color_monochrome_depth",
        "category": "color",
        "name": "Monochrome Depth",
        "rule": (
            "Choose a single hue (blue, slate, or sage) and use 5+ shades of it as the entire "
            "palette. Darkest shade for text, mid-shades for borders and dividers, lightest for "
            "backgrounds. Add white for content cards. No secondary hue — all depth comes from "
            "tonal variation alone."
        ),
        "tailwind_hints": "bg-slate-50, border-slate-200, text-slate-900, accents slate-600",
        "vibe": ["sophisticated", "corporate", "clean", "b2b", "trustworthy", "professional"],
    },

    # ── TYPOGRAPHY ─────────────────────────────────────────────────────────────

    {
        "id": "type_editorial_bold",
        "category": "typography",
        "name": "Editorial Bold",
        "rule": (
            "Extreme type weight contrast: display headings at text-6xl to text-8xl with "
            "font-black and tracking-tighter. Subheadings at text-2xl font-semibold. "
            "Body at text-base with leading-relaxed. Use one font family — variation comes "
            "from weight and size alone, not multiple typefaces."
        ),
        "tailwind_hints": "text-7xl font-black tracking-tighter for display, text-base leading-relaxed for body, text-xs uppercase tracking-widest for labels",
        "vibe": ["editorial", "bold", "portfolio", "impact", "premium", "expressive", "agency"],
    },
    {
        "id": "type_humanist_refined",
        "category": "typography",
        "name": "Humanist Refined",
        "rule": (
            "Modest type scale: headings at text-2xl to text-3xl font-semibold, body at "
            "text-base or text-sm with leading-loose for readability. Emphasise refinement "
            "over impact. Muted secondary text at opacity-60 for hierarchy without harshness. "
            "Letter-spacing: tight on headings, normal on body."
        ),
        "tailwind_hints": "text-2xl font-semibold tracking-tight, text-sm leading-loose, text-gray-400 for secondary",
        "vibe": ["refined", "readable", "saas", "dashboard", "documentation", "professional", "calm"],
    },
    {
        "id": "type_mixed_weight",
        "category": "typography",
        "name": "Mixed Weight Hierarchy",
        "rule": (
            "Within a single heading, combine weights: font-light for the first word/line and "
            "font-bold for the key word. Create rhythm with alternating text sizes between sections. "
            "Labels in uppercase with wide letter-spacing (tracking-widest). "
            "Body copy at text-lg for comfortable reading."
        ),
        "tailwind_hints": "span font-light + span font-bold within h1, text-xs uppercase tracking-widest for metadata",
        "vibe": ["creative", "agency", "marketing", "expressive", "premium", "mixed"],
    },
    {
        "id": "type_mono_accent",
        "category": "typography",
        "name": "Monospace Accent",
        "rule": (
            "Use a sans-serif for all headings and body copy. Introduce font-mono specifically "
            "for numbers, stats, code snippets, tags, status badges, and technical metadata. "
            "The contrast between proportional and monospace creates a modern technical "
            "personality that signals precision."
        ),
        "tailwind_hints": "font-sans for prose, font-mono for numbers/stats/tags/badges, tabular-nums on all numeric displays",
        "vibe": ["technical", "developer", "data", "dashboard", "saas", "product", "code"],
    },
    {
        "id": "type_large_body",
        "category": "typography",
        "name": "Comfortable Large Body",
        "rule": (
            "Prioritise reading comfort above all: body text at text-lg (18px), generous "
            "line-height at leading-8, paragraphs capped at max-w-prose. Headings are "
            "understated (text-2xl max). The large breathable body text IS the design statement. "
            "Perfect for content-heavy and documentation apps."
        ),
        "tailwind_hints": "text-lg leading-8 max-w-prose for content, headings text-xl font-medium, no display-size text",
        "vibe": ["readable", "blog", "documentation", "content", "accessible", "comfortable", "writing"],
    },

    # ── SPACING ─────────────────────────────────────────────────────────────────

    {
        "id": "spacing_generous",
        "category": "spacing",
        "name": "Generous Breathing Room",
        "rule": (
            "Apply extreme whitespace throughout: section padding at py-24 or py-32, card "
            "padding at p-10 or p-12, gaps between elements at gap-12 or gap-16. Content must "
            "have room to breathe — never crowd elements. The whitespace itself communicates "
            "premium quality and intentional design."
        ),
        "tailwind_hints": "py-24 for sections, p-10 for cards, gap-12 for grids, mt-16 between major blocks",
        "vibe": ["premium", "minimal", "luxury", "editorial", "spacious", "breathing", "upscale"],
    },
    {
        "id": "spacing_rhythmic",
        "category": "spacing",
        "name": "8px Grid Rhythm",
        "rule": (
            "Apply strict 8px grid spacing: all padding, margin, and gap values are multiples "
            "of 8px (8, 16, 24, 32, 48, 64). Use 4px only for micro-adjustments within "
            "components. This invisible grid creates visual harmony that users feel even when "
            "they can't name it."
        ),
        "tailwind_hints": "p-4 (16px), p-6 (24px), p-8 (32px), gap-6, mt-12 — always multiples of Tailwind's 4-unit scale",
        "vibe": ["systematic", "clean", "organised", "professional", "saas", "product", "grid"],
    },
    {
        "id": "spacing_dense",
        "category": "spacing",
        "name": "Information Dense",
        "rule": (
            "Pack information efficiently for power users: compact card padding (p-4), tight "
            "gaps (gap-3), smaller body text (text-sm). Use dividers and subtle background "
            "shifts instead of large whitespace to delineate sections. Every pixel earns its "
            "place — this is the aesthetic of tools built for professionals."
        ),
        "tailwind_hints": "p-3 or p-4 for cards, gap-3 grids, text-sm body, border-b divide-y for row separation",
        "vibe": ["dashboard", "data", "admin", "crm", "productivity", "power-user", "dense"],
    },
    {
        "id": "spacing_asymmetric",
        "category": "spacing",
        "name": "Asymmetric Tension",
        "rule": (
            "Use intentionally asymmetric spacing to create visual tension and interest: "
            "large left padding (pl-16) with smaller right (pr-8), tall top sections with "
            "compact bottoms. Offset elements slightly from the grid. "
            "This reads as designed, not templated."
        ),
        "tailwind_hints": "pl-16 pr-8 for sections, unequal column widths (grid-cols-5 with col-span-3 + col-span-2)",
        "vibe": ["creative", "agency", "portfolio", "editorial", "expressive", "distinctive"],
    },
    {
        "id": "spacing_sectioned",
        "category": "spacing",
        "name": "Clear Section Blocks",
        "rule": (
            "Divide the page into clearly separated full-width sections with alternating "
            "backgrounds (white / gray-50 / white). Each section gets consistent py-20 vertical "
            "padding. Elements within sections use tighter spacing (gap-8). The section blocks "
            "create natural page rhythm and scanability."
        ),
        "tailwind_hints": "alternating bg-white and bg-gray-50 sections, each py-20, inner content gap-8",
        "vibe": ["landing", "marketing", "saas", "product", "structured", "clear", "sections"],
    },

    # ── BORDER RADIUS ──────────────────────────────────────────────────────────

    {
        "id": "radius_pill",
        "category": "radius",
        "name": "Pill Softness",
        "rule": (
            "Round everything aggressively: buttons fully rounded (rounded-full), cards at "
            "rounded-3xl, inputs at rounded-2xl, badges at rounded-full. The consistent soft "
            "roundness signals friendliness and approachability. Works beautifully with "
            "bold saturated colors."
        ),
        "tailwind_hints": "rounded-full for buttons/badges, rounded-3xl for cards, rounded-2xl for inputs/modals",
        "vibe": ["friendly", "consumer", "app", "colorful", "approachable", "mobile", "playful"],
    },
    {
        "id": "radius_subtle",
        "category": "radius",
        "name": "Subtle Polish",
        "rule": (
            "Use understated rounding: buttons at rounded-lg (8px), cards at rounded-xl (12px), "
            "inputs at rounded-md (6px). Small enough to feel purposeful and modern, not "
            "aggressive. This is the professional standard for SaaS and B2B products."
        ),
        "tailwind_hints": "rounded-lg for buttons, rounded-xl for cards/containers, rounded-md for form elements",
        "vibe": ["professional", "saas", "b2b", "clean", "balanced", "polished", "standard"],
    },
    {
        "id": "radius_sharp",
        "category": "radius",
        "name": "Sharp Brutalist",
        "rule": (
            "Use zero or minimal rounding: rounded-none on all cards and containers, "
            "rounded-sm (2px) on buttons and inputs only. Pair with strong borders (border-2) "
            "and bold typography. The sharpness reads as confident and editorial — "
            "not unfinished."
        ),
        "tailwind_hints": "rounded-none for cards/containers, rounded-sm for buttons, border-2 to compensate for no shadow",
        "vibe": ["brutalist", "editorial", "bold", "graphic", "portfolio", "distinctive", "strong"],
    },
    {
        "id": "radius_mixed",
        "category": "radius",
        "name": "Strategic Mixing",
        "rule": (
            "Mix radius intentionally to signal hierarchy: cards at rounded-2xl, primary buttons "
            "rounded-full, secondary buttons rounded-lg, tags and badges rounded-full, "
            "modals at rounded-3xl. Rounder = more important or interactive. "
            "The variation creates implicit hierarchy."
        ),
        "tailwind_hints": "rounded-2xl for cards, rounded-full for primary buttons + badges, rounded-lg for secondary buttons",
        "vibe": ["nuanced", "hierarchy", "modern", "product", "considered", "sophisticated"],
    },

    # ── SHADOW ──────────────────────────────────────────────────────────────────

    {
        "id": "shadow_soft",
        "category": "shadow",
        "name": "Soft Diffuse Elevation",
        "rule": (
            "Use large diffuse shadows with low opacity for elevation: shadow-lg or shadow-xl "
            "on cards (never shadow-md — it reads as cheap). On hover, transition to shadow-2xl. "
            "Combine with a very light page background (gray-50 or stone-50) so the white "
            "cards visually lift off the surface."
        ),
        "tailwind_hints": "shadow-lg on cards, shadow-xl on modals/dropdowns, shadow-2xl on hover, transition-shadow duration-200",
        "vibe": ["premium", "soft", "modern", "clean", "elevated", "professional", "light"],
    },
    {
        "id": "shadow_colored",
        "category": "shadow",
        "name": "Colored Brand Shadows",
        "rule": (
            "Tint shadows with the brand accent color using Tailwind's shadow-color utilities. "
            "A violet card with shadow-violet-200/50, a button with shadow-blue-300/40 — "
            "this reads as premium and intentional. Apply colored shadows only to key "
            "interactive elements, not every card."
        ),
        "tailwind_hints": "shadow-lg shadow-violet-200/50 for branded elements, hover:shadow-xl, combine with the primary color",
        "vibe": ["branded", "premium", "vibrant", "modern", "intentional", "distinctive", "accent"],
    },
    {
        "id": "shadow_flat",
        "category": "shadow",
        "name": "Flat Bordered",
        "rule": (
            "Zero shadows. Define elevation purely with border-width and background contrast: "
            "active/selected cards get border-2 border-gray-900, default cards get "
            "border border-gray-200 on bg-white over bg-gray-50. "
            "The flat aesthetic reads as confident and graphic."
        ),
        "tailwind_hints": "no shadow classes, border border-gray-200 for cards, border-2 border-gray-900 for active states",
        "vibe": ["flat", "brutalist", "graphic", "editorial", "clean", "bold", "sharp"],
    },
    {
        "id": "shadow_inner",
        "category": "shadow",
        "name": "Inner Depth",
        "rule": (
            "Use inset shadows on input fields and wells to create tactile depth: shadow-inner "
            "on text inputs, crisp focus rings (ring-2 ring-offset-2) on focus. "
            "Outer shadows on buttons only on active/pressed state (active:translate-y-px + "
            "active:shadow-sm). Creates a satisfying physical button feel."
        ),
        "tailwind_hints": "shadow-inner on inputs, focus:ring-2 focus:ring-offset-2 for focus, active:translate-y-px active:shadow-sm",
        "vibe": ["tactile", "physical", "interactive", "polished", "app", "product", "form"],
    },

    # ── CARD ────────────────────────────────────────────────────────────────────

    {
        "id": "card_glass",
        "category": "card",
        "name": "Glassmorphism",
        "rule": (
            "Cards with frosted glass effect: semi-transparent white background (bg-white/70), "
            "backdrop-blur-md, subtle white border (border border-white/30), very light shadow. "
            "Use over gradient or image backgrounds for maximum visual impact. "
            "The layered depth is visually striking and unmistakably modern."
        ),
        "tailwind_hints": "bg-white/70 backdrop-blur-md border border-white/30 shadow-lg shadow-black/5 rounded-2xl",
        "vibe": ["modern", "premium", "visual", "overlay", "gradient-bg", "striking", "glass"],
    },
    {
        "id": "card_bento",
        "category": "card",
        "name": "Bento Grid Cards",
        "rule": (
            "Arrange feature/content cards in an asymmetric bento-box grid: mix tile sizes "
            "(some span 2 columns or 2 rows), give each card a distinct background tinted from "
            "the palette family, vary heights. Use grid-cols-3 with explicit column/row spans. "
            "Each tile is a distinct visual entity."
        ),
        "tailwind_hints": "grid grid-cols-3 gap-4 auto-rows-[200px], col-span-2 for wide tiles, row-span-2 for tall tiles",
        "vibe": ["modern", "portfolio", "feature", "dashboard", "creative", "distinctive", "mosaic"],
    },
    {
        "id": "card_bordered",
        "category": "card",
        "name": "Clean Bordered",
        "rule": (
            "White cards with a single 1px border (border-gray-200 on white backgrounds). "
            "No shadow. Featured/highlighted cards get a 4px top border in the brand accent "
            "(border-t-4 border-violet-500 or border-t-4 border-amber-500). "
            "Clean, fast-loading, zero visual noise."
        ),
        "tailwind_hints": "bg-white border border-gray-200 rounded-xl, border-t-4 border-violet-500 for featured cards",
        "vibe": ["clean", "minimal", "saas", "professional", "b2b", "fast", "light"],
    },
    {
        "id": "card_floating",
        "category": "card",
        "name": "Floating Elevated",
        "rule": (
            "Cards appear to float off the page: bg-white, rounded-2xl, shadow-xl. "
            "On hover: -translate-y-1 and shadow-2xl to reveal the float. "
            "Use a light background behind cards (bg-gray-50 or bg-stone-100) so the "
            "elevation contrast is visible. The hover motion creates delightful interactivity."
        ),
        "tailwind_hints": "shadow-xl hover:-translate-y-1 hover:shadow-2xl transition-all duration-200, bg-white on bg-gray-50",
        "vibe": ["interactive", "modern", "product", "engaging", "polished", "clean", "hover"],
    },
    {
        "id": "card_dark",
        "category": "card",
        "name": "Dark Surface Cards",
        "rule": (
            "Use dark cards (bg-gray-900 or bg-zinc-900) as the primary container even in "
            "light-mode apps. White text on dark cards creates striking contrast. "
            "Accent the card with a subtle gradient top border or ring-1 ring-white/10. "
            "Reserve white cards for secondary/supporting content."
        ),
        "tailwind_hints": "bg-gray-900 text-white rounded-2xl p-6, ring-1 ring-white/10 for border, white cards for secondary",
        "vibe": ["bold", "tech", "developer", "dark", "striking", "contrast", "night"],
    },

    # ── BUTTON ──────────────────────────────────────────────────────────────────

    {
        "id": "button_gradient_pill",
        "category": "button",
        "name": "Gradient Pill CTA",
        "rule": (
            "Primary button: rounded-full with a gradient background "
            "(from-violet-600 to-blue-500, or from-orange-500 to-rose-500), white text "
            "font-semibold, px-8 py-3. On hover: brightness-110 and scale-105. "
            "Secondary: ghost rounded-full with border-2 matching the gradient start color."
        ),
        "tailwind_hints": "rounded-full bg-gradient-to-r from-violet-600 to-blue-500 hover:brightness-110 hover:scale-105 transition-all",
        "vibe": ["vibrant", "consumer", "cta", "energetic", "app", "marketing", "colorful"],
    },
    {
        "id": "button_solid_minimal",
        "category": "button",
        "name": "Solid Minimal",
        "rule": (
            "Primary: solid brand color, rounded-lg, font-medium, no gradients. "
            "Hover: darken one shade. Secondary: bg-gray-100 text-gray-900 hover:bg-gray-200. "
            "Destructive: bg-red-600 hover:bg-red-700. "
            "The restraint makes the primary button feel authoritative."
        ),
        "tailwind_hints": "bg-violet-700 hover:bg-violet-800 text-white rounded-lg px-5 py-2.5 font-medium transition-colors",
        "vibe": ["saas", "b2b", "professional", "clean", "authoritative", "minimal", "standard"],
    },
    {
        "id": "button_brutalist",
        "category": "button",
        "name": "Brutalist Offset",
        "rule": (
            "Buttons with thick border (border-2 border-black or border-gray-900), "
            "white or yellow fill, dark text. An offset box shadow creates a 3D press effect: "
            "shadow-[4px_4px_0_0_black]. On hover: translate-x-0.5 translate-y-0.5 "
            "reducing shadow to [2px_2px]. The physical press is satisfying and distinctive."
        ),
        "tailwind_hints": "border-2 border-black shadow-[4px_4px_0_0_black] hover:shadow-[2px_2px_0_0_black] hover:translate-x-0.5 hover:translate-y-0.5 transition-all",
        "vibe": ["brutalist", "editorial", "graphic", "bold", "portfolio", "distinctive", "physical"],
    },
    {
        "id": "button_ghost_refined",
        "category": "button",
        "name": "Ghost Refined",
        "rule": (
            "Make ghost/outlined buttons the dominant style: border border-gray-300 rounded-lg "
            "text-gray-700 hover:bg-gray-50 hover:border-gray-400. Reserve ONE solid filled "
            "button for the single most critical action on screen. "
            "The restraint makes that sole filled button impossible to miss."
        ),
        "tailwind_hints": "border border-gray-300 hover:border-gray-400 hover:bg-gray-50, filled bg only for the single primary CTA",
        "vibe": ["refined", "minimal", "premium", "editorial", "understated", "luxury", "restrained"],
    },

    # ── ANIMATION ──────────────────────────────────────────────────────────────

    {
        "id": "anim_subtle_lift",
        "category": "animation",
        "name": "Subtle Lift",
        "rule": (
            "All interactive cards: hover:-translate-y-1 transition-all duration-200 ease-out. "
            "Buttons: hover:scale-[1.02]. Links: expanding underline via pseudo-element "
            "(scaleX 0→1). All transitions at 150-200ms — fast enough to feel responsive, "
            "slow enough to register as intentional motion."
        ),
        "tailwind_hints": "hover:-translate-y-1 transition-all duration-200 for cards, hover:scale-[1.02] for buttons, duration-150 for color changes",
        "vibe": ["polished", "premium", "interactive", "engaging", "smooth", "professional"],
    },
    {
        "id": "anim_spring",
        "category": "animation",
        "name": "Spring Bounce",
        "rule": (
            "Spring-like motion throughout: buttons on click compress (active:scale-95) then "
            "release. Cards on enter fade-in and slide up (translate-y-4 → translate-y-0 with "
            "opacity 0→1). Use ease-out for entrances, ease-in for exits. "
            "Adds playful energy without crossing into gimmicky territory."
        ),
        "tailwind_hints": "active:scale-95 for buttons, fade-in + slide-in-from-bottom-4 for card entrances, duration-300",
        "vibe": ["playful", "consumer", "app", "engaging", "lively", "modern", "fun"],
    },
    {
        "id": "anim_smooth_slide",
        "category": "animation",
        "name": "Smooth Slide",
        "rule": (
            "Directional motion for all state changes: panels slide from the right "
            "(translate-x-full → translate-x-0), modals fade+scale "
            "(opacity-0 scale-95 → opacity-100 scale-100), dropdowns slide from -translate-y-2. "
            "All at duration-200 ease-out. Never use abrupt show/hide."
        ),
        "tailwind_hints": "transition-transform duration-200 ease-out for panels, scale-95 opacity-0 for modal entry, -translate-y-2 for dropdowns",
        "vibe": ["smooth", "polished", "app", "navigation", "fluid", "professional", "motion"],
    },
    {
        "id": "anim_minimal_fade",
        "category": "animation",
        "name": "Minimal Fade",
        "rule": (
            "Restrained motion — all transitions are opacity-based fades only, no movement. "
            "duration-150 for micro-interactions (hover color changes), "
            "duration-300 for content transitions (tab switches, panel loads). "
            "Zero transform animations. The stillness reads as confident and editorial."
        ),
        "tailwind_hints": "transition-opacity duration-150 for hover, duration-300 for content changes, no translate/scale transforms",
        "vibe": ["editorial", "premium", "luxury", "minimal", "confident", "restrained", "still"],
    },

    # ── LAYOUT ──────────────────────────────────────────────────────────────────

    {
        "id": "layout_bento_grid",
        "category": "layout",
        "name": "Bento Feature Grid",
        "rule": (
            "For feature/content sections: use an asymmetric bento grid of 3 columns where "
            "tiles mix sizes (1×1, 2×1, 1×2). Each tile gets a unique background tinted from "
            "the brand palette. Fixed auto-rows height creates the mosaic grid structure. "
            "Visual interest without complex illustration work."
        ),
        "tailwind_hints": "grid-cols-3 gap-4 auto-rows-[200px], col-span-2 for wide, row-span-2 for tall tiles",
        "vibe": ["modern", "feature", "portfolio", "grid", "visual", "distinctive", "mosaic"],
    },
    {
        "id": "layout_asymmetric_hero",
        "category": "layout",
        "name": "Asymmetric Hero Split",
        "rule": (
            "Split the hero into 60/40 columns: large headline + CTA on the left, "
            "a product screenshot or illustration on the right with a slight rotation "
            "(-rotate-2 or rotate-1) and drop shadow. Add a subtle gradient blob behind "
            "the image. The asymmetry deliberately breaks template-like symmetry."
        ),
        "tailwind_hints": "grid-cols-5, text col-span-3, image col-span-2 with -rotate-2 shadow-2xl relative",
        "vibe": ["landing", "marketing", "saas", "product", "hero", "expressive", "launch"],
    },
    {
        "id": "layout_sidebar_canvas",
        "category": "layout",
        "name": "Fixed Sidebar Canvas",
        "rule": (
            "Fixed narrow sidebar (w-64) with brand color or deep dark background for navigation. "
            "Main content area takes the remaining width as a white/light canvas. "
            "Sidebar nav items use a subtle active highlight (bg-white/10 or solid accent). "
            "This is the canonical app shell layout for tools and dashboards."
        ),
        "tailwind_hints": "flex h-screen, sidebar w-64 bg-gray-900 fixed left-0, main ml-64 flex-1 overflow-auto",
        "vibe": ["app", "dashboard", "saas", "admin", "product", "tool", "sidebar"],
    },
    {
        "id": "layout_magazine",
        "category": "layout",
        "name": "Magazine Grid",
        "rule": (
            "Use a 12-column grid with intentional irregularity: headline spans 8 cols, "
            "intro text spans 5 cols offset, images break out of the grid with negative margins. "
            "Mix full-bleed sections with contained content columns. "
            "Content feels curated, not auto-generated."
        ),
        "tailwind_hints": "grid-cols-12, col-span-8 for main content, col-span-5 col-start-2 for intro, -mx-8 for bleed images",
        "vibe": ["editorial", "content", "blog", "portfolio", "magazine", "expressive", "writing"],
    },
    {
        "id": "layout_centered_focus",
        "category": "layout",
        "name": "Centered Focus Column",
        "rule": (
            "All content in a centered max-w-2xl or max-w-3xl column with generous horizontal "
            "padding. Single-column flow throughout. Use horizontal dividers (divide-y) to "
            "separate sections instead of background alternation. The narrow focus column "
            "forces content quality and creates a premium reading experience."
        ),
        "tailwind_hints": "max-w-2xl mx-auto px-6, divide-y divide-gray-200 for sections, py-12 between sections",
        "vibe": ["focused", "form", "settings", "auth", "reading", "minimal", "structured"],
    },

    # ── TEXTURE ─────────────────────────────────────────────────────────────────

    {
        "id": "texture_grain",
        "category": "texture",
        "name": "Subtle Grain Overlay",
        "rule": (
            "Add a subtle noise/grain texture to hero backgrounds and colored sections using "
            "a CSS SVG filter or a semi-transparent grain overlay at opacity-[0.03] to "
            "opacity-[0.06]. The grain adds analog warmth that distinguishes the app from "
            "purely sterile digital interfaces. Imperceptible on first glance, unmistakable on inspection."
        ),
        "tailwind_hints": "relative overflow-hidden, before:absolute before:inset-0 before:bg-[url('/noise.png')] before:opacity-[0.04] before:pointer-events-none",
        "vibe": ["premium", "warm", "analog", "editorial", "distinctive", "textured", "rich"],
    },
    {
        "id": "texture_dot_grid",
        "category": "texture",
        "name": "Dot Grid Background",
        "rule": (
            "Use a subtle dot grid pattern via CSS radial-gradient on the page background or "
            "hero section. Very low contrast: gray-200 dots on gray-50 for light mode, "
            "gray-800 dots on gray-950 for dark. The pattern adds visual depth and suggests "
            "precision and structure without competing with content."
        ),
        "tailwind_hints": "bg-[radial-gradient(circle,_#e5e7eb_1px,_transparent_1px)] bg-[size:20px_20px] for light dot grid",
        "vibe": ["tech", "developer", "saas", "structured", "systematic", "precision", "grid"],
    },
    {
        "id": "texture_gradient_mesh",
        "category": "texture",
        "name": "Gradient Mesh",
        "rule": (
            "Use a soft multi-color gradient mesh as the hero or page background: "
            "3 overlapping radial gradients in brand colors at very low opacity (0.15-0.25) "
            "over white, combined with blur-3xl. The result is a sophisticated color wash — "
            "more complex than a linear gradient but never distracting."
        ),
        "tailwind_hints": "multiple absolute divs with radial-gradient + blur-3xl + opacity-20, z-0 behind content z-10",
        "vibe": ["vibrant", "modern", "hero", "landing", "colorful", "premium", "gradient"],
    },
    {
        "id": "texture_clean",
        "category": "texture",
        "name": "Pure Flat Clean",
        "rule": (
            "No textures, patterns, or gradients on backgrounds. Pure solid colors only: "
            "white, gray-50, gray-100 for all surfaces. All visual interest comes from "
            "typography, color blocks, and well-crafted components. "
            "The absolute cleanliness IS the design statement — restraint as luxury."
        ),
        "tailwind_hints": "bg-white, bg-gray-50, bg-gray-100 only — zero gradient or pattern utilities on backgrounds",
        "vibe": ["ultra-minimal", "clean", "fast", "swiss", "professional", "timeless", "restrained"],
    },
]

CATEGORIES = list({r["category"] for r in UI_RULES})
