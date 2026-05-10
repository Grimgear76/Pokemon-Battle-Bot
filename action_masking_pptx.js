const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Action Masking in Pokemon Battle Bot";
pres.author = "Pokemon Battle Bot";

// Palette
const NAVY   = "0D1B3E";
const BLUE   = "1A3A6B";
const GOLD   = "F0C040";
const TEAL   = "0D9488";
const WHITE  = "FFFFFF";
const LGRAY  = "E8EEF8";
const DGRAY  = "334155";
const RED    = "DC2626";
const GREEN  = "16A34A";
const AMBER  = "D97706";

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

// ─── SLIDE 1: TITLE ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Decorative accent bar left
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: GOLD }, line: { color: GOLD } });

  // Subtitle tag
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 1.0, w: 2.6, h: 0.38, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("GEN 2 RANDOM BATTLE RL", { x: 0.45, y: 1.0, w: 2.6, h: 0.38, fontSize: 9, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });

  // Main title
  s.addText([
    { text: "Action Masking", options: { breakLine: true } },
    { text: "in the Pokemon Battle Bot", options: { fontSize: 28, bold: false, color: "C0D0F0" } },
  ], { x: 0.45, y: 1.55, w: 9.0, h: 2.2, fontSize: 46, bold: true, color: WHITE, fontFace: "Georgia" });

  // Description
  s.addText("How illegal & wasteful actions are eliminated from the agent's decision space each turn.", {
    x: 0.45, y: 3.85, w: 7.5, h: 0.7, fontSize: 14, color: "9EB8E0", italic: true,
  });

  // Bottom rule
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 4.75, w: 9.1, h: 0, line: { color: GOLD, width: 1.5 } });
  s.addText("environment.py  ·  mask_env()  ·  _is_move_allowed()", {
    x: 0.45, y: 4.85, w: 9.1, h: 0.5, fontSize: 11, color: "7090C0", fontFace: "Consolas",
  });
}

// ─── SLIDE 2: WHAT IS ACTION MASKING? ────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LGRAY };

  s.addText("What is Action Masking?", { x: 0.45, y: 0.3, w: 9.1, h: 0.65, fontSize: 28, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 1.0, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  // Left column: concept
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 1.15, w: 4.6, h: 3.8, fill: { color: WHITE }, line: { color: "C8D8F0" }, shadow: makeShadow() });
  s.addText("The Concept", { x: 0.55, y: 1.25, w: 4.4, h: 0.4, fontSize: 14, bold: true, color: TEAL });
  s.addText([
    { text: "The agent chooses from 11 discrete actions each turn.", options: { breakLine: true } },
    { text: "\nWithout masking", options: { bold: true, color: RED, breakLine: false } },
    { text: ", the PPO policy wastes gradient on actions that are either impossible (no move in that slot) or strategically nonsensical (using a maxed stat-up move).", options: { breakLine: true } },
    { text: "\nWith masking", options: { bold: true, color: GREEN, breakLine: false } },
    { text: ", illegal actions are zeroed out in the softmax before sampling — the agent never attempts them.", options: {} },
  ], { x: 0.55, y: 1.7, w: 4.4, h: 3.1, fontSize: 13, color: DGRAY });

  // Right column: benefits
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 1.15, w: 4.25, h: 3.8, fill: { color: NAVY }, line: { color: NAVY }, shadow: makeShadow() });
  s.addText("Why It Matters", { x: 5.4, y: 1.25, w: 4.05, h: 0.4, fontSize: 14, bold: true, color: GOLD });
  const benefits = [
    ["Faster convergence", "Policy learns meaningful distinctions sooner"],
    ["No wasted gradient", "Loss stays focused on real strategic choices"],
    ["Prevents illegal moves", "Struggle, recharge handled cleanly"],
    ["Anti-stall", "Hard blocks consecutive switch-loops"],
    ["Domain rules enforced", "Sleep Talk, Rest, immunity, weather — all gated"],
  ];
  benefits.forEach(([title, desc], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.75 + i * 0.62, w: 0.28, h: 0.28, fill: { color: GOLD }, line: { color: GOLD } });
    s.addText([
      { text: title + "  ", options: { bold: true, color: WHITE } },
      { text: desc, options: { color: "A0B8D8" } },
    ], { x: 5.78, y: 1.72 + i * 0.62, w: 3.65, h: 0.55, fontSize: 12 });
  });
}

// ─── SLIDE 3: ACTION SPACE ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: WHITE };

  s.addText("The 11-Action Space", { x: 0.45, y: 0.28, w: 9.1, h: 0.58, fontSize: 28, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  // Three groups
  const groups = [
    {
      label: "SWITCH ACTIONS",
      color: BLUE,
      x: 0.3,
      actions: [
        ["0", "Switch to team slot 0"],
        ["1", "Switch to team slot 1"],
        ["2", "Switch to team slot 2"],
        ["3", "Switch to team slot 3"],
        ["4", "Switch to team slot 4"],
        ["5", "Switch to team slot 5"],
      ],
    },
    {
      label: "MOVE ACTIONS",
      color: TEAL,
      x: 3.6,
      actions: [
        ["6", "Use move slot 0"],
        ["7", "Use move slot 1"],
        ["8", "Use move slot 2"],
        ["9", "Use move slot 3"],
      ],
    },
    {
      label: "DEFAULT ACTION",
      color: AMBER,
      x: 6.9,
      actions: [
        ["10", "Struggle / Recharge /\nForced fallback"],
      ],
    },
  ];

  groups.forEach(({ label, color, x, actions }) => {
    const totalH = 0.55 * actions.length + 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.0, w: 3.0, h: totalH, fill: { color: color }, line: { color: color } });
    s.addText(label, { x: x + 0.1, y: 1.0, w: 2.8, h: 0.5, fontSize: 10, bold: true, color: WHITE, charSpacing: 2, margin: 0, valign: "middle" });
    actions.forEach(([num, desc], i) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.58 + i * 0.55, w: 3.0, h: 0.5,
        fill: { color: i % 2 === 0 ? "F0F4FA" : WHITE },
        line: { color: "D0DCF0" },
      });
      s.addText(num, { x: x + 0.08, y: 1.58 + i * 0.55, w: 0.42, h: 0.5, fontSize: 16, bold: true, color: color, align: "center", valign: "middle", margin: 0 });
      s.addText(desc, { x: x + 0.55, y: 1.58 + i * 0.55, w: 2.35, h: 0.5, fontSize: 12, color: DGRAY, valign: "middle", margin: 0 });
    });
  });

  // Slot alignment note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 4.85, w: 9.45, h: 0.55, fill: { color: "FEF9E7" }, line: { color: GOLD } });
  s.addText("Switch slots 0–5 use team insertion order to stay in sync with embed_battle_impl(). Move slots 6–9 map to the active pokémon's moves.values() in order.", {
    x: 0.42, y: 4.88, w: 9.2, h: 0.48, fontSize: 11.5, color: DGRAY, italic: true, valign: "middle",
  });
}

// ─── SLIDE 4: IMPLEMENTATION STACK ──────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LGRAY };

  s.addText("Implementation Stack", { x: 0.45, y: 0.28, w: 9.1, h: 0.58, fontSize: 28, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  const layers = [
    { label: "MaskablePPO (sb3_contrib)", sub: "Policy reads the mask before softmax; never samples illegal actions", color: NAVY, textColor: WHITE },
    { label: "ActionMasker Wrapper", sub: "Calls mask_env(env) every step to produce a binary (0/1) mask of shape (11,)", color: BLUE, textColor: WHITE },
    { label: "mask_env()  ←  environment.py", sub: "Core logic: iterates moves & switches, applies all validity checks, returns np.int8 array", color: TEAL, textColor: WHITE },
    { label: "_is_move_allowed()  ←  environment.py", sub: "Per-move filter: type immunity, Sleep Talk, Rest, heals, status, weather, stat-boost saturation", color: "0F766E", textColor: WHITE },
    { label: "poke-env  ·  available_moves / available_switches", sub: "Engine-level lists; mask_env uses these as the first gate before applying semantic filters", color: DGRAY, textColor: WHITE },
  ];

  layers.forEach(({ label, sub, color, textColor }, i) => {
    const y = 1.05 + i * 0.84;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 9.1, h: 0.72, fill: { color }, line: { color }, shadow: makeShadow() });
    s.addText(label, { x: 0.58, y: y + 0.05, w: 9.0, h: 0.3, fontSize: 13, bold: true, color: textColor, fontFace: "Consolas", margin: 0 });
    s.addText(sub, { x: 0.58, y: y + 0.37, w: 8.9, h: 0.28, fontSize: 11, color: i < 2 ? "B0C8E8" : "A0D8D0", margin: 0 });

    // Arrow down (except last)
    if (i < layers.length - 1) {
      s.addShape(pres.shapes.RECTANGLE, { x: 4.87, y: y + 0.72, w: 0.26, h: 0.12, fill: { color: GOLD }, line: { color: GOLD } });
    }
  });
}

// ─── SLIDE 5: mask_env() FLOW ─────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: WHITE };

  s.addText("mask_env() — Decision Flow", { x: 0.45, y: 0.28, w: 9.1, h: 0.58, fontSize: 28, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  // Flow boxes
  const steps = [
    { text: "Start: build zeros mask [11]", y: 1.05, w: 3.8, x: 3.1, color: BLUE, textColor: WHITE, h: 0.48 },
    { text: "No battle / no active mon?", y: 1.72, w: 3.8, x: 3.1, color: DGRAY, textColor: WHITE, h: 0.48, diamond: true },
    { text: "No moves AND no switches?  OR  must_recharge?", y: 2.39, w: 4.6, x: 2.7, color: DGRAY, textColor: WHITE, h: 0.48, diamond: true },
    { text: "Enable action 10 only  (Struggle / Recharge)", y: 3.06, w: 4.4, x: 2.8, color: RED, textColor: WHITE, h: 0.48 },
    { text: "For each move slot 6–9 in available_moves → _is_move_allowed() → enable", y: 1.05, w: 3.5, x: 0.25, color: TEAL, textColor: WHITE, h: 0.7 },
    { text: "For each switch slot 0–5 in available_switches → apply allow/block logic → enable", y: 2.05, w: 3.5, x: 0.25, color: TEAL, textColor: WHITE, h: 0.7 },
    { text: "Nothing enabled? → fallback: enable action 10", y: 3.06, w: 3.4, x: 0.25, color: AMBER, textColor: WHITE, h: 0.48 },
    { text: "Return mask array → MaskablePPO", y: 3.73, w: 3.8, x: 3.1, color: NAVY, textColor: WHITE, h: 0.48 },
    { text: "Return mask array → MaskablePPO", y: 3.73, w: 3.4, x: 0.25, color: NAVY, textColor: WHITE, h: 0.48 },
  ];

  // Simplified, linear flow on right side; parallel on left
  // Right: main conditional chain
  const main = [
    { text: "Initialize: mask = np.zeros(11)", y: 1.05, color: BLUE },
    { text: "No battle / no active mon?  →  action 10 only, return", y: 1.65, color: DGRAY },
    { text: "No moves & no switches, OR must_recharge?  →  action 10, return", y: 2.25, color: DGRAY },
    { text: "Not force_switch: evaluate move slots 6–9 via _is_move_allowed()", y: 2.85, color: TEAL },
    { text: "Evaluate switch slots 0–5 with allow_switch & block logic", y: 3.45, color: TEAL },
    { text: "If mask still all-zero  →  enable action 10 (final fallback)", y: 4.05, color: AMBER },
    { text: "Return mask  →  MaskablePPO samples from valid set", y: 4.65, color: NAVY },
  ];

  main.forEach(({ text, y, color }, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 9.1, h: 0.48, fill: { color }, line: { color }, shadow: makeShadow() });
    s.addText(`${i + 1}.  ${text}`, { x: 0.58, y, w: 8.9, h: 0.48, fontSize: 12.5, bold: i === 0 || i === 6, color: WHITE, valign: "middle", fontFace: i === 0 || i === 6 ? "Georgia" : "Calibri" });
    if (i < main.length - 1) {
      s.addShape(pres.shapes.RECTANGLE, { x: 4.87, y: y + 0.48, w: 0.26, h: 0.12, fill: { color: GOLD }, line: { color: GOLD } });
    }
  });
}

// ─── SLIDE 6: MOVE MASKING RULES ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LGRAY };

  s.addText("Move Masking Rules  —  _is_move_allowed()", { x: 0.3, y: 0.22, w: 9.5, h: 0.58, fontSize: 25, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.3, y: 0.85, w: 9.5, h: 0, line: { color: GOLD, width: 2 } });

  const rules = [
    { rule: "Type Immunity", cond: "Move deals 0× damage vs. opponent's type pair", result: "BLOCK", color: RED },
    { rule: "Sleep Talk", cond: "Move ID is sleeptalk AND own mon is not asleep", result: "BLOCK", color: RED },
    { rule: "Rest", cond: "Own mon is already asleep, OR HP > 80%, OR healthy & HP > 50%", result: "BLOCK", color: RED },
    { rule: "Healing Moves", cond: "Move has heal AND HP > 85%  (Rest exempt)", result: "BLOCK", color: RED },
    { rule: "Status Moves", cond: "Move inflicts status AND opponent already has a status condition", result: "BLOCK", color: RED },
    { rule: "Weather Moves", cond: "Move sets weather AND that weather is already active", result: "BLOCK", color: RED },
    { rule: "Stat Boost Saturation", cond: "Non-damaging; relevant boost already at +6 (Agility/SD/Growth/etc.)", result: "BLOCK", color: RED },
    { rule: "Incapacitation Rule", cond: "Own mon asleep or frozen AND move is non-damaging AND not Sleep Talk", result: "BLOCK", color: RED },
    { rule: "All other moves", cond: "Move is in available_moves and none of the above fire", result: "ALLOW", color: GREEN },
  ];

  rules.forEach(({ rule, cond, result, color }, i) => {
    const y = 1.0 + i * 0.49;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y, w: 9.5, h: 0.44, fill: { color: i % 2 === 0 ? WHITE : "F0F4FA" }, line: { color: "D0DCF0" } });
    s.addText(rule, { x: 0.38, y, w: 1.85, h: 0.44, fontSize: 11, bold: true, color: NAVY, valign: "middle", margin: 0 });
    s.addText(cond, { x: 2.28, y, w: 6.1, h: 0.44, fontSize: 11, color: DGRAY, valign: "middle", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.43, y: y + 0.06, w: 1.2, h: 0.32, fill: { color }, line: { color } });
    s.addText(result, { x: 8.43, y: y + 0.06, w: 1.2, h: 0.32, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  });
}

// ─── SLIDE 7: SWITCH MASKING RULES ───────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: WHITE };

  s.addText("Switch Masking Rules", { x: 0.45, y: 0.28, w: 9.1, h: 0.58, fontSize: 28, bold: true, color: NAVY, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  // Left: allow conditions
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.0, w: 4.5, h: 0.42, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("ALLOW SWITCH when…", { x: 0.35, y: 1.0, w: 4.5, h: 0.42, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const allows = [
    "battle.force_switch is True  (KO replacement)",
    "Own mon is incapacitated (asleep or frozen)",
    "Available moves list is non-empty  (has real options)",
  ];
  allows.forEach((txt, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.45 + i * 0.62, w: 4.5, h: 0.54, fill: { color: i % 2 === 0 ? "EDF7F5" : WHITE }, line: { color: "B0E0D8" } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.45 + i * 0.62, w: 0.18, h: 0.54, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(txt, { x: 0.62, y: 1.45 + i * 0.62, w: 4.15, h: 0.54, fontSize: 12.5, color: DGRAY, valign: "middle" });
  });

  // Right: block conditions
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.0, w: 4.5, h: 0.42, fill: { color: RED }, line: { color: RED } });
  s.addText("BLOCK SWITCH when…", { x: 5.15, y: 1.0, w: 4.5, h: 0.42, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const blocks = [
    ["Anti-stall rule fires", "consec_voluntary_switches ≥ 2  AND  at least one move is available  AND  not force_switch  AND  not incapacitated"],
    ["Slot not in available_switches", "poke-env reports this switch slot as unavailable"],
    ["allow_switch is False", "Not forced, not incapacitated, no moves → no switch allowed"],
  ];
  blocks.forEach(([title, desc], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.45 + i * 0.75, w: 4.5, h: 0.67, fill: { color: i % 2 === 0 ? "FEF2F2" : WHITE }, line: { color: "FECACA" } });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.45 + i * 0.75, w: 0.18, h: 0.67, fill: { color: RED }, line: { color: RED } });
    s.addText(title, { x: 5.42, y: 1.48 + i * 0.75, w: 4.1, h: 0.25, fontSize: 12, bold: true, color: RED, margin: 0 });
    s.addText(desc, { x: 5.42, y: 1.73 + i * 0.75, w: 4.1, h: 0.35, fontSize: 11, color: DGRAY, margin: 0 });
  });

  // Note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.4, w: 9.3, h: 0.92, fill: { color: "FFF7E6" }, line: { color: GOLD } });
  s.addText("Voluntary switch detection", { x: 0.5, y: 4.47, w: 9.0, h: 0.25, fontSize: 12, bold: true, color: AMBER, margin: 0 });
  s.addText("A switch is 'voluntary' when: species changed AND agent chose a switch slot AND the previous mon is still alive (ruling out Whirlwind/Roar phazes and forced KO replacements). The counter is reset to 0 whenever the agent uses a move.", {
    x: 0.5, y: 4.68, w: 9.0, h: 0.58, fontSize: 11, color: DGRAY,
  });
}

// ─── SLIDE 8: ANTI-STALL DEEP DIVE ───────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("Anti-Stall Mechanism", { x: 0.45, y: 0.28, w: 9.1, h: 0.58, fontSize: 28, bold: true, color: WHITE, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  // Timeline boxes
  const turns = [
    { label: "Turn N",   action: "Voluntary switch  (slot 0–5, prev mon alive)", sw: 1, color: AMBER },
    { label: "Turn N+1", action: "Voluntary switch again  (different slot)",       sw: 2, color: AMBER },
    { label: "Turn N+2", action: "Voluntary switch BLOCKED — must use a move",     sw: 2, color: RED, blocked: true },
    { label: "Turn N+2", action: "Agent uses a move → counter resets to 0",        sw: 0, color: GREEN },
  ];

  turns.forEach(({ label, action, sw, color, blocked }, i) => {
    const y = 1.1 + i * 1.0;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 1.4, h: 0.72, fill: { color: BLUE }, line: { color: BLUE } });
    s.addText(label, { x: 0.4, y, w: 1.4, h: 0.72, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: 1.9, y, w: 5.5, h: 0.72, fill: { color: blocked ? "3D1515" : "1A3A6B" }, line: { color: blocked ? RED : BLUE } });
    s.addText(action, { x: 2.0, y, w: 5.3, h: 0.72, fontSize: 12.5, color: blocked ? RED : WHITE, valign: "middle" });

    s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y, w: 2.0, h: 0.72, fill: { color }, line: { color } });
    s.addText(`consec_sw = ${sw}${blocked ? "  🚫" : ""}`, { x: 7.55, y, w: 2.0, h: 0.72, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  });

  // Code snippet
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 5.05, w: 9.15, h: 0.38, fill: { color: "0A0F1E" }, line: { color: TEAL } });
  s.addText("block_voluntary_switch = (consec_sw >= 2  and  not force_switch  and  not incapacitated  and  any(action_mask[6:10]))", {
    x: 0.5, y: 5.07, w: 9.0, h: 0.33, fontSize: 10.5, color: "7FE8D8", fontFace: "Consolas", valign: "middle",
  });
}

// ─── SLIDE 9: SUMMARY ────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: GOLD }, line: { color: GOLD } });

  s.addText("Summary", { x: 0.45, y: 0.3, w: 9.0, h: 0.55, fontSize: 32, bold: true, color: WHITE, fontFace: "Georgia" });
  s.addShape(pres.shapes.LINE, { x: 0.45, y: 0.9, w: 9.1, h: 0, line: { color: GOLD, width: 2 } });

  const points = [
    ["11-action discrete space", "Slots 0–5 = switch, 6–9 = move, 10 = default"],
    ["ActionMasker wrapper", "sb3_contrib.common.wrappers.ActionMasker wraps every env"],
    ["mask_env() is called every step", "Returns np.int8[11] — 1 = valid, 0 = masked out"],
    ["9 per-move semantic filters", "Immunity, Sleep Talk, Rest, heals, status, weather, stat caps, incapacitation"],
    ["Switch gating", "Force-switch, incapacitation, and move availability govern whether switches are offered"],
    ["Anti-stall hard rule", "≥2 consecutive voluntary switches → all switch slots masked until a move is used"],
    ["Fallback action 10", "Always enabled when nothing else is valid (Struggle / Recharge)"],
  ];

  points.forEach(([title, desc], i) => {
    const y = 1.1 + i * 0.61;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 0.36, h: 0.42, fill: { color: GOLD }, line: { color: GOLD } });
    s.addText(`${i + 1}`, { x: 0.45, y, w: 0.36, h: 0.42, fontSize: 14, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0 });
    s.addText([
      { text: title + "  ", options: { bold: true, color: WHITE } },
      { text: desc, options: { color: "8EB4D8" } },
    ], { x: 0.92, y, w: 8.6, h: 0.42, fontSize: 12.5, valign: "middle" });
  });

  s.addShape(pres.shapes.LINE, { x: 0.45, y: 5.25, w: 9.1, h: 0, line: { color: "2A4070", width: 1 } });
  s.addText("environment.py  ·  mask_env()  ·  _is_move_allowed()  ·  sb3_contrib ActionMasker  ·  MaskablePPO", {
    x: 0.45, y: 5.3, w: 9.1, h: 0.28, fontSize: 10, color: "4A6090", fontFace: "Consolas",
  });
}

pres.writeFile({ fileName: "Action_Masking_Summary.pptx" })
  .then(() => console.log("✅  Action_Masking_Summary.pptx written"))
  .catch(err => { console.error(err); process.exit(1); });
