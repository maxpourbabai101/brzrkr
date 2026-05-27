"""Theme — gothic dark palette inspired by Berserk / Shamo / Vagabond.

All colors, fonts, and decorative glyphs live here so the entire app
restyles by editing this one file. The vibe: black iron, blood
crimson, parchment bone, ember warning. Heavy serif headers,
monospaced data, generous negative space.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette — Berserk forge / Vagabond ink
# ---------------------------------------------------------------------------
class C:
    # Voids and panels
    VOID         = "#06040a"      # absolute black with violet whisper
    NIGHT        = "#0c080e"      # window background
    PANEL        = "#14101a"      # primary card
    PANEL_HI     = "#1d1824"      # elevated / hover
    IRON         = "#2c2630"      # darker iron rim
    BORDER       = "#3a2030"      # subtle red-tinted border

    # Bloods
    BLOOD        = "#8b0a14"      # primary crimson
    BLOOD_HI     = "#c41e2a"      # brighter accent for hover/select
    BLOOD_DIM    = "#5a0810"      # backgrounds where blood is muted
    EMBER        = "#e54818"      # alert orange-red
    FORGE        = "#ff3010"      # active/critical fire

    # Bones and parchment
    BONE         = "#e6dcc8"      # primary text
    PARCHMENT    = "#b8ad96"      # secondary text
    ASH          = "#7a7068"      # tertiary / muted
    OBSIDIAN     = "#2a2520"      # background for text inputs

    # Vital signs
    LIFE         = "#5a8c4a"      # green for gains (muted, never bright)
    DEATH        = "#a01828"      # red for losses (matches BLOOD)
    OMEN         = "#c4942a"      # amber warning
    WIND         = "#5a7a8a"      # steel-blue, neutral indicator

    # Special
    SIGIL        = "#a52840"      # rune accent (between BLOOD and BLOOD_HI)
    GHOST        = "#3a3038"      # disabled / inactive

    # Vagabond / Shamo ink accents (brush-painted feel)
    INK_DEEP     = "#0a0708"      # near-black, blue undertone (sumi-e shadow)
    INK_MID      = "#1a1418"      # mid-tone wash
    INK_HI       = "#2a2228"      # wet highlight
    BRUSH_RED    = "#7a1018"      # dried blood, painted
    WOUND        = "#d42030"      # fresh wound, kinetic accent
    SCAR         = "#5a3030"      # old scar tissue
    SAGE         = "#4a5040"      # vagabond's robe — muted green-gray
    PAPER        = "#dcd2bc"      # parchment, slightly warmer than BONE
    SUMI         = "#0c0408"      # sumi-e ink — pure dark


# ---------------------------------------------------------------------------
# Typography — Cinzel/Trajan-style serif if available; fallback graceful
# ---------------------------------------------------------------------------
FONT_DISPLAY = ("Cinzel", "Trajan Pro", "Optima", "Georgia", "Times")  # heavy serif headers
FONT_SERIF   = ("Optima", "Georgia", "Hoefler Text", "Times New Roman", "serif")
FONT_MONO    = ("Berkeley Mono", "SF Mono", "Menlo", "Monaco", "Consolas")
FONT_SANS    = ("SF Pro Display", "Helvetica Neue", "Helvetica", "Arial")

# Font sizes
SIZE_TITLE   = 22
SIZE_HEADER  = 14
SIZE_BODY    = 12
SIZE_SMALL   = 11
SIZE_TINY    = 10
SIZE_METRIC  = 28   # big numbers


# ---------------------------------------------------------------------------
# Decorative glyphs (unicode, monospace-safe)
# ---------------------------------------------------------------------------
class G:
    # Gothic flourishes
    SIGIL_L      = "❰"           # decorative left bracket
    SIGIL_R      = "❱"           # decorative right bracket
    CROSS        = "✠"           # iron cross
    ORNATE       = "❦"           # leaf flourish
    DAGGER       = "†"
    STAR         = "✦"
    SUN          = "☉"
    MOON         = "☾"
    SKULL        = "☠"
    SCYTHE       = "🜨"          # mock-runic
    RUNE_F       = "ᚠ"           # rune fehu (wealth)
    RUNE_R       = "ᚱ"           # rune raido (journey)
    RUNE_T       = "ᛏ"           # rune tiwaz (victory)
    RUNE_O       = "ᛟ"           # rune othala (inheritance)

    # Indicators
    DOT_ON       = "●"
    DOT_OFF      = "○"
    DOT_DIM      = "◌"
    UP           = "▲"
    DOWN         = "▼"
    RIGHT        = "▶"
    DIAMOND      = "◆"
    BLOCK        = "█"
    FULL         = "▓"
    HALF         = "▒"
    LIGHT        = "░"

    # Operations
    EXEC         = "⚔"           # crossed swords (action)
    SHIELD       = "⛨"           # shield (defense / risk)
    HOURGLASS    = "⌛"
    GEAR         = "⚙"


# ---------------------------------------------------------------------------
# Themed strings for section headers
# ---------------------------------------------------------------------------
def banner(text: str) -> str:
    """Render a gothic banner heading: ❰ ✠ TEXT ✠ ❱"""
    return f"{G.SIGIL_L} {G.CROSS} {text.upper()} {G.CROSS} {G.SIGIL_R}"


def sub_banner(text: str) -> str:
    return f"{G.ORNATE} {text.upper()} {G.ORNATE}"


# ---------------------------------------------------------------------------
# Status helpers — what color/glyph for a given state
# ---------------------------------------------------------------------------
def state_color(state: str) -> str:
    return {
        "ok":      C.LIFE,
        "alive":   C.LIFE,
        "warn":    C.OMEN,
        "danger":  C.FORGE,
        "err":     C.FORGE,
        "dead":    C.GHOST,
        "neutral": C.ASH,
        "info":    C.WIND,
    }.get(state, C.ASH)


def state_glyph(state: str) -> str:
    return {
        "ok": G.DOT_ON,
        "alive": G.DOT_ON,
        "warn": G.DOT_ON,
        "danger": G.SKULL,
        "err": G.SKULL,
        "dead": G.DOT_OFF,
        "neutral": G.DOT_DIM,
    }.get(state, G.DOT_DIM)


def pnl_color(value: float) -> str:
    if value > 0:
        return C.LIFE
    if value < 0:
        return C.DEATH
    return C.ASH
