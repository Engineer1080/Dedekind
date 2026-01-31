# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""
Palettes for dark and light themes used in Spyder.
"""

# Third-party imports
from qdarkstyle.colorsystem import Blue, Gray
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette

# Local imports
from spyder.config.gui import is_dark_interface, is_dedekind_theme
from spyder.utils.color_system import (Green, Red, Orange, GroupDark,
                                       GroupLight, Logos)

# =============================================================================
# ---- Dedekind Studio palettes (green accent theme)
# =============================================================================
class DedekindPaletteDark(DarkPalette):
    """Dark palette for Dedekind Studio with green accents and green-tinted UI."""

    # ---- QDarkStyle base overrides (green instead of blue) ----
    COLOR_ACCENT_1 = Green.B50
    COLOR_ACCENT_2 = Green.B60
    COLOR_ACCENT_3 = Green.B70
    COLOR_ACCENT_4 = Green.B80
    COLOR_ACCENT_5 = Green.B90

    # Green-tinted dark backgrounds (replacing blue-grays)
    COLOR_BACKGROUND_1 = '#162018'
    COLOR_BACKGROUND_2 = '#1e2d24'
    COLOR_BACKGROUND_3 = '#243328'
    COLOR_BACKGROUND_4 = '#2a3d30'
    COLOR_BACKGROUND_5 = '#354a3d'
    COLOR_BACKGROUND_6 = '#3d5648'

    # Muted green for disabled elements
    COLOR_DISABLED = '#5a7a65'

    # Colors for information and feedback in dialogs
    COLOR_SUCCESS_1 = Green.B40
    COLOR_SUCCESS_2 = Green.B70
    COLOR_SUCCESS_3 = Green.B90

    COLOR_ERROR_1 = Red.B40
    COLOR_ERROR_2 = Red.B70
    COLOR_ERROR_3 = Red.B110

    COLOR_WARN_1 = Orange.B40
    COLOR_WARN_2 = Orange.B70
    COLOR_WARN_3 = Orange.B90
    COLOR_WARN_4 = Orange.B100

    # Icon colors - green accent
    ICON_1 = Gray.B140
    ICON_2 = Green.B80
    ICON_3 = Green.B90
    ICON_4 = Red.B70
    ICON_5 = Orange.B70
    ICON_6 = Gray.B30
    ICON_7 = GroupDark.B90

    # Colors for icons and variable explorer in dark mode
    GROUP_1 = GroupDark.B10
    GROUP_2 = GroupDark.B20
    GROUP_3 = GroupDark.B30
    GROUP_4 = GroupDark.B40
    GROUP_5 = GroupDark.B50
    GROUP_6 = GroupDark.B60
    GROUP_7 = GroupDark.B70
    GROUP_8 = GroupDark.B80
    GROUP_9 = GroupDark.B90
    GROUP_10 = GroupDark.B100
    GROUP_11 = GroupDark.B110
    GROUP_12 = GroupDark.B120

    # Colors for highlight in editor - green
    COLOR_HIGHLIGHT_1 = Green.B10
    COLOR_HIGHLIGHT_2 = Green.B20
    COLOR_HIGHLIGHT_3 = Green.B30
    COLOR_HIGHLIGHT_4 = Green.B50

    # Colors for occurrences from find widget
    COLOR_OCCURRENCE_1 = Gray.B10
    COLOR_OCCURRENCE_2 = Gray.B20
    COLOR_OCCURRENCE_3 = Gray.B30
    COLOR_OCCURRENCE_4 = Gray.B50
    COLOR_OCCURRENCE_5 = Gray.B80

    # Colors for Spyder and Python logos
    PYTHON_LOGO_UP = Logos.B10
    PYTHON_LOGO_DOWN = Logos.B20
    SPYDER_LOGO_BACKGROUND = Logos.B30
    SPYDER_LOGO_WEB = Logos.B40
    SPYDER_LOGO_SNAKE = Logos.B40

    # For special tabs
    SPECIAL_TABS_SEPARATOR = Gray.B70
    SPECIAL_TABS_SELECTED = Green.B60

    # For the heart used to ask for donations
    COLOR_HEART = Green.B80

    # For editor tooltips
    TIP_TITLE_COLOR = Green.B80
    TIP_CHAR_HIGHLIGHT_COLOR = Orange.B90


# =============================================================================
# ---- Spyder palettes
# =============================================================================
class SpyderPaletteDark(DarkPalette):
    """Dark palette for Spyder."""

    # Colors for information and feedback in dialogs
    COLOR_SUCCESS_1 = Green.B40
    COLOR_SUCCESS_2 = Green.B70
    COLOR_SUCCESS_3 = Green.B90

    COLOR_ERROR_1 = Red.B40
    COLOR_ERROR_2 = Red.B70
    COLOR_ERROR_3 = Red.B110

    COLOR_WARN_1 = Orange.B40
    COLOR_WARN_2 = Orange.B70
    COLOR_WARN_3 = Orange.B90
    COLOR_WARN_4 = Orange.B100

    # Icon colors
    ICON_1 = Gray.B140
    ICON_2 = Blue.B80
    ICON_3 = Green.B80
    ICON_4 = Red.B70
    ICON_5 = Orange.B70
    ICON_6 = Gray.B30
    ICON_7 = GroupDark.B90

    # Colors for icons and variable explorer in dark mode
    GROUP_1 = GroupDark.B10
    GROUP_2 = GroupDark.B20
    GROUP_3 = GroupDark.B30
    GROUP_4 = GroupDark.B40
    GROUP_5 = GroupDark.B50
    GROUP_6 = GroupDark.B60
    GROUP_7 = GroupDark.B70
    GROUP_8 = GroupDark.B80
    GROUP_9 = GroupDark.B90
    GROUP_10 = GroupDark.B100
    GROUP_11 = GroupDark.B110
    GROUP_12 = GroupDark.B120

    # Colors for highlight in editor
    COLOR_HIGHLIGHT_1 = Blue.B10
    COLOR_HIGHLIGHT_2 = Blue.B20
    COLOR_HIGHLIGHT_3 = Blue.B30
    COLOR_HIGHLIGHT_4 = Blue.B50

    # Colors for occurrences from find widget
    COLOR_OCCURRENCE_1 = Gray.B10
    COLOR_OCCURRENCE_2 = Gray.B20
    COLOR_OCCURRENCE_3 = Gray.B30
    COLOR_OCCURRENCE_4 = Gray.B50
    COLOR_OCCURRENCE_5 = Gray.B80

    # Colors for Spyder and Python logos
    PYTHON_LOGO_UP = Logos.B10
    PYTHON_LOGO_DOWN = Logos.B20
    SPYDER_LOGO_BACKGROUND = Logos.B30
    SPYDER_LOGO_WEB = Logos.B40
    SPYDER_LOGO_SNAKE = Logos.B40

    # For special tabs
    SPECIAL_TABS_SEPARATOR = Gray.B70
    SPECIAL_TABS_SELECTED = DarkPalette.COLOR_ACCENT_2

    # For the heart used to ask for donations
    COLOR_HEART = Blue.B80

    # For editor tooltips
    TIP_TITLE_COLOR = Green.B80
    TIP_CHAR_HIGHLIGHT_COLOR = Orange.B90


class SpyderPaletteLight(LightPalette):
    """Light palette for Spyder."""

    # Colors for information and feedback in dialogs
    COLOR_SUCCESS_1 = Green.B40
    COLOR_SUCCESS_2 = Green.B70
    COLOR_SUCCESS_3 = Green.B30

    COLOR_ERROR_1 = Red.B40
    COLOR_ERROR_2 = Red.B70
    COLOR_ERROR_3 = Red.B110

    COLOR_WARN_1 = Orange.B40
    COLOR_WARN_2 = Orange.B70
    COLOR_WARN_3 = Orange.B50
    COLOR_WARN_4 = Orange.B40

    # Icon colors
    ICON_1 = Gray.B30
    ICON_2 = Blue.B50
    ICON_3 = Green.B30
    ICON_4 = Red.B70
    ICON_5 = Orange.B70
    ICON_6 = Gray.B140
    ICON_7 = GroupLight.B90

    # Colors for icons and variable explorer in light mode
    GROUP_1 = GroupLight.B10
    GROUP_2 = GroupLight.B20
    GROUP_3 = GroupLight.B30
    GROUP_4 = GroupLight.B40
    GROUP_5 = GroupLight.B50
    GROUP_6 = GroupLight.B60
    GROUP_7 = GroupLight.B70
    GROUP_8 = GroupLight.B80
    GROUP_9 = GroupLight.B90
    GROUP_10 = GroupLight.B100
    GROUP_11 = GroupLight.B110
    GROUP_12 = GroupLight.B120

    # Colors for highlight in editor
    COLOR_HIGHLIGHT_1 = Blue.B140
    COLOR_HIGHLIGHT_2 = Blue.B130
    COLOR_HIGHLIGHT_3 = Blue.B120
    COLOR_HIGHLIGHT_4 = Blue.B110

    # Colors for occurrences from find widget
    COLOR_OCCURRENCE_1 = Gray.B120
    COLOR_OCCURRENCE_2 = Gray.B110
    COLOR_OCCURRENCE_3 = Gray.B100
    COLOR_OCCURRENCE_4 = Gray.B90
    COLOR_OCCURRENCE_5 = Gray.B60

    # Colors for Spyder and Python logos
    PYTHON_LOGO_UP = Logos.B10
    PYTHON_LOGO_DOWN = Logos.B20
    SPYDER_LOGO_BACKGROUND = Logos.B30
    SPYDER_LOGO_WEB = Logos.B50
    SPYDER_LOGO_SNAKE = Logos.B30

    # For special tabs
    SPECIAL_TABS_SEPARATOR = Gray.B70
    SPECIAL_TABS_SELECTED = LightPalette.COLOR_ACCENT_5

    # For the heart used to ask for donations
    COLOR_HEART = Red.B70

    # For editor tooltips
    TIP_TITLE_COLOR = Green.B20
    TIP_CHAR_HIGHLIGHT_COLOR = Orange.B30


# =============================================================================
# ---- Exported classes
# =============================================================================
if is_dedekind_theme():
    SpyderPalette = DedekindPaletteDark
elif is_dark_interface():
    SpyderPalette = SpyderPaletteDark
else:
    SpyderPalette = SpyderPaletteLight
