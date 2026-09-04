use glyphon::{Attrs, Color, Family, FontSystem};

/// The single typeface policy for Minerust's text UI.
///
/// Windows ships Segoe UI in `C:\\Windows\\Fonts`; loading the files explicitly
/// makes the game independent of font discovery order. On another OS, glyphon
/// falls back to its system sans-serif resolver while retaining one API.
#[derive(Clone, Copy)]
pub struct Font {
    family: &'static str,
}

impl Font {
    pub const WINDOWS_FAMILY: &'static str = "Segoe UI";

    pub fn load(font_system: &mut FontSystem) -> Self {
        #[cfg(target_os = "windows")]
        for path in [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"] {
            // The font can already be registered by FontSystem::new(); loading
            // it again is harmless and ensures portable packaged builds find it.
            let _ = font_system.db_mut().load_font_file(path);
        }
        Self { family: Self::WINDOWS_FAMILY }
    }

    pub fn attrs(self) -> Attrs<'static> {
        Attrs::new().family(Family::Name(self.family))
    }

    pub fn colored(self, color: Color) -> Attrs<'static> {
        self.attrs().color(color)
    }
}
