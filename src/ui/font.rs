use glyphon::{Attrs, Color, Family, FontSystem};

/// The single typeface policy for Minerust's text UI.
///
/// Windows ships Segoe UI in `C:\\Windows\\Fonts`; loading the files explicitly
/// makes the game independent of font discovery order. On another OS, glyphon
/// falls back to its system sans-serif resolver while retaining one API.
#[derive(Clone, Copy)]
pub struct Font {
    family: Family<'static>,
}

impl Font {
    pub const WINDOWS_FAMILY: &'static str = "Segoe UI";

    pub fn load(_font_system: &mut FontSystem) -> Self {
        #[cfg(target_os = "windows")]
        for path in [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"] {
            // The font can already be registered by FontSystem::new(); loading
            // it again is harmless and ensures portable packaged builds find it.
            let _ = _font_system.db_mut().load_font_file(path);
        }
        let family = if cfg!(target_os = "windows") {
            Family::Name(Self::WINDOWS_FAMILY)
        } else {
            Family::SansSerif
        };
        Self { family }
    }

    pub fn attrs(self) -> Attrs<'static> {
        Attrs::new().family(self.family)
    }

    pub fn colored(self, color: Color) -> Attrs<'static> {
        self.attrs().color(color)
    }
}
