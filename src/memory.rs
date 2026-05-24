//! Memory guard — auto-detección de presión de memoria.
//!
//! Lee `/proc/meminfo` en Linux para conocer la memoria disponible
//! antes de operaciones grandes (inserción, build_index, chunking).
//! Si la memoria libre cae por debajo de un umbral, retorna un error
//! claro en vez de dejar que el OOM-killer mate el proceso.
//!
//! # Uso
//!
//! ```ignore
//! use dogma_vdb::memory::{check_memory, MemoryPressure};
//!
//! match check_memory() {
//!     Ok(pressure) => eprintln!("Memoria: {pressure}"),
//!     Err(e) => eprintln!("ERROR: {e}"),
//! }
//! ```

use crate::error::{Error, Result};

/// Umbral de advertencia — por debajo se logea warning pero se continúa.
const WARN_FREE_PCT: f64 = 25.0;

/// Umbral crítico — por debajo se aborta con error claro.
const CRITICAL_FREE_PCT: f64 = 20.0;

/// Resultado del chequeo de memoria.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPressure {
    /// Memoria normal — continuar.
    Normal,
    /// Presión moderada — se permite continuar pero se registra advertencia.
    Low { free_pct: f64, free_mb: f64, total_mb: f64 },
    /// Presión crítica — la operación debe abortarse.
    Critical { free_pct: f64, free_mb: f64, total_mb: f64 },
}

impl std::fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryPressure::Normal => write!(f, "memoria normal"),
            MemoryPressure::Low { free_pct, free_mb, total_mb } => {
                write!(
                    f,
                    "memoria BAJA: {:.1}% libre ({:.0} MB de {:.0} MB total)",
                    free_pct, free_mb, total_mb
                )
            }
            MemoryPressure::Critical { free_pct, free_mb, total_mb } => {
                write!(
                    f,
                    "memoria CRÍTICA: {:.1}% libre ({:.0} MB de {:.0} MB total)",
                    free_pct, free_mb, total_mb
                )
            }
        }
    }
}

/// Lee `/proc/meminfo` y calcula la memoria disponible.
///
/// Returns `(available_kb, total_kb)` o `None` si no se puede leer.
pub fn read_meminfo_kb() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut mem_avail = 0u64;
    let mut mem_total = 0u64;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_avail = rest.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("MemTotal:") {
            mem_total = rest.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    if mem_total == 0 {
        return None;
    }
    Some((mem_avail, mem_total))
}

/// Verifica la memoria disponible.
///
/// - `Normal`: hay suficiente memoria.
/// - `Low`: memoria baja (log + continuar).
/// - `Critical`: memoria crítica (retorna error).
pub fn check_memory() -> Result<MemoryPressure> {
    let Some((avail_kb, total_kb)) = read_meminfo_kb() else {
        // No se puede leer /proc/meminfo — permitir continuar (no estamos en Linux)
        return Ok(MemoryPressure::Normal);
    };

    let free_pct = (avail_kb as f64 / total_kb as f64) * 100.0;
    let free_mb = avail_kb as f64 / 1024.0;
    let total_mb = total_kb as f64 / 1024.0;

    if free_pct < CRITICAL_FREE_PCT {
        let pressure = MemoryPressure::Critical { free_pct, free_mb, total_mb };
        eprintln!("⚠️  {}", pressure);
        return Err(Error::OutOfMemory(format!(
            "{} — abortando para evitar OOM",
            pressure
        )));
    }

    if free_pct < WARN_FREE_PCT {
        let pressure = MemoryPressure::Low { free_pct, free_mb, total_mb };
        eprintln!("⚠️  {} — continuando con precaución", pressure);
        Ok(pressure)
    } else {
        Ok(MemoryPressure::Normal)
    }
}

/// Versión simplificada: retorna `true` si hay memoria suficiente.
/// Si hay presión crítica, retorna `Err` con el mensaje.
/// Si hay presión baja, registra advertencia y retorna `Ok(true)`.
pub fn ensure_memory() -> Result<()> {
    match check_memory()? {
        MemoryPressure::Critical { .. } => {
            // check_memory ya retornó Err en este caso, pero por si acaso:
            unreachable!("Critical should have returned Err from check_memory")
        }
        MemoryPressure::Low { .. } => {
            // Ya se logueó la advertencia, continuar
            Ok(())
        }
        MemoryPressure::Normal => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_meminfo() {
        let result = read_meminfo_kb();
        // En Linux debería funcionar siempre
        assert!(result.is_some(), "/proc/meminfo debería ser legible en Linux");
        let (avail, total) = result.unwrap();
        assert!(total > 0, "MemTotal debe ser > 0");
        assert!(avail <= total, "MemAvailable no debe exceder MemTotal");
    }

    #[test]
    fn test_check_memory_returns_ok() {
        // En un sistema con memoria normal esto debe retornar Normal
        let result = check_memory();
        assert!(result.is_ok(), "check_memory no debe fallar en condiciones normales");
    }

    #[test]
    fn test_memory_pressure_display() {
        let p = MemoryPressure::Normal;
        assert!(!p.to_string().is_empty());

        let p = MemoryPressure::Low { free_pct: 8.5, free_mb: 600.0, total_mb: 7400.0 };
        let s = p.to_string();
        assert!(s.contains("BAJA"));
        assert!(s.contains("8.5%"));

        let p = MemoryPressure::Critical { free_pct: 3.2, free_mb: 240.0, total_mb: 7400.0 };
        let s = p.to_string();
        assert!(s.contains("CRÍTICA"));
        assert!(s.contains("3.2%"));
    }
}
