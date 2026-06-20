//! Memory guard — reads `/proc/meminfo` to detect memory pressure before
//! large operations (insert, build_index, chunking). Returns a clear error
//! instead of letting the OOM-killer kill the process.
//!
//! **Linux only.** On non-Linux platforms (macOS, Windows), `read_meminfo_kb`
//! returns `None` and all checks return `Normal`.  This is a deliberate
//! trade-off: Linux is the primary deployment target for vector databases,
//! and `/proc/meminfo` is the most reliable source of memory availability.
//! On other platforms, memory pressure detection should be handled by the
//! embedding runtime (e.g., ONNX Runtime) or the OS itself.

use crate::error::{Error, Result};

const WARN_FREE_PCT: f64 = 25.0;
const CRITICAL_FREE_PCT: f64 = 20.0;

/// Result of a memory availability check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPressure {
    Normal,
    Low {
        free_pct: f64,
        free_mb: f64,
        total_mb: f64,
    },
    Critical {
        free_pct: f64,
        free_mb: f64,
        total_mb: f64,
    },
}

impl std::fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryPressure::Normal => write!(f, "memory normal"),
            MemoryPressure::Low {
                free_pct,
                free_mb,
                total_mb,
            } => {
                write!(
                    f,
                    "memory LOW: {:.1}% free ({:.0} MB of {:.0} MB total)",
                    free_pct, free_mb, total_mb
                )
            }
            MemoryPressure::Critical {
                free_pct,
                free_mb,
                total_mb,
            } => {
                write!(
                    f,
                    "memory CRITICAL: {:.1}% free ({:.0} MB of {:.0} MB total)",
                    free_pct, free_mb, total_mb
                )
            }
        }
    }
}

/// Read available / total memory from `/proc/meminfo`.
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

/// Returns `Normal`, `Low` (warning logged, continue), or `Critical` (error).
pub fn check_memory() -> Result<MemoryPressure> {
    let Some((avail_kb, total_kb)) = read_meminfo_kb() else {
        return Ok(MemoryPressure::Normal);
    };

    let free_pct = (avail_kb as f64 / total_kb as f64) * 100.0;
    let free_mb = avail_kb as f64 / 1024.0;
    let total_mb = total_kb as f64 / 1024.0;

    if free_pct < CRITICAL_FREE_PCT {
        let pressure = MemoryPressure::Critical {
            free_pct,
            free_mb,
            total_mb,
        };
        log::error!("{pressure} — aborting to avoid OOM");
        return Err(Error::OutOfMemory(format!(
            "{pressure} — aborting to avoid OOM"
        )));
    }

    if free_pct < WARN_FREE_PCT {
        let pressure = MemoryPressure::Low {
            free_pct,
            free_mb,
            total_mb,
        };
        log::warn!("{pressure} — continuing cautiously");
        Ok(pressure)
    } else {
        Ok(MemoryPressure::Normal)
    }
}

/// Convenience wrapper: returns `Err` on critical pressure, `Ok` otherwise.
pub fn ensure_memory() -> Result<()> {
    match check_memory()? {
        MemoryPressure::Critical { .. } => {
            unreachable!("Critical should have returned Err from check_memory")
        }
        MemoryPressure::Low { .. } => Ok(()),
        MemoryPressure::Normal => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_meminfo() {
        let result = read_meminfo_kb();
        assert!(
            result.is_some(),
            "/proc/meminfo should be readable on Linux"
        );
        let (avail, total) = result.unwrap();
        assert!(total > 0, "MemTotal must be > 0");
        assert!(avail <= total, "MemAvailable must not exceed MemTotal");
    }

    #[test]
    fn test_check_memory_returns_ok() {
        let result = check_memory();
        assert!(
            result.is_ok(),
            "check_memory should not fail under normal conditions"
        );
    }

    #[test]
    fn test_memory_pressure_display() {
        let p = MemoryPressure::Normal;
        assert!(!p.to_string().is_empty());

        let p = MemoryPressure::Low {
            free_pct: 8.5,
            free_mb: 600.0,
            total_mb: 7400.0,
        };
        let s = p.to_string();
        assert!(s.contains("LOW"));
        assert!(s.contains("8.5%"));

        let p = MemoryPressure::Critical {
            free_pct: 3.2,
            free_mb: 240.0,
            total_mb: 7400.0,
        };
        let s = p.to_string();
        assert!(s.contains("CRITICAL"));
        assert!(s.contains("3.2%"));
    }
}
