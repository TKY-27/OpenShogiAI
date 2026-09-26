//! Evaluator-neutral session primitives shared by both USI adapters: the output
//! boundary, completion gating for deferred `bestmove` publication, stale-output
//! suppression, bounded line reading, and evidence formatting.

use std::{
    io::{self, BufRead, Write},
    sync::{Condvar, Mutex, PoisonError},
};

use open_shogi_core::{MATE_SCORE, SearchInfo, is_mate_score, to_usi_move};

/// A line-oriented output boundary. Implementations must not add logging to the USI stream.
pub trait ProtocolSink: Send + Sync {
    fn send(&self, line: &str);
}

pub(crate) struct WriterSink<W: Write + Send> {
    writer: Mutex<W>,
}

impl<W: Write + Send> WriterSink<W> {
    pub(crate) fn new(writer: W) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl<W: Write + Send> ProtocolSink for WriterSink<W> {
    // Write failures are ignored by contract: a GUI that died while leaving stdin
    // open still closes stdin, which ends the session through the reader thread.
    fn send(&self, line: &str) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writeln!(writer, "{line}");
            let _ = writer.flush();
        }
    }
}

/// Deferred publication for `go infinite`: the worker retains its completed result but
/// the gate keeps it from writing `bestmove` until the owning session releases it.
#[derive(Default)]
pub(crate) struct CompletionGate {
    released: Mutex<bool>,
    notification: Condvar,
}

impl CompletionGate {
    pub(crate) fn release(&self) {
        if let Ok(mut released) = self.released.lock() {
            *released = true;
            self.notification.notify_all();
        }
    }

    /// Waits until released. `false` means the gate is poisoned and the completion must
    /// stay unpublished.
    pub(crate) fn wait(&self) -> bool {
        let Ok(mut released) = self.released.lock() else {
            return false;
        };
        while !*released {
            let Ok(next) = self.notification.wait(released) else {
                return false;
            };
            released = next;
        }
        true
    }
}

/// Generation counter for stale-output suppression. Advancing the generation before
/// superseding a worker makes every later write from that worker invisible.
#[derive(Default)]
pub(crate) struct OutputAuthority {
    generation: Mutex<u64>,
}

impl OutputAuthority {
    /// Saturates only after 2^64 advances; equality-based suppression is unaffected at
    /// any reachable generation count.
    pub(crate) fn advance(&self) -> u64 {
        let mut generation = self
            .generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *generation = generation.saturating_add(1);
        *generation
    }

    /// The generation currently accepted by the session; superseded generations are
    /// stale for both output and failure reporting.
    #[cfg(feature = "pure-only")]
    pub(crate) fn current(&self) -> u64 {
        *self
            .generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn send_if_current<F>(&self, generation: u64, sink: &dyn ProtocolSink, make_line: F)
    where
        F: FnOnce() -> String,
    {
        let current = self
            .generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if *current == generation {
            sink.send(&make_line());
        }
    }
}

pub(crate) enum BoundedLine {
    Line(String),
    TooLong,
    Eof,
}

/// Reads one newline-terminated line without ever allocating more than `maximum + 1`
/// bytes, even when a peer sends a line without a newline.
pub(crate) fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    maximum: usize,
) -> io::Result<BoundedLine> {
    let mut bytes = Vec::with_capacity(maximum.min(4_096));
    let mut saw_input = false;
    let mut too_long = false;
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !saw_input {
                return Ok(BoundedLine::Eof);
            }
            return finish_bounded_line(bytes, too_long);
        }
        saw_input = true;
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_length = newline.unwrap_or(buffer.len());
        if !too_long {
            let remaining = maximum.saturating_add(1).saturating_sub(bytes.len());
            let copy_length = content_length.min(remaining);
            bytes.extend_from_slice(&buffer[..copy_length]);
            too_long = copy_length < content_length || bytes.len() > maximum;
        }
        let consumed = newline.map_or(buffer.len(), |index| index + 1);
        reader.consume(consumed);
        if newline.is_some() {
            return finish_bounded_line(bytes, too_long);
        }
    }
}

fn finish_bounded_line(bytes: Vec<u8>, too_long: bool) -> io::Result<BoundedLine> {
    if too_long {
        return Ok(BoundedLine::TooLong);
    }
    String::from_utf8(bytes)
        .map(BoundedLine::Line)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Replaces control characters so peer-controlled text can never inject or truncate
/// USI protocol lines.
pub(crate) fn sanitized(message: &str) -> String {
    message
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

/// Search-statistics prefix shared by progress and completion lines.
pub(crate) fn format_search_stats(
    depth: u8,
    seldepth: u8,
    nodes: u64,
    nps: u64,
    elapsed_ms: u128,
) -> String {
    format!("info depth {depth} seldepth {seldepth} nodes {nodes} nps {nps} time {elapsed_ms}")
}

/// One completed-depth progress line built only from search evidence, with `pv` last.
pub(crate) fn format_search_info(info: &SearchInfo) -> String {
    let pv = info
        .pv
        .iter()
        .copied()
        .map(to_usi_move)
        .collect::<Vec<_>>()
        .join(" ");
    let mut line = format!(
        "{} score {}",
        format_search_stats(
            info.depth,
            info.seldepth,
            info.nodes,
            info.nps,
            info.elapsed.as_millis(),
        ),
        score_field(info.score)
    );
    if !pv.is_empty() {
        line.push_str(" pv ");
        line.push_str(&pv);
    }
    line
}

fn score_field(score: i32) -> String {
    if !is_mate_score(score) {
        return format!("cp {score}");
    }
    let distance = MATE_SCORE.saturating_sub(score.saturating_abs()).max(0);
    let moves = (distance + 1) / 2;
    if score < 0 && moves != 0 {
        format!("mate -{moves}")
    } else {
        format!("mate {moves}")
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use super::ProtocolSink;

    /// In-process output sink for protocol tests.
    #[derive(Default)]
    pub(crate) struct MemorySink(pub(crate) Mutex<Vec<String>>);

    impl ProtocolSink for MemorySink {
        fn send(&self, line: &str) {
            self.0.lock().unwrap().push(line.to_owned());
        }
    }

    impl MemorySink {
        pub(crate) fn lines(&self) -> std::sync::MutexGuard<'_, Vec<String>> {
            self.0.lock().unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{BoundedLine, OutputAuthority, read_bounded_line};
    use crate::lifecycle::test_support::MemorySink;

    #[test]
    fn invalidated_output_authority_suppresses_stale_worker_output() {
        let authority = OutputAuthority::default();
        let sink = MemorySink::default();
        let generation = authority.advance();

        authority.send_if_current(generation, &sink, || "current".to_owned());
        authority.advance();
        authority.send_if_current(generation, &sink, || "stale".to_owned());

        assert_eq!(*sink.lines(), ["current"]);
    }

    #[test]
    fn bounded_reader_drains_overlong_no_newline_input() {
        let mut reader = Cursor::new(vec![b'x'; 4_097]);
        assert!(matches!(
            read_bounded_line(&mut reader, 4_096).unwrap(),
            BoundedLine::TooLong
        ));
        assert!(matches!(
            read_bounded_line(&mut reader, 4_096).unwrap(),
            BoundedLine::Eof
        ));
    }

    #[test]
    fn sanitized_output_drops_control_characters() {
        assert_eq!(super::sanitized("a\nb\r\nc\u{7}d"), "a b  c d");
    }
}
