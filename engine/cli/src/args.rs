use std::{
    io::{self, BufRead},
    path::PathBuf,
    str::FromStr,
};

pub enum BoundedInputLine {
    Line(String),
    TooLong,
    Eof,
}

pub fn next_value<'a>(
    arguments: &'a [String],
    index: &mut usize,
    option: &str,
) -> Result<&'a str, String> {
    *index = index.saturating_add(1);
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{option} requires a value"))
}

pub fn parse_next<T>(arguments: &[String], index: &mut usize, option: &str) -> Result<T, String>
where
    T: FromStr,
{
    next_value(arguments, index, option)?
        .parse::<T>()
        .map_err(|_| format!("{option} has an invalid value"))
}

pub fn path_next(arguments: &[String], index: &mut usize, option: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(next_value(arguments, index, option)?))
}

pub fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    maximum: usize,
) -> io::Result<BoundedInputLine> {
    let mut bytes = Vec::with_capacity(maximum.min(4_096));
    let mut saw_input = false;
    let mut too_long = false;
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !saw_input {
                return Ok(BoundedInputLine::Eof);
            }
            return finish_line(bytes, too_long);
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
            return finish_line(bytes, too_long);
        }
    }
}

fn finish_line(bytes: Vec<u8>, too_long: bool) -> io::Result<BoundedInputLine> {
    if too_long {
        return Ok(BoundedInputLine::TooLong);
    }
    String::from_utf8(bytes)
        .map(BoundedInputLine::Line)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{BoundedInputLine, read_bounded_line};

    #[test]
    fn bounded_reader_drains_no_newline_overflow() {
        let mut reader = Cursor::new(vec![b'x'; 129]);
        assert!(matches!(
            read_bounded_line(&mut reader, 128).unwrap(),
            BoundedInputLine::TooLong
        ));
        assert!(matches!(
            read_bounded_line(&mut reader, 128).unwrap(),
            BoundedInputLine::Eof
        ));
    }
}
