use crate::models::PostEntity;

pub fn build_reply_with_entities(raw: &str, max_len: usize) -> (String, Vec<PostEntity>) {
    let (text, mut spans) = strip_markers(raw);

    spans.extend(find_at_handles(&text));
    spans = merge_spans(spans);

    let (text, truncated) = truncate_chars(&text, max_len);
    let spans = clip_spans(spans, text.chars().count());

    let entities: Vec<PostEntity> = spans
        .into_iter()
        .filter(|(_, len)| *len > 0)
        .map(|(offset, length)| PostEntity {
            entity_type: "bold".to_string(),
            offset,
            length,
            color: None,
            font_size_value: None,
        })
        .collect();

    let mut text = text;
    if truncated {
        text.push('…');
    }

    (text, entities)
}

fn strip_markers(raw: &str) -> (String, Vec<(u64, u64)>) {
    let mut text = String::with_capacity(raw.len());
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let chars: Vec<char> = raw.chars().collect();

    let mut i = 0;
    let mut in_bold = false;
    let mut span_start: u64 = 0;
    let mut out_chars: u64 = 0;

    while i < chars.len() {
        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if in_bold {
                let len = out_chars - span_start;
                if len > 0 {
                    spans.push((span_start, len));
                }
                in_bold = false;
            } else {
                span_start = out_chars;
                in_bold = true;
            }
            i += 2;
            continue;
        }
        text.push(chars[i]);
        out_chars += 1;
        i += 1;
    }

    (text, spans)
}

fn find_at_handles(text: &str) -> Vec<(u64, u64)> {
    let mut spans = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '@' {
            let prev_ok = i == 0
                || matches!(
                    chars[i - 1],
                    ' ' | '\t' | '\n' | '\r' | '(' | '[' | '{' | ',' | ';'
                );
            if prev_ok {
                let mut j = i + 1;
                while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                if j > i + 1 {
                    spans.push((i as u64, (j - i) as u64));
                    i = j;
                    continue;
                }
            }
        }
        i += 1;
    }

    merge_spans(spans)
}

fn merge_spans(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if spans.len() < 2 {
        return spans;
    }
    spans.sort_unstable_by_key(|(start, _)| *start);

    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, len) in spans {
        let end = start + len;
        match merged.last_mut() {
            Some((m_start, m_len)) if start <= *m_start + *m_len => {
                let m_end = *m_start + *m_len;
                if end > m_end {
                    *m_len = end - *m_start;
                }
            }
            _ => merged.push((start, len)),
        }
    }
    merged
}

fn clip_spans(spans: Vec<(u64, u64)>, text_chars: usize) -> Vec<(u64, u64)> {
    let limit = text_chars as u64;
    let mut clipped = Vec::new();
    for (start, len) in spans {
        if start >= limit {
            continue;
        }
        let end = (start + len).min(limit);
        if end > start {
            clipped.push((start, end - start));
        }
    }
    clipped
}

fn truncate_chars(text: &str, max_len: usize) -> (String, bool) {
    let count = text.chars().count();
    if count <= max_len {
        return (text.to_string(), false);
    }
    (text.chars().take(max_len).collect(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_bold_markers_and_records_spans() {
        let raw = "Visit **Riyadh** soon";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Visit Riyadh soon");
        assert_eq!(entities.len(), 1);
        assert_eq!((entities[0].offset, entities[0].length), (6, 6));
    }

    #[test]
    fn auto_bolds_at_handles() {
        let raw = "Thanks @toast for the tip";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Thanks @toast for the tip");
        assert_eq!(entities.len(), 1);
        assert_eq!((entities[0].offset, entities[0].length), (7, 6));
    }

    #[test]
    fn merges_overlapping_spans() {
        let raw = "**@toast in Riyadh** now";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "@toast in Riyadh now");
        assert_eq!(entities.len(), 1);
        assert_eq!((entities[0].offset, entities[0].length), (0, 16));
    }

    #[test]
    fn truncates_and_clips_spans() {
        let raw = "**Hello World** and the rest of this long reply goes on and on";
        let (text, entities) = build_reply_with_entities(raw, 15);
        assert_eq!(text, "Hello World and…");
        assert_eq!(entities.len(), 1);
        assert_eq!((entities[0].offset, entities[0].length), (0, 11));
    }

    #[test]
    fn unbalanced_marker_is_dropped() {
        let raw = "Oops **unclosed";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Oops unclosed");
        assert!(entities.is_empty());
    }
}
