use crate::models::PostEntity;

#[derive(Debug, Clone, PartialEq)]
enum SpanKind {
    Bold,
    Code(Option<String>),
}

#[derive(Debug, Clone)]
struct Span {
    offset: u64,
    length: u64,
    kind: SpanKind,
}

pub fn build_reply_with_entities(raw: &str, max_len: usize) -> (String, Vec<PostEntity>) {
    let (text, spans) = parse_entities(raw);

    let spans = merge_spans(spans);

    let (text, truncated) = truncate_chars(&text, max_len);
    let spans = clip_spans(spans, text.chars().count());

    let entities: Vec<PostEntity> = spans
        .into_iter()
        .filter(|s| s.length > 0)
        .map(|s| PostEntity {
            entity_type: match s.kind {
                SpanKind::Bold => "bold".to_string(),
                SpanKind::Code(_) => "code_block".to_string(),
            },
            offset: s.offset,
            length: s.length,
            color: None,
            font_size_value: None,
            language: match s.kind {
                SpanKind::Code(lang) => lang,
                SpanKind::Bold => None,
            },
        })
        .collect();

    let mut text = text;
    if truncated {
        text.push('…');
    }

    (text, entities)
}

/// One pass over the raw reply:
/// - fenced code blocks (```lang ... ```) have their fence lines stripped and
///   become `code_block` spans; content inside them is emitted verbatim (no
///   bold/at-handle processing),
/// - `**...**` becomes a `bold` span (markers stripped),
/// - `@handle` becomes a `bold` span (outside code only),
/// - single backticks wrapping a bare URL (the model's inline-code habit)
///   are stripped so the link stays clickable (outside code blocks only).
fn parse_entities(raw: &str) -> (String, Vec<Span>) {
    let chars: Vec<char> = raw.chars().collect();
    let mut text = String::with_capacity(raw.len());
    let mut spans: Vec<Span> = Vec::new();
    let mut out_chars: u64 = 0;

    let mut i = 0;
    let mut in_bold = false;
    let mut bold_start: u64 = 0;
    let mut in_code = false;
    let mut code_start: u64 = 0;
    let mut code_lang: Option<String> = None;

    while i < chars.len() {
        if chars[i] == '`' && i + 2 < chars.len() && chars[i + 1] == '`' && chars[i + 2] == '`' {
            if !in_code {
                if at_line_start(&text) {
                    // Opening fence: consume it plus the language tag up to the newline.
                    in_code = true;
                    code_start = out_chars;
                    code_lang = None;
                    i += 3;
                    let mut lang = String::new();
                    while i < chars.len() && chars[i] != '\n' {
                        if !chars[i].is_whitespace() {
                            lang.push(chars[i]);
                        }
                        i += 1;
                    }
                    if i < chars.len() {
                        i += 1; // consume the fence line's newline
                    }
                    if !lang.is_empty() {
                        code_lang = Some(sanitize_language(&lang));
                    }
                    continue;
                }
            } else if at_line_start(&text) && fence_line_is_blank(&chars, i + 3) {
                // Closing fence: only a lone ``` on its line closes the block.
                // The app renders the code block without a trailing newline
                // when it is the final content, so trim it in that case.
                let has_following = {
                    let mut j = i + 3;
                    while j < chars.len() && chars[j].is_whitespace() {
                        j += 1;
                    }
                    j < chars.len()
                };
                if !has_following {
                    trim_trailing_newlines(&mut text, &mut out_chars);
                }
                let len = out_chars - code_start;
                if len > 0 {
                    spans.push(Span {
                        offset: code_start,
                        length: len,
                        kind: SpanKind::Code(code_lang.clone()),
                    });
                }
                code_lang = None;
                in_code = false;
                let mut j = i + 3;
                while j < chars.len() && chars[j] != '\n' {
                    j += 1;
                }
                i = j;
                if i < chars.len() {
                    i += 1; // consume the closing fence's newline
                }
                continue;
            }
            // Not a fence in this position: fall through and emit backticks.
        }

        if !in_code {
            // The model habitually wraps URLs in inline-code backticks. Strip
            // the backticks when they enclose a bare URL so the link survives.
            if let Some(close) = inline_url_close(&chars, i) {
                let mut j = i + 1;
                while j < close {
                    text.push(chars[j]);
                    out_chars += 1;
                    j += 1;
                }
                i = close + 1;
                continue;
            }
        }

        if in_code {
            text.push(chars[i]);
            out_chars += 1;
            i += 1;
            continue;
        }

        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if in_bold {
                let len = out_chars - bold_start;
                if len > 0 {
                    spans.push(Span {
                        offset: bold_start,
                        length: len,
                        kind: SpanKind::Bold,
                    });
                }
                in_bold = false;
            } else {
                bold_start = out_chars;
                in_bold = true;
            }
            i += 2;
            continue;
        }

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
                    spans.push(Span {
                        offset: out_chars,
                        length: (j - i) as u64,
                        kind: SpanKind::Bold,
                    });
                    while i < j {
                        text.push(chars[i]);
                        out_chars += 1;
                        i += 1;
                    }
                    continue;
                }
            }
        }

        text.push(chars[i]);
        out_chars += 1;
        i += 1;
    }

    // Unclosed code fence: the markers are dropped (mirrors unclosed bold);
    // the emitted content stays as plain text without a span. Trim the
    // trailing newline the same way a closed block would.
    if in_code {
        trim_trailing_newlines(&mut text, &mut out_chars);
    }
    (text, spans)
}

/// Pop trailing newlines from the emitted text (the app's code blocks do not
/// keep the newline that precedes the closing fence).
fn trim_trailing_newlines(text: &mut String, out_chars: &mut u64) {
    let mut removed = 0;
    while text.ends_with('\n') {
        text.pop();
        removed += 1;
    }
    *out_chars -= removed as u64;
}

/// True when everything on the current output line is whitespace (a fence must
/// start its own line — no inline code fences).
fn at_line_start(text: &str) -> bool {
    match text.rfind('\n') {
        Some(idx) => text[idx + 1..].chars().all(char::is_whitespace),
        None => text.chars().all(char::is_whitespace),
    }
}

/// For a candidate closing fence: every char up to the end of the line (or end
/// of input) is whitespace.
fn fence_line_is_blank(chars: &[char], from: usize) -> bool {
    let mut j = from;
    while j < chars.len() && chars[j] != '\n' {
        if !chars[j].is_whitespace() {
            return false;
        }
        j += 1;
    }
    true
}

/// If `chars[i]` is a backtick that opens inline code wrapping a bare URL
/// (`` `https://…` ``, `` `qwen.ai/blog` ``), return the index of the closing
/// backtick. Content with whitespace is never a URL and is left alone.
fn inline_url_close(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let mut content = String::new();
    while j < chars.len() && chars[j] != '`' && chars[j] != '\n' {
        if chars[j].is_whitespace() {
            return None;
        }
        content.push(chars[j]);
        j += 1;
    }
    if j >= chars.len() || chars[j] != '`' {
        return None;
    }
    is_bare_url(&content).then_some(j)
}

/// Loose URL check: an http(s) link, or a dotted domain (with optional path
/// and query). Matches bare domains like `qwen.ai/blog` but not plain words
/// like `rustc`.
fn is_bare_url(s: &str) -> bool {
    if s.starts_with("http://") || s.starts_with("https://") {
        return s.len() > "https://".len();
    }
    let host = s.split('/').next().unwrap_or(s);
    host.find('.').is_some()
        && host.chars().all(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

/// Keep only characters that make sense in a language tag.
fn sanitize_language(lang: &str) -> String {
    lang.chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '+' | '#' | '-' | '_' | '.'))
        .collect()
}

fn merge_spans(mut spans: Vec<Span>) -> Vec<Span> {
    if spans.len() < 2 {
        return spans;
    }
    spans.sort_unstable_by_key(|s| s.offset);

    let mut merged: Vec<Span> = Vec::new();
    for span in spans {
        let end = span.offset + span.length;
        match merged.last_mut() {
            Some(last) if last.kind == span.kind && span.offset <= last.offset + last.length => {
                let last_end = last.offset + last.length;
                if end > last_end {
                    last.length = end - last.offset;
                }
            }
            _ => merged.push(span),
        }
    }
    merged
}

fn clip_spans(spans: Vec<Span>, text_chars: usize) -> Vec<Span> {
    let limit = text_chars as u64;
    let mut clipped = Vec::new();
    for span in spans {
        if span.offset >= limit {
            continue;
        }
        let end = (span.offset + span.length).min(limit);
        if end > span.offset {
            clipped.push(Span {
                offset: span.offset,
                length: end - span.offset,
                kind: span.kind,
            });
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

    fn span_of(entities: &[PostEntity], i: usize) -> (String, u64, u64) {
        (
            entities[i].entity_type.clone(),
            entities[i].offset,
            entities[i].length,
        )
    }

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

    #[test]
    fn code_block_strips_fences_and_records_language() {
        let raw = "Here is some code:\n\n```rust\nfn main() {}\n```\nThat's it";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Here is some code:\n\nfn main() {}\nThat's it");
        assert_eq!(entities.len(), 1);
        assert_eq!(span_of(&entities, 0), ("code_block".to_string(), 20, 13));
        assert_eq!(entities[0].language.as_deref(), Some("rust"));
    }

    #[test]
    fn code_block_without_language_omits_field() {
        let raw = "```\nfn main() {}\n```";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "fn main() {}");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "code_block");
        assert_eq!(entities[0].language, None);
    }

    #[test]
    fn bold_inside_code_stays_literal() {
        let raw = "```python\nx = ** 2\nprint(x)\n```";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "x = ** 2\nprint(x)");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "code_block");
    }

    #[test]
    fn at_handle_inside_code_is_not_highlighted() {
        let raw = "```\nuser = \"@toast\"\n```";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "user = \"@toast\"");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "code_block");
    }

    #[test]
    fn code_plus_bold_produce_both_entities() {
        let raw = "**Note:**\n```js\nlet x = 1;\n```";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Note:\nlet x = 1;");
        assert_eq!(entities.len(), 2);
        assert_eq!(span_of(&entities, 0).0, "bold");
        assert_eq!(span_of(&entities, 1).0, "code_block");
        assert_eq!(entities[1].language.as_deref(), Some("js"));
    }

    #[test]
    fn inline_backticks_are_left_alone() {
        let raw = "Use `rustc` in the terminal";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Use `rustc` in the terminal");
        assert!(entities.is_empty());
    }

    #[test]
    fn backticks_around_full_url_are_stripped() {
        let raw = "Announcement: `https://qwen.ai/blog?id=qwen3.8` here";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Announcement: https://qwen.ai/blog?id=qwen3.8 here");
        assert!(entities.is_empty());
    }

    #[test]
    fn backticks_around_bare_domain_are_stripped() {
        let raw = "على:\n`qwen.ai/blog`";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "على:\nqwen.ai/blog");
        assert!(entities.is_empty());
    }

    #[test]
    fn inline_code_with_spaces_is_not_a_url() {
        let raw = "Run `cargo build --release` first";
        let (text, _) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "Run `cargo build --release` first");
    }

    #[test]
    fn unclosed_backtick_stays_put() {
        let raw = "see `https://x.test here";
        let (text, _) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "see `https://x.test here");
    }

    #[test]
    fn backtick_strip_keeps_span_offsets_correct() {
        let raw = "`qwen.ai/blog` is **cool**";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "qwen.ai/blog is cool");
        assert_eq!(entities.len(), 1);
        assert_eq!((entities[0].offset, entities[0].length), (16, 4));
    }

    #[test]
    fn backticks_inside_code_block_are_kept() {
        let raw = "```rust\n// see `https://x.test` in code\n```";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "// see `https://x.test` in code");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "code_block");
    }

    #[test]
    fn unclosed_code_fence_keeps_content_plain() {
        let raw = "```rust\nfn main() {}\n";
        let (text, entities) = build_reply_with_entities(raw, 100);
        assert_eq!(text, "fn main() {}");
        assert!(entities.is_empty());
    }
}
