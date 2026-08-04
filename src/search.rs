//! Web search providers for the `web_search` tool.
//!
//! Primary: Exa's hosted MCP endpoint (`mcp.exa.ai/mcp`) — the same provider
//! opencode uses for its `websearch` tool. It needs no API key (an optional
//! `EXA_API_KEY` env var raises the rate limits) and returns rich, current
//! results built for LLM consumption.
//!
//! Fallback: DuckDuckGo Lite scraping. DDG's bot check rejects requests whose
//! headers put Content-Type before Content-Length, so we set Content-Length as
//! a user header BEFORE Content-Type (reqwest keeps header order) and send the
//! form body manually instead of via `.form()`.

use serde_json::{json, Value};
use std::time::Duration;
use tracing::warn;

/// One normalized search result, shaped for the model.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl SearchResult {
    pub fn to_json(&self) -> Value {
        json!({ "title": self.title, "url": self.url, "snippet": self.snippet })
    }
}

/// Which provider served a search (for logging).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchProvider {
    Exa,
    Ddg,
}

pub fn provider_label(p: SearchProvider) -> &'static str {
    match p {
        SearchProvider::Exa => "exa",
        SearchProvider::Ddg => "ddg",
    }
}

/// Max snippet chars fed back to the model per result.
const SNIPPET_CAP: usize = 300;
/// Default / bounds for the numResults argument.
const DEFAULT_NUM_RESULTS: usize = 8;
const MAX_NUM_RESULTS: usize = 20;

/// Validate/clamp the model-provided result count.
pub fn clamp_num_results(v: Option<u64>) -> usize {
    v.map(|n| (n.clamp(1, MAX_NUM_RESULTS as u64)) as usize)
        .unwrap_or(DEFAULT_NUM_RESULTS)
}

// ── Exa ──

/// Search via Exa's hosted MCP server (JSON-RPC `tools/call`, SSE response).
/// Mirrors opencode's `McpWebSearch.call`.
pub async fn exa_search(
    client: &reqwest::Client,
    query: &str,
    num_results: usize,
) -> Result<Vec<SearchResult>, String> {
    let key = std::env::var("EXA_API_KEY").ok();
    let url = match &key {
        Some(k) => format!("https://mcp.exa.ai/mcp?exaApiKey={}", form_encode(k)),
        None => "https://mcp.exa.ai/mcp".to_string(),
    };
    let args = json!({
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "livecrawl": "fallback",
        "contextMaxCharacters": 10000,
    });
    let text = mcp_call(client, &url, "web_search_exa", args).await?;
    let results = parse_exa_text(&text);
    Ok(results)
}

/// POST a JSON-RPC `tools/call` to an MCP endpoint and return the first
/// non-empty `text` content item (plain JSON or SSE `data:` lines).
async fn mcp_call(
    client: &reqwest::Client,
    url: &str,
    tool: &str,
    args: Value,
) -> Result<String, String> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool, "arguments": args },
    });
    let resp = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "application/json, text/event-stream")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("provider request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("provider returned HTTP {}", resp.status().as_u16()));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| format!("failed to read provider response: {e}"))?;
    extract_mcp_text(&body).ok_or_else(|| "no usable content in provider response".to_string())
}

/// Extract the tool result text from an MCP response body: try the whole body
/// as a JSON payload first, then scan SSE `data: ` lines.
fn extract_mcp_text(body: &str) -> Option<String> {
    if let Some(t) = parse_payload(body) {
        return Some(t);
    }
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if let Some(t) = parse_payload(data) {
                return Some(t);
            }
        }
    }
    None
}

fn parse_payload(s: &str) -> Option<String> {
    let s = s.trim();
    if !s.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(s).ok()?;
    let content = v.get("result")?.get("content")?.as_array()?;
    content
        .iter()
        .find_map(|c| {
            c.get("text")
                .and_then(|t| t.as_str())
                .filter(|t| !t.is_empty())
                .map(String::from)
        })
}

/// Parse an Exa result blob: blocks of `Title:`/`URL:`/`Published:`/... lines
/// followed by `Highlights:` free text, separated by blank lines.
fn parse_exa_text(text: &str) -> Vec<SearchResult> {
    let mut out = Vec::new();
    let mut title = String::new();
    let mut url = String::new();
    let mut snippet = String::new();
    let mut in_highlights = false;

    let flush = |title: &mut String,
                 url: &mut String,
                 snippet: &mut String,
                 in_highlights: &mut bool,
                 out: &mut Vec<SearchResult>| {
        if !url.is_empty() {
            out.push(SearchResult {
                title: title.clone(),
                url: url.clone(),
                snippet: truncate_chars(snippet, SNIPPET_CAP),
            });
        }
        title.clear();
        url.clear();
        snippet.clear();
        *in_highlights = false;
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            flush(&mut title, &mut url, &mut snippet, &mut in_highlights, &mut out);
            continue;
        }
        if let Some(rest) = line.strip_prefix("Title: ") {
            title = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("URL: ") {
            url = rest.trim().to_string();
        } else if line == "Highlights:" {
            in_highlights = true;
        } else if in_highlights {
            if !snippet.is_empty() {
                snippet.push(' ');
            }
            snippet.push_str(line);
        }
    }
    flush(&mut title, &mut url, &mut snippet, &mut in_highlights, &mut out);

    out.retain(|r| !r.url.is_empty() && !r.title.is_empty());
    out
}

// ── DuckDuckGo Lite (fallback) ──

/// Search DuckDuckGo Lite. May retry once when DDG answers with an
/// anomaly/challenge page (a full-size body that parses to zero results).
pub async fn ddg_search(
    client: &reqwest::Client,
    query: &str,
    num_results: usize,
) -> Result<Vec<SearchResult>, String> {
    let mut results = ddg_search_once(client, query).await?;
    if results.is_empty() {
        warn!("ddg_search {query:?}: empty parse (anomaly page?), retrying once");
        tokio::time::sleep(Duration::from_millis(1200)).await;
        results = ddg_search_once(client, query).await?;
    }
    results.truncate(num_results);
    Ok(results)
}

async fn ddg_search_once(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<SearchResult>, String> {
    let body = format!("q={}", form_encode(query));
    let content_length = body.len().to_string();
    let resp = client
        .post("https://lite.duckduckgo.com/lite/")
        .header(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        )
        .header("Content-Length", &content_length)
        .header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|e| format!("search request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("search engine returned HTTP {}", resp.status().as_u16()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("failed to read search results: {e}"))?;
    let raw = String::from_utf8_lossy(&bytes);
    Ok(parse_ddg_lite(&raw)
        .into_iter()
        .map(|(title, url, snippet)| SearchResult { title, url, snippet })
        .filter(|r| !r.url.is_empty())
        .collect())
}

/// Parse DuckDuckGo Lite results page into (title, url, snippet) triples, in
/// page order. Snippets are matched to the nearest result-link that follows
/// them, so a result without a snippet does not shift the pairing.
fn parse_ddg_lite(html: &str) -> Vec<(String, String, String)> {
    let mut results: Vec<(String, String, String)> = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = html[search_from..].find("class='result-link'") {
        let abs = search_from + rel;
        let link_html = &html[abs..];

        // href: walk back from the class attribute to the href="..."
        let href = html[..abs]
            .rfind("href=\"")
            .and_then(|hs| {
                let rest = &html[hs + 6..];
                rest.find('"').map(|end| rest[..end].to_string())
            })
            .unwrap_or_default();

        // title: text between the closing '>' of the anchor and </a>
        let title = link_html
            .find('>')
            .and_then(|gt| {
                let after = &link_html[gt + 1..];
                after.find("</a>").map(|end| clean_html_text(&after[..end]))
            })
            .unwrap_or_default();

        // snippet: the first result-snippet cell between this link and the
        // NEXT result-link, so a result without a snippet stays empty instead
        // of stealing the following result's snippet.
        let segment = match link_html.find("class='result-link'") {
            Some(first) => {
                let rest = &link_html[first + "class='result-link'".len()..];
                match rest.find("class='result-link'") {
                    Some(next) => &rest[..next],
                    None => rest,
                }
            }
            None => link_html,
        };
        let snippet = segment
            .find("class='result-snippet'")
            .and_then(|sp| {
                let cell = &segment[sp..];
                cell.find('>').and_then(|gt| {
                    let after = &cell[gt + 1..];
                    after
                        .find("</td>")
                        .map(|end| truncate_chars(&clean_html_text(&after[..end]), SNIPPET_CAP))
                })
            })
            .unwrap_or_default();

        results.push((title, href, snippet));

        search_from = abs
            + link_html
                .find("</a>")
                .map(|p| p + 4)
                .unwrap_or(link_html.len());
    }
    results
}

/// Strip HTML tags then decode entities.
fn clean_html_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
            continue;
        }
        if in_tag {
            if ch == '>' {
                in_tag = false;
            }
            continue;
        }
        out.push(ch);
    }
    decode_entities(&out)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Percent-encode a string as application/x-www-form-urlencoded (space → '+').
fn form_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(*b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                let mut n = *b;
                for _ in 0..2 {
                    let nib = n >> 4;
                    let c = match nib {
                        0..=9 => (b'0' + nib) as char,
                        _ => (b'A' + nib - 10) as char,
                    };
                    out.push(c);
                    n <<= 4;
                }
            }
        }
    }
    out
}

/// Decode common HTML entities and numeric character references.
pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(rel) = s[i..].find(';') {
                let entity = &s[i..i + rel + 1];
                let decoded = match entity {
                    "&amp;" => "&",
                    "&lt;" => "<",
                    "&gt;" => ">",
                    "&quot;" => "\"",
                    "&apos;" | "&#39;" => "'",
                    "&nbsp;" => " ",
                    "&ndash;" => "–",
                    "&mdash;" => "—",
                    "&hellip;" => "…",
                    "&rsquo;" | "&lsquo;" => "'",
                    "&ldquo;" | "&rdquo;" => "\"",
                    "&copy;" => "©",
                    _ => {
                        if let Some(num) = entity.strip_prefix("&#") {
                            let num = num.strip_suffix(';').unwrap_or(num);
                            let n = if let Some(hex) = num
                                .strip_prefix('x')
                                .or_else(|| num.strip_prefix('X'))
                            {
                                u32::from_str_radix(hex, 16).ok()
                            } else {
                                num.parse::<u32>().ok()
                            };
                            match n.and_then(char::from_u32) {
                                Some(ch) => {
                                    out.push(ch);
                                    i += rel + 1;
                                    continue;
                                }
                                None => {
                                    // fall through: keep the raw text
                                }
                            }
                        }
                        // unknown entity: copy verbatim
                        out.push_str(entity);
                        i += rel + 1;
                        continue;
                    }
                };
                out.push_str(decoded);
                i += rel + 1;
                continue;
            }
        }
        // plain char (may be multi-byte)
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_num_results_bounds() {
        assert_eq!(clamp_num_results(None), 8);
        assert_eq!(clamp_num_results(Some(3)), 3);
        assert_eq!(clamp_num_results(Some(0)), 1);
        assert_eq!(clamp_num_results(Some(99)), 20);
    }

    #[test]
    fn extract_mcp_text_plain_json() {
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"hello results"}]}}"#;
        assert_eq!(extract_mcp_text(body).as_deref(), Some("hello results"));
    }

    #[test]
    fn extract_mcp_text_sse_lines() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"sse result\"}]}}\n\nevent: message\n";
        assert_eq!(extract_mcp_text(body).as_deref(), Some("sse result"));
    }

    #[test]
    fn extract_mcp_text_garbage() {
        assert_eq!(extract_mcp_text("<html>anomaly page</html>"), None);
        assert_eq!(extract_mcp_text(""), None);
    }

    #[test]
    fn parse_exa_text_blocks() {
        let text = "Title: White House to meet with AI companies | CNN Business\n\
            URL: https://www.cnn.com/example\n\
            Published: 2026-08-03T18:51:01.000Z\n\
            Author: Hadas Gold\n\
            Highlights:\n\
            # White House to meet with top AI companies\n\
            Major AI companies will meet on Tuesday.\n\
            \n\
            Title: Another Story | Reuters\n\
            URL: https://www.reuters.com/example\n\
            Published: 2026-08-03T10:00:00.000Z\n\
            Highlights:\n\
            A second headline here.";
        let results = parse_exa_text(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "White House to meet with AI companies | CNN Business");
        assert_eq!(results[0].url, "https://www.cnn.com/example");
        assert!(results[0].snippet.contains("Major AI companies"));
        assert_eq!(results[1].url, "https://www.reuters.com/example");
    }

    #[test]
    fn parse_exa_text_skips_blocks_without_url() {
        let text = "Title: No URL here\nHighlights:\nsome text\n\nURL: https://x.com/t\nTitle: Real One\nHighlights:\nt";
        let results = parse_exa_text(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://x.com/t");
    }

    #[test]
    fn ddg_lite_parses_results_with_snippets() {
        let html = r#"<table>
            <tr><td></td><td><a rel="nofollow" href="https://example.com/a" class='result-link'>Alpha &amp; Beta News</a></td></tr>
            <tr><td></td><td class='result-snippet'>Read the <b>latest</b> on AI &quot;today&quot;.</td></tr>
            <tr><td></td><td><a rel="nofollow" href="https://example.com/b" class='result-link'>Second Result</a></td></tr>
        </table>"#;
        let results = parse_ddg_lite(html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "Alpha & Beta News");
        assert_eq!(results[0].1, "https://example.com/a");
        assert_eq!(results[0].2, "Read the latest on AI \"today\".");
        assert_eq!(results[1].0, "Second Result");
        assert_eq!(results[1].1, "https://example.com/b");
    }

    #[test]
    fn ddg_lite_result_without_snippet_keeps_pairing() {
        let html = r#"<table>
            <tr><td></td><td><a rel="nofollow" href="https://example.com/a" class='result-link'>First</a></td></tr>
            <tr><td></td><td><a rel="nofollow" href="https://example.com/b" class='result-link'>Second</a></td></tr>
            <tr><td></td><td class='result-snippet'>Second's snippet</td></tr>
        </table>"#;
        let results = parse_ddg_lite(html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, "https://example.com/a");
        assert_eq!(results[0].2, "");
        assert_eq!(results[1].2, "Second's snippet");
    }

    #[test]
    fn form_encode_handles_spaces_unicode_and_safe_chars() {
        assert_eq!(form_encode("AI news 2025"), "AI+news+2025");
        assert_eq!(form_encode("a-b.c_d~e"), "a-b.c_d~e");
        assert_eq!(form_encode("café & чай"), "caf%C3%A9+%26+%D1%87%D0%B0%D0%B9");
    }

    #[test]
    fn decode_entities_common_and_numeric() {
        assert_eq!(decode_entities("a &amp; b &lt;c&gt; &quot;d&quot;"), "a & b <c> \"d\"");
        assert_eq!(decode_entities("&#65;&#x42;"), "AB");
        assert_eq!(decode_entities("&bogus; x & nope"), "&bogus; x & nope");
    }

    #[test]
    fn decode_entities_keeps_multibyte_utf8_intact() {
        assert_eq!(decode_entities("مرحبا 👋 &amp; done"), "مرحبا 👋 & done");
    }
}
