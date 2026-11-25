//! The built-in `web_fetch` tool: GET one http(s) URL from the app (not the
//! execution environment) and return readable text.
//!
//! - Every hop (the URL and each redirect, at most 5) is checked BEFORE it is
//!   requested: http/https only, no user:password in the URL, and the host is
//!   resolved here and the request pinned to the checked addresses (so a DNS
//!   answer cannot change between the check and the connect). Loopback,
//!   private, link-local, unique-local, CGNAT and other non-public addresses
//!   are "local network": refused unless the user approved that exact host
//!   for this call.
//! - 15 s for the whole fetch, 2 MB download cap, no cookies, no auth
//!   headers, no proxy (a proxy would resolve names itself), a plain
//!   User-Agent.
//! - HTML becomes text: scripts, styles and tags dropped, links kept as
//!   "text (url)". The result is untrusted data like any tool output.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use app_core::{AppError, AppResult};
use reqwest::Url;

pub const USER_AGENT: &str = "LocalAIStudio/0.1 (+web_fetch)";
pub const MAX_REDIRECTS: usize = 5;
pub const TIMEOUT: Duration = Duration::from_secs(15);
pub const MAX_DOWNLOAD: usize = 2 * 1024 * 1024;
/// Text returned to the model (the dispatcher caps tool output at 32 K characters).
pub const MAX_TEXT_CHARS: usize = 30_000;

/// Parse and check the form of a URL (not where it points).
pub fn parse_url(s: &str) -> AppResult<Url> {
    let u = Url::parse(s.trim()).map_err(|e| AppError::InvalidInput(format!("\"{}\" is not a valid URL ({e}); use a full http:// or https:// address", s.trim())))?;
    if !matches!(u.scheme(), "http" | "https") {
        return Err(AppError::PermissionDenied(format!("only http and https URLs can be fetched, not {}:", u.scheme())));
    }
    if !u.username().is_empty() || u.password().is_some() {
        return Err(AppError::PermissionDenied("URLs with a user name or password are not fetched (web_fetch never sends credentials)".into()));
    }
    if u.host_str().is_none_or(str::is_empty) {
        return Err(AppError::InvalidInput("the URL has no host".into()));
    }
    Ok(u)
}

/// Not a public internet address: loopback, private, link-local,
/// unique-local, CGNAT, unspecified, broadcast, multicast, documentation and
/// reserved ranges, and IPv6 forms that embed such an IPv4 address.
pub fn is_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_local_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_local_v4(v4);
            }
            let s = v6.segments();
            // NAT64 (64:ff9b::/96) and IPv4-compatible (::a.b.c.d) embed an IPv4 address.
            if (s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0, 0, 0, 0]) || (s[0..6] == [0, 0, 0, 0, 0, 0] && !v6.is_loopback() && !v6.is_unspecified()) {
                let o = v6.octets();
                return is_local_v4(Ipv4Addr::new(o[12], o[13], o[14], o[15]));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
                || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (s[0] & 0xffc0) == 0xfec0 // fec0::/10 site-local (deprecated)
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
                || v6 == Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)
        }
    }
}

fn is_local_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_private() // 10/8, 172.16/12, 192.168/16
        || ip.is_loopback() // 127/8
        || ip.is_link_local() // 169.254/16
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || o[0] == 0 // 0/8 "this network"
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64/10 shared (CGNAT)
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0/24 protocol assignments
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18/15 benchmarking
        || o[0] >= 240 // 240/4 reserved
}

/// `host:port` of a URL, as approvals name it.
pub fn authority(u: &Url) -> String {
    format!("{}:{}", u.host_str().unwrap_or_default(), u.port_or_known_default().unwrap_or(0))
}

/// Where a URL points, after resolving its host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Public(Vec<SocketAddr>),
    /// A local network address: why, and the addresses.
    Local(String, Vec<SocketAddr>),
}

/// Resolve the host of `u` (DNS for names) and classify it. `localhost` and
/// `*.localhost` are local whatever DNS says; one local address among the
/// answers makes the whole host local.
pub async fn resolve(u: &Url) -> AppResult<Target> {
    let host = u.host_str().unwrap_or_default().trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    let port = u.port_or_known_default().unwrap_or(80);
    let named_local = host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") || host.ends_with(".internal") || host.ends_with(".lan") || host.ends_with(".home.arpa");
    let addrs: Vec<SocketAddr> = match host.parse::<IpAddr>() {
        Ok(ip) => vec![SocketAddr::new(ip, port)],
        // Local by name: no lookup needed to say so (and many don't resolve).
        Err(_) if named_local => match tokio::net::lookup_host((host.as_str(), port)).await {
            Ok(a) => a.collect(),
            Err(_) => return Ok(Target::Local(format!("{host} is a local network name"), vec![])),
        },
        Err(_) => tokio::time::timeout(Duration::from_secs(8), tokio::net::lookup_host((host.as_str(), port)))
            .await
            .map_err(|_| AppError::Runtime(format!("looking up {host} timed out")))?
            .map_err(|e| AppError::NotFound(format!("could not resolve {host}: {e}")))?
            .collect(),
    };
    if addrs.is_empty() {
        return Err(AppError::NotFound(format!("{host} has no addresses")));
    }
    let local: Vec<String> = addrs.iter().filter(|a| is_local_ip(a.ip())).map(|a| a.ip().to_string()).collect();
    if named_local || !local.is_empty() {
        let shown = if local.is_empty() { addrs.iter().map(|a| a.ip().to_string()).collect::<Vec<_>>().join(", ") } else { local.join(", ") };
        return Ok(Target::Local(format!("{host} is a local network address ({shown})"), addrs));
    }
    Ok(Target::Public(addrs))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fetched {
    pub status: u16,
    pub final_url: String,
    pub content_type: String,
    pub text: String,
    /// The body was cut at the download cap or the text at MAX_TEXT_CHARS.
    pub truncated: bool,
    pub bytes: usize,
    pub redirects: usize,
}

impl Fetched {
    /// The tool result: a short header, then the text.
    pub fn render(&self) -> String {
        format!(
            "status: {}\nfinal url: {}\ncontent-type: {}\nbytes: {}{}{}\n\n{}",
            self.status,
            self.final_url,
            if self.content_type.is_empty() { "(none)" } else { &self.content_type },
            self.bytes,
            if self.redirects > 0 { format!(" (after {} redirect{})", self.redirects, if self.redirects == 1 { "" } else { "s" }) } else { String::new() },
            if self.truncated { ", truncated" } else { "" },
            self.text
        )
    }
}

/// Fetch `url`. `approved_local` lists the `host:port` authorities the user
/// approved although they are local network addresses (empty: public only).
pub async fn fetch(url: &str, approved_local: &[String]) -> AppResult<Fetched> {
    match tokio::time::timeout(TIMEOUT, fetch_inner(url, approved_local)).await {
        Ok(r) => r,
        Err(_) => Err(AppError::Runtime(format!("the page did not load within {} s", TIMEOUT.as_secs()))),
    }
}

async fn fetch_inner(url: &str, approved_local: &[String]) -> AppResult<Fetched> {
    let mut u = parse_url(url)?;
    let mut redirects = 0;
    loop {
        let addrs = match resolve(&u).await? {
            Target::Public(a) => a,
            Target::Local(why, a) if approved_local.iter().any(|x| x.eq_ignore_ascii_case(&authority(&u))) => {
                let _ = why;
                a
            }
            Target::Local(why, _) => {
                let hop = if redirects > 0 { "the page redirected to a local network address" } else { "local network address" };
                return Err(AppError::PermissionDenied(format!("{hop}: {why}; not fetched")));
            }
        };
        let host = u.host_str().unwrap_or_default().to_string();
        let mut b = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .user_agent(USER_AGENT)
            .timeout(TIMEOUT)
            .connect_timeout(Duration::from_secs(8));
        // Pin the connection to the addresses just checked.
        if host.parse::<IpAddr>().is_err() && !host.starts_with('[') {
            b = b.resolve_to_addrs(&host, &addrs);
        }
        let client = b.build().map_err(|e| AppError::Runtime(format!("could not set up the request: {e}")))?;
        let resp = client
            .get(u.clone())
            .header(reqwest::header::ACCEPT, "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5")
            .send()
            .await
            .map_err(|e| AppError::Runtime(format!("could not fetch {u}: {}", without_url(e))))?;
        let status = resp.status();
        if status.is_redirection()
            && let Some(loc) = resp.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok())
        {
            if redirects >= MAX_REDIRECTS {
                return Err(AppError::Runtime(format!("more than {MAX_REDIRECTS} redirects; stopped at {u}")));
            }
            let next = u.join(loc).map_err(|e| AppError::Runtime(format!("bad redirect to \"{loc}\": {e}")))?;
            u = parse_url(next.as_str())?;
            redirects += 1;
            continue;
        }
        let content_type = resp.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
        let (body, cut) = read_capped(resp).await?;
        let ct = content_type.to_ascii_lowercase();
        let looks_html = ct.contains("html") || (ct.is_empty() && sniff_html(&body));
        let texty = looks_html || ct.starts_with("text/") || ct.contains("json") || ct.contains("xml") || ct.contains("javascript") || ct.contains("csv") || ct.contains("yaml") || ct.is_empty();
        let mut text = if !texty || body.iter().take(4096).any(|&b| b == 0) {
            format!("[binary content ({}), {} bytes; not shown]", if ct.is_empty() { "unknown type" } else { &content_type }, body.len())
        } else {
            let raw = String::from_utf8_lossy(&body);
            if looks_html { html_to_text(&raw, &u) } else { raw.into_owned() }
        };
        let mut truncated = cut;
        if text.chars().count() > MAX_TEXT_CHARS {
            text = text.chars().take(MAX_TEXT_CHARS).collect::<String>() + "\n[text truncated]";
            truncated = true;
        }
        return Ok(Fetched { status: status.as_u16(), final_url: u.to_string(), content_type, text, truncated, bytes: body.len(), redirects });
    }
}

/// reqwest errors repeat the URL; keep the cause only.
fn without_url(e: reqwest::Error) -> String {
    let e = e.without_url();
    let mut s = e.to_string();
    let mut src = std::error::Error::source(&e);
    while let Some(c) = src {
        s.push_str(&format!(": {c}"));
        src = c.source();
    }
    s
}

async fn read_capped(mut resp: reqwest::Response) -> AppResult<(Vec<u8>, bool)> {
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| AppError::Runtime(format!("the download failed: {}", without_url(e))))? {
        let room = MAX_DOWNLOAD - body.len();
        if chunk.len() >= room {
            body.extend_from_slice(&chunk[..room]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body, false))
}

fn sniff_html(b: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&b[..b.len().min(512)]).to_ascii_lowercase();
    head.contains("<!doctype html") || head.contains("<html")
}

// ---------- HTML to text ----------

const DROP: [&str; 8] = ["script", "style", "noscript", "template", "svg", "iframe", "object", "canvas"];
const BLOCK: [&str; 33] = [
    "p", "div", "br", "tr", "section", "article", "header", "footer", "nav", "main", "aside", "ul", "ol", "table", "thead", "tbody", "pre", "blockquote", "hr", "form", "dl", "dt", "dd", "figure",
    "figcaption", "address", "details", "summary", "fieldset", "legend", "caption", "body", "html",
];

/// Readable text from HTML: the title, then the text with block breaks, list
/// bullets, headings marked with `#`, and links as "text (url)" (relative
/// links made absolute against `base`).
pub fn html_to_text(html: &str, base: &Url) -> String {
    let mut out = String::new();
    let mut title = String::new();
    let mut in_title = false;
    let mut pre = 0usize;
    let mut links: Vec<(Option<String>, usize)> = vec![];
    let b = html.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'<' {
            let next = html[i..].find('<').map(|n| i + n).unwrap_or(b.len());
            let text = decode_entities(&html[i..next]);
            if in_title {
                title.push_str(&text);
            } else {
                push_text(&mut out, &text, pre > 0);
            }
            i = next;
            continue;
        }
        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map(|n| i + n + 3).unwrap_or(b.len());
            continue;
        }
        let end = tag_end(b, i);
        let inner = &html[i + 1..end.saturating_sub(1).max(i + 1)];
        i = end;
        let closing = inner.starts_with('/');
        let body = inner.trim_start_matches('/');
        let name: String = body.chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        if name.is_empty() {
            continue; // <!doctype>, <?xml?>, stray '<'
        }
        if !closing && DROP.contains(&name.as_str()) {
            // Skip everything up to the matching close tag.
            let close = format!("</{name}");
            i = find_ci(html, i, &close).map(|n| tag_end(b, n)).unwrap_or(b.len());
            continue;
        }
        match (name.as_str(), closing) {
            ("title", false) => in_title = true,
            ("title", true) => in_title = false,
            ("head", true) => in_title = false,
            ("a", false) => {
                let href = attr(body, "href").map(|h| decode_entities(&h)).filter(|h| !h.trim().is_empty() && !h.trim_start().starts_with('#') && !h.trim_start().to_ascii_lowercase().starts_with("javascript:"));
                let abs = href.map(|h| base.join(h.trim()).map(|u| u.to_string()).unwrap_or(h));
                links.push((abs, out.len()));
            }
            ("a", true) => {
                if let Some((Some(url), start)) = links.pop() {
                    let label = out.get(start..).unwrap_or("").trim().to_string();
                    if !label.is_empty() && label != url {
                        while out.ends_with(' ') {
                            out.pop();
                        }
                        out.push_str(&format!(" ({url})"));
                    } else if label.is_empty() {
                        push_text(&mut out, &url, false);
                    }
                }
            }
            ("li", false) => {
                newline(&mut out, 1);
                out.push_str("- ");
            }
            ("img", false) => {
                if let Some(alt) = attr(body, "alt").map(|a| decode_entities(&a)).filter(|a| !a.trim().is_empty()) {
                    push_text(&mut out, &format!("[image: {}]", alt.trim()), false);
                }
            }
            ("td" | "th", _) => push_text(&mut out, " ", false),
            (h, false) if h.len() == 2 && h.starts_with('h') && h.as_bytes()[1].is_ascii_digit() => {
                newline(&mut out, 2);
                out.push_str(&"#".repeat(((h.as_bytes()[1] - b'0') as usize).clamp(1, 6)));
                out.push(' ');
            }
            (h, true) if h.len() == 2 && h.starts_with('h') && h.as_bytes()[1].is_ascii_digit() => newline(&mut out, 2),
            ("pre", false) => {
                pre += 1;
                newline(&mut out, 1);
            }
            ("pre", true) => {
                pre = pre.saturating_sub(1);
                newline(&mut out, 1);
            }
            (n, _) if BLOCK.contains(&n) => newline(&mut out, if n == "p" { 2 } else { 1 }),
            _ => {}
        }
    }
    let body = tidy(&out);
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    if title.is_empty() { body } else { format!("Title: {title}\n\n{body}") }
}

/// Index just past the `>` that ends the tag starting at `start` (quotes
/// in attribute values may contain `>`).
pub(crate) fn tag_end(b: &[u8], start: usize) -> usize {
    let mut q: Option<u8> = None;
    let mut j = start + 1;
    while j < b.len() {
        match (q, b[j]) {
            (None, b'"' | b'\'') => q = Some(b[j]),
            (Some(c), x) if x == c => q = None,
            (None, b'>') => return j + 1,
            _ => {}
        }
        j += 1;
    }
    b.len()
}

fn find_ci(hay: &str, from: usize, needle: &str) -> Option<usize> {
    let lower = hay.get(from..)?.to_ascii_lowercase();
    lower.find(needle).map(|n| from + n)
}

/// The value of attribute `name` in a tag's inside (`a href="x" ...`).
pub(crate) fn attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(pos) = lower[from..].find(name).map(|p| p + from) {
        let before_ok = pos == 0 || lower.as_bytes()[pos - 1].is_ascii_whitespace();
        let rest = lower[pos + name.len()..].trim_start();
        if before_ok && rest.starts_with('=') {
            let vstart = tag.len() - rest.len() + 1;
            let v = tag[vstart..].trim_start();
            return Some(match v.chars().next() {
                Some(q @ ('"' | '\'')) => v[1..].split(q).next().unwrap_or("").to_string(),
                _ => v.split(|c: char| c.is_whitespace() || c == '>').next().unwrap_or("").to_string(),
            });
        }
        from = pos + name.len();
    }
    None
}

fn push_text(out: &mut String, text: &str, keep_space: bool) {
    if keep_space {
        out.push_str(text);
        return;
    }
    if text.is_empty() {
        return;
    }
    let space = |out: &mut String| {
        if !out.is_empty() && !out.ends_with([' ', '\n']) {
            out.push(' ');
        }
    };
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        space(out);
        return;
    }
    if text.starts_with(char::is_whitespace) {
        space(out);
    }
    out.push_str(&words.join(" "));
    if text.ends_with(char::is_whitespace) {
        out.push(' ');
    }
}

fn newline(out: &mut String, n: usize) {
    while out.ends_with(' ') {
        out.pop();
    }
    let have = out.chars().rev().take_while(|&c| c == '\n').count();
    for _ in have..n {
        out.push('\n');
    }
}

fn tidy(s: &str) -> String {
    let mut out = String::new();
    let mut blank = 0;
    for line in s.lines() {
        let l = line.trim_end();
        if l.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
            out.push('\n');
        } else {
            blank = 0;
            out.push_str(l);
            out.push('\n');
        }
    }
    out.trim_matches('\n').to_string()
}

/// `&amp;`, `&lt;`, `&#39;`, `&#x27;` and the common named entities.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let end = rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).map(|n| n + 1);
        let (ent, has_semi) = match end {
            Some(e) if rest[e..].starts_with(';') => (&rest[1..e], true),
            _ => ("", false),
        };
        let ch = if !has_semi {
            None
        } else if let Some(num) = ent.strip_prefix('#') {
            let v = if let Some(h) = num.strip_prefix(['x', 'X']) { u32::from_str_radix(h, 16).ok() } else { num.parse().ok() };
            v.and_then(char::from_u32).map(|c| c.to_string())
        } else {
            match ent {
                "amp" => Some("&"),
                "lt" => Some("<"),
                "gt" => Some(">"),
                "quot" => Some("\""),
                "apos" => Some("'"),
                "nbsp" => Some(" "),
                "ndash" => Some("-"),
                "mdash" => Some("-"),
                "hellip" => Some("..."),
                "copy" => Some("(c)"),
                "reg" => Some("(R)"),
                "trade" => Some("(TM)"),
                "laquo" => Some("<<"),
                "raquo" => Some(">>"),
                "lsquo" | "rsquo" => Some("'"),
                "ldquo" | "rdquo" => Some("\""),
                "bull" | "middot" => Some("*"),
                _ => None,
            }
            .map(str::to_string)
        };
        match ch {
            Some(c) => {
                out.push_str(&c);
                rest = &rest[ent.len() + 2..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn local_and_private_addresses_are_recognised() {
        for l in [
            "127.0.0.1", "127.255.0.9", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.10", "169.254.169.254", "0.0.0.0", "100.64.0.1", "255.255.255.255", "224.0.0.1", "::1", "::",
            "fc00::1", "fd12:3456::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254", "64:ff9b::a9fe:a9fe", "::7f00:1", "2001:db8::1",
        ] {
            assert!(is_local_ip(ip(l)), "{l} must be local");
        }
        for p in ["8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1", "192.169.0.1", "100.128.0.1", "2606:4700:4700::1111", "2a00:1450:4001::200e", "::ffff:8.8.8.8"] {
            assert!(!is_local_ip(ip(p)), "{p} is public");
        }
    }

    #[test]
    fn url_forms_are_checked() {
        assert!(parse_url("https://example.com/a?b=1").is_ok());
        assert!(parse_url("ftp://example.com/").is_err());
        assert!(parse_url("file:///C:/Windows/win.ini").is_err());
        assert!(parse_url("https://user:pw@example.com/").is_err(), "no credentials in URLs");
        assert!(parse_url("not a url").is_err());
        assert!(parse_url("javascript:alert(1)").is_err());
        // Odd IPv4 spellings are normalised by the URL parser, so the IP check sees them.
        assert_eq!(parse_url("http://2130706433/").unwrap().host_str(), Some("127.0.0.1"));
        assert_eq!(parse_url("http://0x7f.1/").unwrap().host_str(), Some("127.0.0.1"));
        assert_eq!(authority(&parse_url("http://Example.com/x").unwrap()), "example.com:80");
        assert_eq!(authority(&parse_url("https://127.0.0.1:1234/v1").unwrap()), "127.0.0.1:1234");
    }

    #[tokio::test]
    async fn resolving_marks_local_hosts_before_any_request() {
        for u in ["http://127.0.0.1:1234/api/status", "http://localhost/", "http://foo.localhost/", "http://[::1]:8080/", "http://2130706433/", "http://169.254.169.254/latest/meta-data", "http://192.168.0.1/"] {
            let t = resolve(&parse_url(u).unwrap()).await.unwrap();
            assert!(matches!(t, Target::Local(..)), "{u}: {t:?}");
        }
        assert!(matches!(resolve(&parse_url("http://8.8.8.8/").unwrap()).await.unwrap(), Target::Public(_)));
    }

    #[tokio::test]
    async fn local_addresses_are_refused_unless_approved() {
        // A local server that would answer if the request were made.
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = l.accept().await else { break };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 2048];
                    let n = s.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let (status, extra, body) = if req.starts_with("GET /hop") {
                        ("302 Found", "Location: http://127.0.0.2:9/\r\n".to_string(), String::new())
                    } else {
                        ("200 OK", "Content-Type: text/html; charset=utf-8\r\n".to_string(), format!("<html><head><title>T</title></head><body><p>hi <a href=\"/x\">there</a></p><pre>{}</pre></body></html>", req.lines().find(|l| l.to_ascii_lowercase().starts_with("user-agent")).unwrap_or("")))
                    };
                    let _ = s.write_all(format!("HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
                });
            }
        });
        let url = format!("http://127.0.0.1:{port}/page");
        let e = fetch(&url, &[]).await.unwrap_err().to_string();
        assert!(e.contains("local network address"), "{e}");
        let ok = fetch(&url, &[format!("127.0.0.1:{port}")]).await.unwrap();
        assert_eq!(ok.status, 200);
        assert!(ok.text.contains("Title: T") && ok.text.contains(&format!("there (http://127.0.0.1:{port}/x)")), "{}", ok.text);
        assert!(ok.text.contains(USER_AGENT), "plain user agent sent: {}", ok.text);
        assert!(ok.render().starts_with("status: 200\nfinal url: "));
        // An approved host that redirects to another local address: not followed.
        let e = fetch(&format!("http://127.0.0.1:{port}/hop"), &[format!("127.0.0.1:{port}")]).await.unwrap_err().to_string();
        assert!(e.contains("redirected to a local network address"), "{e}");
    }

    #[test]
    fn html_becomes_readable_text_with_links() {
        let base = Url::parse("https://example.com/docs/page.html").unwrap();
        let html = r##"<!DOCTYPE html><html><head><title> My  Page </title><style>body { color: red }</style>
            <script>var secret = "<p>not text</p>";</script></head>
            <body><!-- a comment --><h1>Heading &amp; more</h1>
            <p>First   paragraph with <a href="/about">a link</a> and <a href='other.html'>relative</a>.</p>
            <ul><li>one</li><li>two &lt;b&gt;</li></ul>
            <p>Entities: &quot;q&quot; &#39;s&#39; &#x263A; &nbsp;end</p>
            <a href="javascript:void(0)">js</a> <a href="#top">top</a> <a href="https://example.com/x">https://example.com/x</a>
            <noscript>enable js</noscript><img src="a.png" alt="A cat"><SCRIPT type="x">ALSO HIDDEN</SCRIPT></body></html>"##;
        let t = html_to_text(html, &base);
        assert!(t.starts_with("Title: My Page"), "{t}");
        assert!(t.contains("# Heading & more"), "{t}");
        assert!(t.contains("First paragraph with a link (https://example.com/about) and relative (https://example.com/docs/other.html)."), "{t}");
        assert!(t.contains("- one\n- two <b>"), "{t}");
        assert!(t.contains("Entities: \"q\" 's' \u{263A}"), "{t}");
        assert!(t.contains("[image: A cat]"), "{t}");
        assert!(t.contains("js top https://example.com/x") && !t.contains("javascript:") && !t.contains("(#top)"), "{t}");
        for hidden in ["color: red", "secret", "not text", "a comment", "enable js", "ALSO HIDDEN"] {
            assert!(!t.contains(hidden), "{hidden} leaked: {t}");
        }
    }

    #[test]
    fn entities_and_odd_markup_do_not_break_the_converter() {
        assert_eq!(decode_entities("a &amp; b &unknown; &#60; & c"), "a & b &unknown; < & c");
        let base = Url::parse("https://e.com/").unwrap();
        assert_eq!(html_to_text("<p title=\"a > b\">text</p>", &base), "text");
        assert_eq!(html_to_text("unclosed <b", &base), "unclosed");
        assert_eq!(html_to_text("<script>never closed", &base), "");
        assert!(html_to_text("<pre>  keep\n   spacing</pre>", &base).contains("  keep\n   spacing"));
    }
}
