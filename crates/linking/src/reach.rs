//! Can a PC on another network reach this helper?
//!
//! The app asks the router (UPnP IGD) to forward the link port and for its
//! internet-side (WAN) address. No third-party "what is my IP" service is
//! contacted. If the router's WAN address is itself private or in the
//! carrier-grade NAT range (100.64.0.0/10), the internet provider shares one
//! public address between customers (CGNAT) or there is a second router in
//! front, and inbound connections can't work directly.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

use serde::Serialize;

/// UPnP mapping lease; the app renews it while helping.
pub const UPNP_LEASE_SECS: u32 = 2 * 60 * 60;
const UPNP_DESCRIPTION: &str = "Local AI Studio (linked PCs)";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UpnpOutcome {
    /// Not tried (the user didn't ask).
    NotTried,
    /// No UPnP router answered (UPnP off, or not supported).
    NoGateway { error: String },
    Mapped { external_ip: Option<IpAddr>, external_port: u16, gateway: String },
    MapFailed { external_ip: Option<IpAddr>, error: String, gateway: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Other networks can connect to `public_ip:port`.
    Reachable,
    /// Router has a public address but the port isn't forwarded.
    NeedsPortForward,
    /// The router's own internet address is not public: CGNAT or double NAT.
    Cgnat,
    /// Couldn't check (no UPnP); a manual port forward may work.
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReachReport {
    pub local_ip: Option<IpAddr>,
    pub port: u16,
    pub upnp: UpnpOutcome,
    /// The router's WAN address when it is public (what the other PC types).
    pub public_ip: Option<IpAddr>,
    pub verdict: Verdict,
    pub message: String,
    pub steps: Vec<String>,
}

/// 100.64.0.0/10 (RFC 6598, carrier-grade NAT).
pub fn is_cgnat(ip: IpAddr) -> bool {
    matches!(ip, IpAddr::V4(v) if v.octets()[0] == 100 && (v.octets()[1] & 0xC0) == 64)
}

/// An address that only works inside a local network.
pub fn is_lan_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => v.is_private() || v.is_loopback() || v.is_link_local() || is_cgnat(ip),
        IpAddr::V6(v) => {
            v.is_loopback() || (v.segments()[0] & 0xfe00) == 0xfc00 || (v.segments()[0] & 0xffc0) == 0xfe80 || v.to_ipv4_mapped().is_some_and(|m| is_lan_address(IpAddr::V4(m)))
        }
    }
}

/// Not usable from the internet: private, CGNAT, loopback, link-local, unspecified.
pub fn is_non_public(ip: IpAddr) -> bool {
    is_lan_address(ip) || ip.is_unspecified() || matches!(ip, IpAddr::V4(v) if v.is_broadcast() || v.is_documentation())
}

/// This PC's LAN address: the source address the OS would use to reach
/// `toward` (no packet is sent; UDP connect only picks a route).
pub fn local_ip_toward(toward: IpAddr) -> Option<IpAddr> {
    let s = UdpSocket::bind(if toward.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" }).ok()?;
    s.connect(SocketAddr::new(toward, 9)).ok()?;
    let ip = s.local_addr().ok()?.ip();
    (!ip.is_unspecified()).then_some(ip)
}

/// This PC's LAN address on the default route (TEST-NET-1 as the target, so
/// nothing real is contacted).
pub fn local_ip() -> Option<IpAddr> {
    local_ip_toward(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)))
}

/// Decide what to tell the user (pure; tested with fixtures).
pub fn classify(local_ip: Option<IpAddr>, port: u16, upnp: UpnpOutcome) -> ReachReport {
    let lan = local_ip.map(|i| i.to_string()).unwrap_or_else(|| "this PC's local IP (ipconfig)".into());
    let forward_steps = |gateway: Option<&str>| {
        vec![
            format!("Open your router's settings page{}.", gateway.map(|g| format!(" (usually http://{g})")).unwrap_or_default()),
            format!("Forward TCP port {port} to {lan}, port {port}."),
            "Give this PC a fixed local IP there (a DHCP reservation), so the rule keeps working.".to_string(),
            format!("On the other PC, type your router's public IP address and port {port}."),
        ]
    };
    let ext = match &upnp {
        UpnpOutcome::Mapped { external_ip, .. } | UpnpOutcome::MapFailed { external_ip, .. } => *external_ip,
        _ => None,
    };
    let gw_host = |g: &str| g.split(':').next().unwrap_or(g).to_string();
    let (verdict, public_ip, message, steps) = match (&upnp, ext) {
        (_, Some(ip)) if is_non_public(ip) => (
            Verdict::Cgnat,
            None,
            format!(
                "Your router's internet address is {ip}, which is not a public address. Your internet provider probably shares one public address between customers (CGNAT), or there is a second router in front of yours. PCs on other networks can't connect to this one directly. PCs on the same network still can."
            ),
            vec![
                "If there's a second router (for example your provider's modem and your own router), forward the port on both, or put the modem in bridge mode.".to_string(),
                "Otherwise ask your provider for a public IPv4 address, or link from the same network.".to_string(),
            ],
        ),
        (UpnpOutcome::Mapped { external_port, .. }, Some(ip)) => (
            Verdict::Reachable,
            Some(ip),
            format!("The router opened port {external_port} automatically (UPnP). PCs on other networks can link to {}.", SocketAddr::new(ip, *external_port)),
            Vec::new(),
        ),
        (UpnpOutcome::Mapped { external_port, .. }, None) => (
            Verdict::Unknown,
            None,
            format!("The router opened port {external_port} (UPnP) but didn't say its internet address. Use your router's public IP address on the other PC."),
            Vec::new(),
        ),
        (UpnpOutcome::MapFailed { error, gateway, .. }, Some(ip)) => (
            Verdict::NeedsPortForward,
            Some(ip),
            format!("The router refused to open the port automatically ({error}). Forward it by hand; the other PC then types {}.", SocketAddr::new(ip, port)),
            forward_steps(Some(&gw_host(gateway))),
        ),
        (UpnpOutcome::MapFailed { error, gateway, .. }, None) => (
            Verdict::NeedsPortForward,
            None,
            format!("The router refused to open the port automatically ({error}). Forward it by hand."),
            forward_steps(Some(&gw_host(gateway))),
        ),
        (UpnpOutcome::NoGateway { .. }, None) => (
            Verdict::Unknown,
            None,
            "No router answered UPnP (it may be turned off), so the app can't open the port or see your public address without asking an outside service. Forward the port by hand.".to_string(),
            forward_steps(None),
        ),
        (UpnpOutcome::NotTried, _) | (UpnpOutcome::NoGateway { .. }, Some(_)) => (
            Verdict::Unknown,
            None,
            "Same network: the other PC types this PC's local IP. For other networks, check outside access.".to_string(),
            Vec::new(),
        ),
    };
    ReachReport { local_ip, port, upnp, public_ip, verdict, message, steps }
}

/// Ask the router to forward `port` (TCP) to this PC. Takes a few seconds
/// when no router answers.
pub async fn try_upnp(port: u16) -> (Option<IpAddr>, UpnpOutcome) {
    use igd_next::aio::tokio::search_gateway;
    let mut opts = igd_next::SearchOptions::default();
    opts.timeout = Some(Duration::from_secs(4));
    opts.single_search_timeout = Some(Duration::from_secs(2));
    let gw = match search_gateway(opts).await {
        Ok(g) => g,
        Err(e) => return (local_ip(), UpnpOutcome::NoGateway { error: e.to_string() }),
    };
    let gateway = gw.addr.to_string();
    let local = local_ip_toward(gw.addr.ip()).or_else(local_ip);
    let external_ip = gw.get_external_ip().await.ok();
    let Some(IpAddr::V4(lan)) = local else {
        return (local, UpnpOutcome::MapFailed { external_ip, error: "no IPv4 address on this PC".into(), gateway });
    };
    let outcome = match gw.add_port(igd_next::PortMappingProtocol::TCP, port, SocketAddr::V4(SocketAddrV4::new(lan, port)), UPNP_LEASE_SECS, UPNP_DESCRIPTION).await {
        Ok(()) => UpnpOutcome::Mapped { external_ip, external_port: port, gateway },
        Err(e) => UpnpOutcome::MapFailed { external_ip, error: e.to_string(), gateway },
    };
    (local, outcome)
}

/// Remove the mapping made by [`try_upnp`] (best effort).
pub async fn remove_upnp(port: u16) {
    let mut opts = igd_next::SearchOptions::default();
    opts.timeout = Some(Duration::from_secs(3));
    if let Ok(gw) = igd_next::aio::tokio::search_gateway(opts).await {
        let _ = gw.remove_port(igd_next::PortMappingProtocol::TCP, port).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn address_classes() {
        assert!(is_cgnat(ip("100.64.0.1")) && is_cgnat(ip("100.127.255.254")));
        assert!(!is_cgnat(ip("100.128.0.1")) && !is_cgnat(ip("100.63.255.255")));
        for lan in ["10.0.0.5", "172.16.3.4", "192.168.1.20", "127.0.0.1", "169.254.1.1", "fe80::1", "fd00::5", "::1"] {
            assert!(is_lan_address(ip(lan)), "{lan}");
        }
        for public in ["8.8.8.8", "81.2.69.160", "2a00:1450::1"] {
            assert!(!is_non_public(ip(public)), "{public}");
        }
    }

    #[test]
    fn upnp_mapped_with_public_wan_is_reachable() {
        let r = classify(Some(ip("192.168.1.20")), 47070, UpnpOutcome::Mapped { external_ip: Some(ip("81.2.69.160")), external_port: 47070, gateway: "192.168.1.1:5000".into() });
        assert_eq!(r.verdict, Verdict::Reachable);
        assert_eq!(r.public_ip, Some(ip("81.2.69.160")));
        assert!(r.message.contains("81.2.69.160:47070"), "{}", r.message);
    }

    #[test]
    fn private_or_cgnat_wan_means_cgnat() {
        for wan in ["100.72.10.3", "10.20.30.40", "192.168.0.1"] {
            let r = classify(Some(ip("192.168.1.20")), 47070, UpnpOutcome::Mapped { external_ip: Some(ip(wan)), external_port: 47070, gateway: "192.168.1.1:5000".into() });
            assert_eq!(r.verdict, Verdict::Cgnat, "{wan}");
            assert!(r.public_ip.is_none());
            assert!(r.message.contains("CGNAT"));
        }
        let r = classify(None, 47070, UpnpOutcome::MapFailed { external_ip: Some(ip("100.100.1.1")), error: "x".into(), gateway: "g".into() });
        assert_eq!(r.verdict, Verdict::Cgnat);
    }

    #[test]
    fn refused_mapping_or_no_upnp_gives_port_forward_steps() {
        let r = classify(Some(ip("192.168.1.20")), 47070, UpnpOutcome::MapFailed { external_ip: Some(ip("81.2.69.160")), error: "ConflictInMappingEntry".into(), gateway: "192.168.1.1:1900".into() });
        assert_eq!(r.verdict, Verdict::NeedsPortForward);
        assert!(r.steps.iter().any(|s| s.contains("Forward TCP port 47070 to 192.168.1.20")), "{:?}", r.steps);
        assert!(r.steps.iter().any(|s| s.contains("http://192.168.1.1")), "{:?}", r.steps);
        let n = classify(Some(ip("192.168.1.20")), 47070, UpnpOutcome::NoGateway { error: "timeout".into() });
        assert_eq!(n.verdict, Verdict::Unknown);
        assert!(!n.steps.is_empty());
        assert!(n.message.contains("outside service"));
        let not = classify(Some(ip("192.168.1.20")), 47070, UpnpOutcome::NotTried);
        assert_eq!(not.verdict, Verdict::Unknown);
    }
}
