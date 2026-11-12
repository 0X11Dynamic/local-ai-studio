export function bytes(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "n/a";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : digits)} ${units[i]}`;
}

export function rate(bps: number): string {
  return bps > 0 ? `${bytes(bps)}/s` : "";
}

export function percent(a: number, b: number): number {
  return b > 0 ? Math.min(100, Math.max(0, (a / b) * 100)) : 0;
}

export function tokens(n: number | null | undefined): string {
  if (n == null) return "n/a";
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : n.toLocaleString();
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
