// One folder row: path box, Browse (Tauri folder dialog), free space on that
// drive and a warning when it is low. Used by first-run setup and Settings.
import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { bytes } from "../lib/format";
import { api8, type FolderCheck } from "../lib/api8";

export function StorageFolder({ id, label, value, onChange, disabled, onCheck }: {
  id: string; label: string; value: string; onChange: (v: string) => void; disabled?: boolean;
  /** Reports the latest validation result (null while unknown). */
  onCheck?: (c: FolderCheck | null) => void;
}) {
  const [check, setCheck] = useState<FolderCheck | null>(null);
  const seq = useRef(0);

  // Re-check (debounced) whenever the path changes.
  useEffect(() => {
    if (!value.trim()) { setCheck(null); onCheck?.(null); return; }
    const n = ++seq.current;
    const t = setTimeout(() => {
      api8.checkStorageFolder(value)
        .then((c) => { if (n === seq.current) { setCheck(c); onCheck?.(c); } })
        .catch(() => {});
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const browse = async () => {
    const d = await open({ directory: true, defaultPath: value || undefined });
    if (typeof d === "string") onChange(d);
  };

  return (
    <div className="field wide" data-storage-folder={id}>
      <label htmlFor={id}>{label}</label>
      <div className="row">
        <input id={id} className="input mono" style={{ flex: 1, minWidth: 0 }} value={value} disabled={disabled}
          onChange={(e) => onChange(e.target.value)} spellCheck={false} />
        <button className="btn sm" disabled={disabled} onClick={browse}>Browse</button>
      </div>
      {check && !check.ok && <div className="danger-text" data-storage-error>{check.error}</div>}
      {check?.ok && (
        <span className="help" data-storage-free>
          {check.free_bytes != null ? `${bytes(check.free_bytes)} free on this drive.` : ""}
          {!check.exists ? " The folder will be created." : ""}
        </span>
      )}
      {check?.ok && check.low_space && (
        <div className="notice warn" data-storage-low>
          Less than 20 GB free on this drive. Models are often 2 to 20 GB each; pick a drive with more room if you can.
        </div>
      )}
    </div>
  );
}
