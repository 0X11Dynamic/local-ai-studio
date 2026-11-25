import { useEffect, useRef } from "react";

/** In-app confirmation (no native confirm: it blocks WebView automation). */
export function ConfirmDelete({ count, title, busy, onConfirm, onCancel }: {
  /** 1 = a single chat (named by `title`); more = the selected chats. */
  count: number; title?: string; busy: boolean; onConfirm: () => void; onCancel: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onCancel(); }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onCancel]);
  const single = count === 1;
  return (
    <div className="confirm-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onCancel(); }}>
      <div className="panel confirm-box" role="alertdialog" aria-modal="true" aria-labelledby="confirm-del-title" aria-describedby="confirm-del-body">
        <strong id="confirm-del-title" style={{ fontWeight: 600 }}>{single ? "Delete this chat?" : `Delete ${count} chats?`}</strong>
        <p id="confirm-del-body" className="dim" style={{ margin: "6px 0 14px" }}>
          {single && title ? <span className="chat-title" style={{ display: "block", marginBottom: 4 }}>{title}</span> : null}
          {single ? "Its messages are removed; attachments go to the app trash." : "Their messages are removed; attachments go to the app trash."}
        </p>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button ref={cancelRef} className="btn sm" onClick={onCancel} disabled={busy}>Cancel</button>
          <button className="btn sm primary" aria-label="Confirm delete chat" onClick={onConfirm} disabled={busy}>{busy ? "Deleting" : "Delete"}</button>
        </div>
      </div>
    </div>
  );
}
