import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { CaretDown } from "@phosphor-icons/react";

/**
 * A compact button that opens a menu above it (the composer sits at the
 * bottom of the window). Keyboard: Enter, Space, ArrowUp/ArrowDown open it;
 * ArrowUp/ArrowDown/Home/End move between items; Escape closes and returns
 * focus to the button; Tab closes. A click outside closes it.
 *
 * Items are any elements inside the panel with role "menuitem",
 * "menuitemradio" or "menuitemcheckbox".
 */
export function Dropdown({ label, value, title, button, children, className, disabled, panelLabel, width }: {
  /** aria-label of the trigger button (tests use it). */
  label: string;
  /** Written to the button's `value` attribute so `button.value` reads the current choice. */
  value?: string;
  title?: string;
  button: ReactNode;
  /** The panel content; `close` closes the menu (and refocuses the trigger). */
  children: (close: () => void) => ReactNode;
  className?: string;
  disabled?: boolean;
  /** aria-label of the menu panel. */
  panelLabel?: string;
  width?: number;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const id = useId();

  const items = () => [...(panel.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not([aria-disabled="true"])') ?? [])];
  const focusAt = (i: number) => {
    const l = items();
    if (l.length) l[(i + l.length) % l.length].focus();
  };
  const close = useCallback((refocus = true) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  }, []);

  // Focus the checked item (or the first) when the menu opens.
  useEffect(() => {
    if (!open) return;
    const l = items();
    const i = l.findIndex((e) => e.getAttribute("aria-checked") === "true");
    (l[i >= 0 ? i : 0] ?? panel.current)?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const down = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) close(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [open, close]);

  const onPanelKey = (e: React.KeyboardEvent) => {
    const l = items();
    const at = l.indexOf(document.activeElement as HTMLElement);
    if (e.key === "ArrowDown") { e.preventDefault(); focusAt(at + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); focusAt(at < 0 ? l.length - 1 : at - 1); }
    else if (e.key === "Home") { e.preventDefault(); focusAt(0); }
    else if (e.key === "End") { e.preventDefault(); focusAt(l.length - 1); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "Tab") close(false);
  };

  return (
    <div className={`dd${className ? ` ${className}` : ""}`} ref={wrap}>
      <button
        ref={trigger}
        type="button"
        className={`btn sm dd-trigger${open ? " open" : ""}`}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        value={value}
        title={title}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            setOpen(true);
          }
        }}
      >
        {button}
        <CaretDown size={11} weight="bold" aria-hidden className="dd-caret" />
      </button>
      {open && (
        <div ref={panel} id={id} className="dd-panel" role="menu" aria-label={panelLabel ?? label} tabIndex={-1} onKeyDown={onPanelKey} style={width ? { width } : undefined}>
          {children(() => close())}
        </div>
      )}
    </div>
  );
}
