"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function DeleteRunButton({ id }: { id: number }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  return (
    <button
      type="button"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await fetch(`/api/experiments/${id}`, { method: "DELETE" });
        router.refresh();
        setBusy(false);
      }}
      className="rounded-md border border-white/10 px-2 py-1 text-[11px] text-slate-500 transition hover:border-rose-400/40 hover:text-rose-300 disabled:opacity-40"
      aria-label="Delete run"
    >
      {busy ? "…" : "delete"}
    </button>
  );
}
