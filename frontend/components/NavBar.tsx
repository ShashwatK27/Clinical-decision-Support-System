import Link from "next/link";

export default function NavBar() {
  return (
    <header className="container" style={{ paddingTop: 20, paddingBottom: 20 }}>
      <nav style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Link href="/" style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "16px" }}>
          <img src="/logo-emblem.png" alt="MediIntel Logo" style={{ height: "70px", width: "auto", objectFit: "contain" }} />
          <span>MediIntel</span>
        </Link>
        <div style={{ display: "flex", gap: 16 }}>
          <Link href="/login" style={{ color: "#2563eb" }}>
            Sign in
          </Link>
          <Link
            href="/register"
            style={{ background: "#2563eb", color: "white", padding: "12px 18px", borderRadius: 14 }}
          >
            Get started
          </Link>
        </div>
      </nav>
    </header>
  );
}
