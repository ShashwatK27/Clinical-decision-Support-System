import { useRef } from "react";
import useScrollReveal from "../../hooks/useScrollReveal";

export default function Mission() {
  const sectionRef = useRef<HTMLElement>(null);
  useScrollReveal(sectionRef);

  return (
    <section 
      ref={sectionRef}

      style={{
        padding: "var(--section-spacing) 24px",
        background: "radial-gradient(circle at 50% 50%, rgba(47, 216, 213, 0.04) 0%, transparent 60%)",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}
      aria-label="Corporate Mission"
    >
      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          textAlign: "center"
        }}
      >
        <div 
          className="scroll-reveal"
          style={{
            background: "var(--glass-bg)",
            border: "var(--glass-border)",
            borderRadius: "var(--glass-radius-lg)",
            backdropFilter: "var(--glass-blur)",
            boxShadow: "var(--glass-shadow)",
            padding: "80px 40px",
            position: "relative",
            overflow: "hidden"
          }}
        >
          {/* Subtle Inner Glow */}
          <div 
            style={{
              position: "absolute",
              top: "-50%",
              left: "-50%",
              width: "200%",
              height: "200%",
              background: "radial-gradient(circle, rgba(138, 230, 227, 0.02) 0%, transparent 60%)",
              pointerEvents: "none"
            }} 
          />

          <span 
            style={{
              fontSize: "14px",
              fontWeight: 700,
              textTransform: "uppercase",
              color: "var(--accent-primary)",
              letterSpacing: "0.2em",
              display: "block",
              marginBottom: "32px"
            }}
          >
            The CDSS Core Mission
          </span>

          <h2 
            style={{
              fontSize: "var(--font-h2)",
              fontWeight: 600,
              lineHeight: 1.25,
              color: "var(--text-primary)",
              maxWidth: "960px",
              margin: "0 auto 36px",
              letterSpacing: "var(--letter-spacing-tight)"
            }}
          >
            &ldquo;Pioneering the future of clinical decision support by bridging advanced prescription OCR with deep-tech molecular interaction reasoning.&rdquo;
          </h2>

          <p 
            style={{
              fontSize: "var(--font-body)",
              color: "var(--neutral)",
              maxWidth: "680px",
              margin: "0 auto",
              lineHeight: 1.6
            }}
          >
            We are dedicated to establishing next-generation patient safety standards, providing clinical teams with automated scanning and prediction hubs that translate complex drug matrices into high-fidelity, explainable support records.
          </p>

        </div>
      </div>
    </section>
  );
}
