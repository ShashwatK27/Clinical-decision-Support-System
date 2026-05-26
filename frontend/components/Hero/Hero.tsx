import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import HeroScene from "../../scenes/HeroScene";

export default function Hero() {
  const [isInViewport, setIsInViewport] = useState<boolean>(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: "100px", // Pre-load slightly before scrolling back into view
      threshold: 0.05
    };

    const handleIntersection = (entries: IntersectionObserverEntry[]) => {
      entries.forEach(entry => {
        setIsInViewport(entry.isIntersecting);
      });
    };

    const observer = new IntersectionObserver(handleIntersection, observerOptions);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      if (containerRef.current) {
        observer.unobserve(containerRef.current);
      }
    };
  }, []);

  return (
    <section 
      ref={containerRef}
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "120px 24px 80px",
        overflow: "hidden",
        background: "radial-gradient(circle at 10% 20%, rgba(47, 216, 213, 0.05) 0%, transparent 40%)"
      }}
      aria-label="Introduction Section"
    >
      {/* Glow Effects */}
      <div 
        className="glow-ambient"
        style={{
          position: "absolute",
          top: "20%",
          right: "10%",
          width: "400px",
          height: "400px",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(47, 216, 213, 0.08) 0%, transparent 70%)",
          filter: "blur(60px)",
          pointerEvents: "none",
          zIndex: 1
        }} 
      />

      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "48px",
          alignItems: "center",
          zIndex: 2
        }}
        className="hero-grid"
      >
        {/* Left Column: Heading and Details */}
        <div className="animate-fade-up" style={{ animationDelay: "100ms" }}>
          <div 
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "rgba(47, 216, 213, 0.1)",
              border: "1px solid rgba(47, 216, 213, 0.15)",
              color: "var(--accent-primary)",
              padding: "8px 16px",
              borderRadius: "999px",
              fontSize: "13px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.15em",
              marginBottom: "24px"
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--accent-primary)" }} />
            AI-Powered Prescription Intel
          </div>

          <h1 
            style={{
              fontSize: "var(--font-h1)",
              fontWeight: 800,
              lineHeight: 1.05,
              letterSpacing: "var(--letter-spacing-tight)",
              color: "var(--text-primary)",
              marginBottom: "24px"
            }}
          >
            Clinical Precision. <br />
            <span style={{
              background: "linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-soft) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent"
            }}>Futuristic Care.</span>
          </h1>

          <p 
            style={{
              fontSize: "var(--font-body)",
              lineHeight: 1.6,
              color: "var(--neutral)",
              maxWidth: "540px",
              marginBottom: "40px"
            }}
          >
            Discover the next generation of Symmetrical Clinical Decision Support. Empowering modern healthcare teams with handwritten prescription OCR, molecular interaction maps, and instant clinical diagnostic predictions.
          </p>


          <div style={{ display: "flex", flexWrap: "wrap", gap: "16px" }}>
            <Link 
              href="/dashboard" 
              className="btn-hover"
              style={{
                background: "var(--white)",
                color: "var(--black)",
                padding: "16px 32px",
                borderRadius: "999px",
                fontWeight: 700,
                fontSize: "16px",
                boxShadow: "0 10px 30px rgba(255,255,255,0.06)",
                display: "inline-flex",
                alignItems: "center"
              }}
              aria-label="Launch MediIntel Clinician Predictor Console"
            >
              Get Started Now
            </Link>
            <a 
              href="#products" 
              className="btn-hover"
              style={{
                border: "var(--glass-border)",
                background: "rgba(255,255,255,0.02)",
                color: "var(--text-primary)",
                padding: "16px 32px",
                borderRadius: "999px",
                fontWeight: 700,
                fontSize: "16px",
                backdropFilter: "var(--glass-blur)",
                display: "inline-flex",
                alignItems: "center"
              }}
              aria-label="Learn more about our non-invasive products"
            >
              Explore Products
            </a>
          </div>
        </div>

        {/* Right Column: 3D Visualization & Layered Glassmorphic HUD Overlay */}
        <div 
          className="animate-fade-up"
          style={{ 
            animationDelay: "300ms", 
            position: "relative",
            minHeight: "480px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(10, 16, 32, 0.3)",
            border: "var(--glass-border)",
            borderRadius: "var(--glass-radius-lg)",
            backdropFilter: "var(--glass-blur)",
            boxShadow: "var(--glass-shadow)",
            overflow: "hidden"
          }}
        >
          {/* Background 3D Canvas Layer */}
          <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            {isInViewport ? (
              <HeroScene />
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--neutral)", fontSize: "14px" }}>
                3D View Paused (Performance Restored)
              </div>
            )}
          </div>

          {/* Premium Glassmorphic HUD Card (Floating on Top of 3D Canvas) */}
          <div 
            style={{
              position: "absolute",
              zIndex: 2,
              width: "280px",
              padding: "20px 24px",
              background: "rgba(10, 20, 42, 0.8)",
              border: "1px solid rgba(47, 216, 213, 0.4)",
              borderRadius: "20px",
              color: "#f8fafc",
              fontFamily: "var(--font-sans)",
              fontSize: "12px",
              boxShadow: "inset 0 0 20px rgba(47, 216, 213, 0.15), 0 30px 60px rgba(0,0,0,0.6)",
              display: "flex",
              flexDirection: "column",
              gap: "14px",
              pointerEvents: "none",
              userSelect: "none"
            }}
          >
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(47, 216, 213, 0.2)", paddingBottom: "10px" }}>
              <span style={{ fontWeight: 800, color: "var(--accent-primary)", fontSize: "11px", letterSpacing: "0.05em" }}>MEDIINTEL SECURE RX</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                <span className="pulse-dot" style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-primary)", boxShadow: "0 0 10px var(--accent-primary)" }} />
                <span style={{ fontSize: "9px", fontWeight: 700, color: "var(--accent-primary)" }}>LIVE DATA</span>
              </span>
            </div>

            {/* Details */}
            <div style={{ display: "flex", flexDirection: "column", gap: "4px", color: "var(--neutral)", fontSize: "10px" }}>
              <div><strong style={{ color: "var(--text-primary)" }}>PATIENT:</strong> MS-790-MEDIINTEL</div>
              <div><strong style={{ color: "var(--text-primary)" }}>DIAGNOSIS:</strong> SYSTEM HYBRID VERIFIED</div>
            </div>

            {/* Active Matrix */}
            <div style={{ 
              background: "rgba(18, 18, 20, 0.6)", 
              padding: "8px 10px", 
              borderRadius: "10px", 
              border: "1px dashed rgba(47, 216, 213, 0.2)",
              fontSize: "10px",
              color: "var(--accent-soft)",
              display: "flex",
              flexDirection: "column",
              gap: "4px"
            }}>
              <div>▶ Biosensor Synced:</div>
              <div style={{ color: "#f8fafc", fontStyle: "italic", fontSize: "9px" }}>
                Glucowave active telemetry
              </div>
            </div>

            {/* Footer */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "8.5px", color: "rgba(255,255,255,0.3)" }}>
              <span>SECURE CONSOLE</span>
              <span style={{ color: "var(--accent-soft)", fontWeight: 600 }}>v3.1</span>
            </div>
          </div>

          {/* Floating Badges */}
          <div 
            style={{
              position: "absolute",
              top: "14%",
              left: "8%",
              zIndex: 3,
              padding: "8px 14px",
              background: "rgba(10, 16, 32, 0.8)",
              border: "1px solid var(--accent-primary)",
              borderRadius: "999px",
              color: "var(--text-primary)",
              fontSize: "11px",
              fontWeight: 800,
              boxShadow: "0 10px 20px rgba(0,0,0,0.5), 0 0 10px rgba(47, 216, 213, 0.2)",
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--accent-primary)" }} />
            OCR Intake
          </div>

          <div 
            style={{
              position: "absolute",
              bottom: "14%",
              right: "8%",
              zIndex: 3,
              padding: "8px 14px",
              background: "rgba(10, 16, 32, 0.8)",
              border: "1px solid #fbbf24",
              borderRadius: "999px",
              color: "var(--text-primary)",
              fontSize: "11px",
              fontWeight: 800,
              boxShadow: "0 10px 20px rgba(0,0,0,0.5), 0 0 10px rgba(251, 191, 36, 0.2)",
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#fbbf24" }} />
            DDI Alerts
          </div>
        </div>
      </div>

      <style jsx global>{`
        @media (min-width: 1024px) {
          .hero-grid {
            grid-template-columns: 1.2fr 0.8fr !important;
          }
        }
      `}</style>
    </section>
  );
}
