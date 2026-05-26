import { useRef } from "react";
import useScrollReveal from "../../hooks/useScrollReveal";

interface Sector {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  symbol: string;
}

const SECTORS: Sector[] = [
  {
    id: "diabetology",
    title: "Diabetology",
    subtitle: "Non-Invasive Intakes",
    description: "Empowering diabetic care by removing needle pain and continuous invasive patches through precise microwave glucose scanning techniques.",
    symbol: "🩸"
  },
  {
    id: "cardiology",
    title: "Cardiology",
    subtitle: "Real-time Hemodynamics",
    description: "Pioneering vital signs and arterial tracking via advanced non-invasive sensors to provide immediate predictive cardiac metrics.",
    symbol: "🫀"
  },
  {
    id: "dentistry",
    title: "Dentistry",
    subtitle: "Subcutaneous Diagnostics",
    description: "Evaluating local dental bone densities and gingival fluid distributions with highly focused sensor probe architectures.",
    symbol: "🦷"
  },
  {
    id: "oncology",
    title: "Oncology",
    subtitle: "Biomarker Surveillance",
    description: "Providing supportive diagnostics and tissue tracking to assist clinical oncology teams in early monitoring phases.",
    symbol: "🔬"
  }
];

export default function IndustryCards() {
  const sectionRef = useRef<HTMLElement>(null);
  useScrollReveal(sectionRef);

  return (
    <section 
      ref={sectionRef}
      id="specialties"
      style={{
        padding: "var(--section-spacing) 24px",
        backgroundColor: "var(--bg-primary)",
        position: "relative"
      }}
      aria-labelledby="sectors-title"
    >
      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          margin: "0 auto"
        }}
      >
        {/* Section Header */}
        <div 
          className="scroll-reveal"
          style={{ 
            textAlign: "center", 
            marginBottom: "64px" 
          }}
        >
          <span 
            style={{
              fontSize: "14px",
              fontWeight: 700,
              textTransform: "uppercase",
              color: "var(--accent-primary)",
              letterSpacing: "0.15em",
              display: "block",
              marginBottom: "16px"
            }}
          >
            Clinical Domains
          </span>
          <h2 
            id="sectors-title"
            style={{
              fontSize: "var(--font-h2)",
              fontWeight: 700,
              color: "var(--text-primary)",
              letterSpacing: "var(--letter-spacing-tight)"
            }}
          >
            Sectors of Medical Application
          </h2>
          <div 
            style={{
              width: "80px",
              height: "2px",
              background: "linear-gradient(90deg, var(--accent-primary) 0%, transparent 100%)",
              margin: "24px auto 0"
            }}
          />
        </div>

        {/* 4-Card Responsive Grid */}
        <div 
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: "24px"
          }}
        >
          {SECTORS.map((sector) => (
            <div 
              key={sector.id}
              className="scroll-reveal hover-premium"
              style={{
                background: "var(--glass-bg)",
                border: "var(--glass-border)",
                borderRadius: "var(--glass-radius)",
                padding: "36px 28px",
                backdropFilter: "var(--glass-blur)",
                boxShadow: "var(--glass-shadow)",
                transition: "all 0.3s ease",
                display: "flex",
                flexDirection: "column"
              }}
              tabIndex={0}
            >
              {/* Symbol Container */}
              <div 
                style={{
                  fontSize: "36px",
                  marginBottom: "24px",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "64px",
                  height: "64px",
                  borderRadius: "16px",
                  background: "rgba(255, 255, 255, 0.02)",
                  border: "1px solid rgba(255,255,255,0.05)"
                }}
                aria-hidden="true"
              >
                {sector.symbol}
              </div>

              {/* Title & Subtitle */}
              <span 
                style={{
                  fontSize: "12px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  color: "var(--accent-primary)",
                  letterSpacing: "0.1em",
                  display: "block",
                  marginBottom: "8px"
                }}
              >
                {sector.subtitle}
              </span>
              <h3 
                style={{
                  fontSize: "22px",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  marginBottom: "16px"
                }}
              >
                {sector.title}
              </h3>

              {/* Description */}
              <p 
                style={{
                  color: "var(--neutral)",
                  fontSize: "15px",
                  lineHeight: 1.6,
                  margin: 0
                }}
              >
                {sector.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
